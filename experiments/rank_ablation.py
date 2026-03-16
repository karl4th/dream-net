"""
Rank Ablation Study
===================
Tests fast-weight matrix rank r ∈ {2, 4, 8, 16, 32, 64} on the
A → B → C speaker-switch stress test.

For each rank we measure:
  • Mean prediction loss per speaker segment (A / B / C)
  • Recovery speed after each switch (frames until loss ≤ 1.5 × baseline)
  • Fast-weight memory footprint  (bytes, float32)
  • Peak |U| norm after each segment

This tells us the minimum rank that preserves adaptation quality,
i.e., the sweet spot between capacity and efficiency.
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from dream_net import DREAMConfig, DREAMCell, DREAMState


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SPEAKERS = {
    "A": "data/commonvoice/speaker_1/0000.wav",
    "B": "data/commonvoice/speaker_2/0000.wav",
    "C": "data/commonvoice/speaker_3/0000.wav",
}
CLIP_SEC = 5.0
SR       = 16_000
N_MELS   = 80
N_FFT    = 1024
HOP      = 160
WIN      = 400
FPS      = SR / HOP

PRETRAIN_EPOCHS    = 30
PRETRAIN_LR        = 3e-3
PRETRAIN_CLIP_GRAD = 1.0

RANKS = [2, 4, 8, 16, 32, 64]


# ---------------------------------------------------------------------------
# Audio helpers  (identical to stress-test)
# ---------------------------------------------------------------------------

def load_clip(path, n_sec):
    wav, sr = torchaudio.load(path)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    n = int(n_sec * SR)
    return wav[:, :n] if wav.shape[1] >= n else F.pad(wav, (0, n - wav.shape[1]))


def build_mel(wav):
    mel_tf = T.MelSpectrogram(
        sample_rate=SR, n_fft=N_FFT, win_length=WIN, hop_length=HOP,
        n_mels=N_MELS, f_min=20.0, f_max=8000.0, power=2.0,
    )
    logm = torch.log(mel_tf(wav).squeeze(0).T + 1e-6)
    mu   = logm.mean(0, keepdim=True)
    std  = logm.std(0, keepdim=True).clamp(min=1e-4)
    return (logm - mu) / std


def make_features():
    clips     = {k: load_clip(v, CLIP_SEC) for k, v in SPEAKERS.items()}
    feats_a   = build_mel(clips["A"])
    feats_abc = build_mel(torch.cat([clips["A"], clips["B"], clips["C"]], dim=1))
    sw        = int(CLIP_SEC * SR / HOP)
    return feats_a, feats_abc, [sw, 2 * sw]


# ---------------------------------------------------------------------------
# Cell
# ---------------------------------------------------------------------------

class AblationCell(DREAMCell):
    def __init__(self, config):
        super().__init__(config)

    def forward_step(self, x, state):
        x_scale = x.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        x_norm  = (x / x_scale).clamp(-1.0, 1.0)

        x_pred = torch.tanh(state.h @ self.C.T) * x_scale

        # fast-weight prediction correction
        h_U             = torch.bmm(state.h.unsqueeze(1), state.U).squeeze(1)
        pred_correction = (h_U @ self.V.T) * x_scale
        x_pred          = x_pred + 0.5 * pred_correction

        error          = x - x_pred
        error_norm     = error.norm(dim=-1)
        rel_error_norm = (error_norm / x_scale.squeeze(1)).clamp(max=4.0)

        surprise = self.surprise_gate(error, rel_error_norm, state)
        self.update_fast_weights(state.h, error, surprise, state)

        base_effect  = (self.B @ x_norm.T).T
        err_effect   = (self.W @ error.T).T
        input_effect = (
            state.h    * 0.6 +
            base_effect  * 0.2 +
            err_effect   * surprise.unsqueeze(1) * 0.3
        )
        h_new = self.compute_ltc_update(state.h, input_effect, surprise)

        a = 0.05
        state.error_mean   = (1 - a) * state.error_mean   + a * error
        state.error_var    = (1 - a) * state.error_var    + a * (error - state.error_mean) ** 2
        state.avg_surprise = (1 - self.beta_s) * state.avg_surprise + self.beta_s * surprise

        if state.avg_surprise.mean() < self.S_min:
            dU_t = self.sleep_rate * (state.U - state.U_target)
            state.U_target = state.U_target + dU_t
            nt = state.U_target.norm(dim=(1, 2), keepdim=True)
            state.U_target *= (self.target_norm / (nt + 1e-6)).clamp(max=1.5)

        state.h = h_new
        return h_new, state, surprise, rel_error_norm


# ---------------------------------------------------------------------------
# Pre-train  (same as other experiments)
# ---------------------------------------------------------------------------

def pretrain(config, feats_a):
    cell      = AblationCell(config)
    trainable = [cell.C, cell.W, cell.B, cell.tau_sys, cell.ltc_surprise_scale]
    opt       = torch.optim.Adam(trainable, lr=PRETRAIN_LR)

    for epoch in range(PRETRAIN_EPOCHS):
        state = cell.init_state(1)
        for t in range(feats_a.shape[0]):
            x_t           = feats_a[t].unsqueeze(0)
            x_norm_target = (x_t / (x_t.norm() + 1e-6)).clamp(-1, 1)
            loss          = F.mse_loss(torch.tanh(state.h.detach() @ cell.C.T), x_norm_target)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(trainable, PRETRAIN_CLIP_GRAD)
            opt.step()
            with torch.no_grad():
                _, state, _, _ = cell.forward_step(x_t, state)
    return cell


def copy_base_weights(src: AblationCell, dst: AblationCell):
    with torch.no_grad():
        dst.C.copy_(src.C)
        dst.W.copy_(src.W)
        dst.B.copy_(src.B)
        dst.tau_sys.copy_(src.tau_sys)
        dst.ltc_surprise_scale.copy_(src.ltc_surprise_scale)
        dst.eta.copy_(src.eta)


# ---------------------------------------------------------------------------
# Inference for one rank
# ---------------------------------------------------------------------------

def run_rank(rank, pretrained_r16, feats_a, feats_abc, switches, base_config):
    """
    Build a fresh cell with the given rank, copy pre-trained base weights
    (C, W, B come from rank-16 pre-training; V is re-initialised for new rank),
    run inference, return metrics dict.
    """
    config = DREAMConfig(
        input_dim=N_MELS,
        hidden_dim=256,
        rank=rank,
        base_threshold=0.35,
        base_plasticity=0.4,
        forgetting_rate=0.03,
        ltc_tau_sys=5.0,
        ltc_surprise_scale=8.0,
        surprise_temperature=0.12,
        entropy_influence=0.2,
        time_step=0.1,
        sleep_rate=0.005,
        min_surprise_for_sleep=0.25,
        adaptive_forgetting_scale=8.0,
    )

    # Re-pretrain with this rank so C is calibrated for it
    cell = pretrain(config, feats_a)
    cell.eval()

    losses, surprises, u_norms = [], [], []
    t0 = time.perf_counter()

    with torch.no_grad():
        state = cell.init_state(1)
        for t in range(feats_abc.shape[0]):
            x_t = feats_abc[t].unsqueeze(0)
            _, state, surprise, rel_err = cell.forward_step(x_t, state)
            losses.append(rel_err.item())
            surprises.append(surprise.item())
            u_norms.append(state.U.norm().item())

    elapsed = time.perf_counter() - t0
    losses    = np.array(losses)
    sw1, sw2  = switches

    # memory footprint of U (float32, per batch element)
    u_bytes = 256 * rank * 4  # hidden × rank × 4 bytes

    def recovery(arr, sw, thresh):
        after = arr[sw:]
        r = next((i for i, v in enumerate(after) if v <= thresh), None)
        return r

    baseline  = losses[sw1 - 100:sw1].mean()
    thresh    = baseline * 1.5

    return {
        "rank":       rank,
        "loss_A":     losses[:sw1].mean(),
        "loss_B":     losses[sw1:sw2].mean(),
        "loss_C":     losses[sw2:].mean(),
        "loss_all":   losses.mean(),
        "rec_sw1":    recovery(losses, sw1, thresh),
        "rec_sw2":    recovery(losses, sw2, thresh),
        "u_bytes":    u_bytes,
        "u_norm_A":   np.array(u_norms[:sw1]).mean(),
        "u_norm_B":   np.array(u_norms[sw1:sw2]).mean(),
        "u_norm_C":   np.array(u_norms[sw2:]).mean(),
        "ms_per_frame": elapsed / feats_abc.shape[0] * 1000,
        "losses":     losses,
        "surprises":  np.array(surprises),
        "u_norms":    np.array(u_norms),
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def smooth(x, w=12):
    return np.convolve(x, np.ones(w) / w, mode="same")


def plot_ablation(all_results, switches, out="rank_ablation_results.png"):
    T   = len(all_results[0]["losses"])
    t   = np.arange(T) / FPS
    sw  = [s / FPS for s in switches]

    cmap   = plt.cm.viridis
    n      = len(all_results)
    colors = [cmap(i / (n - 1)) for i in range(n)]
    ranks  = [r["rank"]    for r in all_results]
    loss_a = [r["loss_A"]  for r in all_results]
    loss_b = [r["loss_B"]  for r in all_results]
    loss_c = [r["loss_C"]  for r in all_results]
    mem_kb = [r["u_bytes"] / 1024 for r in all_results]
    # mean loss over whole clip — overall efficiency
    loss_all = [r["loss_all"] for r in all_results]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Rank Ablation Study  —  Fast-Weight Matrix U",
                 fontsize=14, fontweight="bold")

    # ── Panel (0,0..2): loss curves, spans full top row ──────────────────
    ax_loss = fig.add_subplot(2, 1, 1)
    for ax in axes[0]:           # hide the three individual top axes
        ax.set_visible(False)

    regions = [(0, sw[0], "Speaker A", "#2196F3"),
               (sw[0], sw[1], "Speaker B", "#E91E63"),
               (sw[1], t[-1], "Speaker C", "#4CAF50")]
    for res, col in zip(all_results, colors):
        ax_loss.plot(t, smooth(res["losses"]), color=col,
                     label=f"rank {res['rank']}", linewidth=1.9, alpha=0.9)
    for s in sw:
        ax_loss.axvline(s, color="black", linestyle="--", linewidth=1.1)
    for x0, x1, lbl, col in regions:
        ax_loss.axvspan(x0, x1, alpha=0.06, color=col)
        ax_loss.text((x0 + x1) / 2, 3.65, lbl, ha="center",
                     fontsize=10, color=col, fontweight="bold", alpha=0.85)
    ax_loss.set_ylabel("Relative Prediction Error", fontsize=10)
    ax_loss.set_xlabel("Time  (seconds)", fontsize=10)
    ax_loss.set_title("Prediction Loss over Time by Rank  —  A → B → C",
                      fontsize=11, fontweight="bold", pad=8)
    ax_loss.legend(fontsize=9, ncol=n, loc="upper center",
                   bbox_to_anchor=(0.5, -0.14), framealpha=0.8)
    ax_loss.grid(True, alpha=0.2)
    ax_loss.set_ylim(0, 4.0)

    bar_kw  = dict(edgecolor="white", linewidth=0.5)
    x       = np.arange(n)
    xlabels = [f"r={r}" for r in ranks]

    def annotate_bars(ax, bars, vals, fmt="{:.2f}", dy=0.01):
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + dy, fmt.format(v),
                    ha="center", va="bottom", fontsize=8)

    # ── Panel (1,0): loss per speaker segment ────────────────────────────
    ax = axes[1, 0]
    w  = 0.26
    b1 = ax.bar(x - w, loss_a, w, label="Speaker A", color="#90CAF9", **bar_kw)
    b2 = ax.bar(x,     loss_b, w, label="Speaker B", color="#F48FB1", **bar_kw)
    b3 = ax.bar(x + w, loss_c, w, label="Speaker C", color="#A5D6A7", **bar_kw)
    ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_ylabel("Mean Loss", fontsize=9)
    ax.set_title("Loss per Speaker Segment", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

    # ── Panel (1,1): overall mean loss (quality/efficiency tradeoff) ──────
    ax = axes[1, 1]
    bars = ax.bar(x, loss_all, color=colors, **bar_kw)
    ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_ylabel("Mean Loss  (all 15 s)", fontsize=9)
    ax.set_title("Overall Adaptation Quality\n(lower = better)", fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)
    annotate_bars(ax, bars, loss_all)

    # ── Panel (1,2): memory cost ──────────────────────────────────────────
    ax = axes[1, 2]
    bars = ax.bar(x, mem_kb, color=colors, **bar_kw)
    ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_ylabel("U matrix  (KB, float32)", fontsize=9)
    ax.set_title("Fast-Weight Memory Footprint\nper batch element", fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)
    annotate_bars(ax, bars, mem_kb, fmt="{:.0f}")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.subplots_adjust(hspace=0.55)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {out}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_table(all_results):
    print(f"\n{'='*82}")
    print(f"  RANK ABLATION — SUMMARY TABLE")
    print(f"{'='*82}")
    print(f"  {'rank':>5}  {'loss_A':>7}  {'loss_B':>7}  {'loss_C':>7}  "
          f"{'rec_sw2':>8}  {'‖U‖_C':>7}  {'mem KB':>7}  {'ms/frame':>9}")
    print(f"  {'-'*77}")
    for r in all_results:
        rec = str(r["rec_sw2"]) if r["rec_sw2"] is not None else "never"
        print(f"  {r['rank']:>5}  {r['loss_A']:>7.4f}  {r['loss_B']:>7.4f}  "
              f"{r['loss_C']:>7.4f}  {rec:>8}  {r['u_norm_C']:>7.3f}  "
              f"{r['u_bytes']/1024:>7.1f}  {r['ms_per_frame']:>9.3f}")
    print(f"{'='*82}\n")

    # sweet-spot heuristic: best loss_C per KB
    best = min(all_results, key=lambda r: r["loss_C"])
    eff  = min(all_results,
               key=lambda r: r["loss_C"] * (r["u_bytes"] / 1024))
    print(f"  Best raw adaptation quality : rank = {best['rank']}  "
          f"(loss_C = {best['loss_C']:.4f})")
    print(f"  Best efficiency (loss × mem): rank = {eff['rank']}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    print("Loading audio ...")
    feats_a, feats_abc, switches = make_features()
    print(f"  Full clip: {feats_abc.shape[0]} frames  |  "
          f"switches at frames {switches}\n")

    base_config = None   # each rank pre-trains independently

    all_results = []
    for rank in RANKS:
        print(f"── rank = {rank:2d} ──────────────────────────────")
        res = run_rank(rank, None, feats_a, feats_abc, switches, base_config)
        all_results.append(res)
        print(f"   loss  A={res['loss_A']:.4f}  B={res['loss_B']:.4f}  "
              f"C={res['loss_C']:.4f}  | rec_sw2={res['rec_sw2']}  "
              f"mem={res['u_bytes']//1024}KB  {res['ms_per_frame']:.2f}ms/frame\n")

    print_table(all_results)
    plot_ablation(all_results, switches)
