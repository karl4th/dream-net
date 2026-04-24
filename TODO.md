# TODO

## Exp 0 — Validate fast weights on audio

Goal: confirm that re-enabled fast weights improve (or at least don't break)
adaptation on the existing speaker-switch audio experiments before touching
any multimodal code.

---

### Setup

- [ ] Pick a short CommonVoice clip (speaker A → speaker B, ~10s each)
- [ ] Verify `experiments/speaker_switch.py` runs cleanly as-is (baseline)

### Pre-train baseline

- [ ] Run `speaker_switch.py` with `cell.fast_weights_enabled = False`
- [ ] Record: mean loss per segment, adaptation speed after switch
- [ ] Save checkpoint as `checkpoints/audio_pretrain_baseline.pt`

### Enable fast weights

- [ ] Add `model.cell.enable_fast_weights()` call in `speaker_switch.py`
      after pre-training, before the switch segment
- [ ] Run experiment
- [ ] Check for NaN / Inf in loss (numerical stability)
- [ ] Check U norm stays bounded (should be ≤ `target_norm * 1.5 = 3.0`)

### Compare

- [ ] Plot loss curves: static vs adaptive (same plot, like existing experiments)
- [ ] Key question: does DREAM-full recover faster after speaker switch?
- [ ] Run `stress_test.py` (A→B→C) with fast weights ON — same check

### Tune if needed

- [ ] If fast weights hurt: lower `base_plasticity` (try 0.1, 0.2)
- [ ] If adaptation too slow: raise `base_plasticity` (try 0.6)
- [ ] If U explodes: lower `adaptive_forgetting_scale` (try 4.0)
- [ ] Lock winning values into `configs/proven_v1.yaml`

### Done when

- [ ] Fast weights ON ≥ fast weights OFF on speaker-switch loss
- [ ] No NaN, no U explosion across 3 runs with different seeds
- [ ] Results documented in `reports/`

---

## Backlog

- [ ] Exp 1 — Synthetic multimodal: swap synthetic data in
      `operator_prediction.py` for richer signal, re-check adaptation delta
- [ ] Exp 2 — Real robot: sensor logging pipeline (IMU + encoders + actions)
- [ ] Exp 3 — Real robot: 3+ operators, measure adaptation speed per operator
- [ ] Exp 4 — Operator switch mid-session
- [ ] Exp 5 — Rank ablation on operator data

---

## Ideas (not scheduled)

- [ ] Emotion-conditioned speech generation
      DREAM angle: fast weights encode emotional "style" (whisper, excited,
      calm) the way they encode speaker style. Surprise gate fires on
      emotional transitions. Needs: SpeechDecoderHead, vocoder (HiFi-GAN),
      emotion embedding as extra encoder input.
