"""DREAM Cell: Dynamic Recall and Elastic Adaptive Memory."""

import math
import torch
import torch.nn as nn
from typing import Tuple, Optional
from dream_net.core.config import DREAMConfig
from dream_net.core.state import DREAMState


class DREAMCell(nn.Module):
    """
    DREAM (Dynamic Recall and Elastic Adaptive Memory) cell.
    
    A continuous-time RNN cell with:
    - Predictive coding with fast weights
    - Surprise-driven plasticity (Hebbian learning)
    - Liquid Time-Constants (LTC) for adaptive integration
    - Sleep consolidation for memory stabilization
    
    This is the core building block. Use it like any PyTorch RNN cell:
    
    Examples
    --------
    >>> from dream import DREAMConfig, DREAMCell
    >>> config = DREAMConfig(input_dim=39, hidden_dim=256)
    >>> cell = DREAMCell(config)
    
    >>> # Process sequence
    >>> state = cell.init_state(batch_size=32)
    >>> for t in range(sequence_length):
    ...     x = input_seq[:, t, :]  # (batch, input_dim)
    ...     h, state = cell(x, state)
    
    >>> # Or process full sequence at once
    >>> output, state = cell.forward_sequence(input_seq)
    
    Parameters
    ----------
    config : DREAMConfig
        Model configuration
    
    Attributes
    ----------
    config : DREAMConfig
        Model configuration
    C : nn.Parameter
        Predictive coding matrix (input_dim, hidden_dim)
    W : nn.Parameter
        Error projection matrix (hidden_dim, input_dim)
    B : nn.Parameter
        Input projection matrix (hidden_dim, input_dim)
    V : nn.Buffer
        Fast weights right factor (input_dim, rank)
    eta : nn.Parameter
        Plasticity coefficient for Hebbian learning
    tau_sys : nn.Parameter
        Base time constant for LTC
    classifier : nn.Linear
        Optional classifier head for downstream tasks
    """
    
    def __init__(self, config: DREAMConfig):
        super().__init__()
        self.config = config
        
        # ================================================================
        # BLOCK 1: Predictive Coding
        # ================================================================
        self.C = nn.Parameter(torch.randn(config.input_dim, config.hidden_dim))
        self.W = nn.Parameter(torch.randn(config.hidden_dim, config.input_dim))
        self.B = nn.Parameter(torch.randn(config.hidden_dim, config.input_dim))
        
        # ================================================================
        # BLOCK 2: Fast Weights (Low-rank decomposition)
        # ================================================================
        # Initialize V with SVD for stability
        V_init = torch.randn(config.input_dim, config.rank)
        U_svd, _, Vh_svd = torch.linalg.svd(V_init, full_matrices=False)
        self.register_buffer('V', U_svd @ Vh_svd)
        
        self.eta = nn.Parameter(torch.tensor(config.base_plasticity))
        
        # ================================================================
        # BLOCK 3: Surprise Gate
        # ================================================================
        # Note: adaptive_tau is now part of DREAMState, not a buffer
        
        # ================================================================
        # BLOCK 4: Liquid Time-Constant (LTC)
        # ================================================================
        self.tau_sys = nn.Parameter(torch.tensor(
            config.ltc_tau_sys if config.ltc_enabled else 0.0
        ))
        self.min_tau = 0.01  # Minimum time constant
        self.max_tau = 50.0  # Maximum time constant
        # Learnable surprise scale for adaptive LTC (stored in log-space: exp(param) = scale)
        # init = log(ltc_surprise_scale) so that exp(param) starts at the configured value
        self.ltc_surprise_scale = nn.Parameter(torch.tensor(math.log(max(config.ltc_surprise_scale, 1e-6))))
        
        # ================================================================
        # Parameters
        # ================================================================
        self.beta = config.error_smoothing
        self.beta_s = config.surprise_smoothing
        self.tau_0 = config.base_threshold
        self.alpha = config.entropy_influence
        self.gamma = config.surprise_temperature
        self.kappa = config.kappa
        self.lambda_ = config.forgetting_rate
        self.target_norm = config.target_norm
        self.sleep_rate = config.sleep_rate
        self.S_min = config.min_surprise_for_sleep
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """
        Initialize weights using Xavier/Kaiming initialization.
        
        Called automatically in __init__. Can be called manually to reset.
        """
        # Xavier initialization for predictive coding matrices
        nn.init.xavier_uniform_(self.C)
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.B)
        
        # Ensure C[0, :] is positive (stability constraint)
        with torch.no_grad():
            self.C[0, :] = torch.abs(self.C[0, :])
        
        # V is already initialized with SVD
        
        # Small initialization for classifier if present
        if hasattr(self, 'classifier') and self.classifier is not None:
            nn.init.xavier_uniform_(self.classifier.weight)
            nn.init.zeros_(self.classifier.bias)
    
    def init_state(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ) -> DREAMState:
        """
        Initialize cell state.
        
        Parameters
        ----------
        batch_size : int
            Batch size
        device : torch.device, optional
            Device for tensors
        dtype : torch.dtype, optional
            Data type for tensors
            
        Returns
        -------
        DREAMState
            Initialized state
        """
        return DREAMState.init_from_config(
            self.config, batch_size, device, dtype
        )
    
    def surprise_gate(
        self,
        error: torch.Tensor,
        error_norm: torch.Tensor,
        state: DREAMState
    ) -> torch.Tensor:
        """
        Compute surprise with habituation (adaptive threshold).

        The effective threshold combines:
        1. Classical entropy-based threshold
        2. Adaptive habituation threshold (learns to ignore constant errors)

        Parameters
        ----------
        error : torch.Tensor
            Prediction error (batch, input_dim)
        error_norm : torch.Tensor
            Norm of error (batch,)
        state : DREAMState
            Current state containing error_var and adaptive_tau

        Returns
        -------
        torch.Tensor
            Surprise values (batch,)
        """
        # Classical entropy from error variance
        eps = 1e-6
        entropy = 0.5 * torch.log(2 * torch.pi * torch.e * (state.error_var.mean(dim=-1) + eps))
        entropy = torch.clamp(entropy, 0.0, 2.0)

        # HABITUATION: Adaptive threshold with slow adaptation
        habituation_rate = 0.001
        new_adaptive_tau = (
            (1 - habituation_rate) * state.adaptive_tau +
            habituation_rate * error_norm
        )
        # Clamp to prevent "deafness"
        new_adaptive_tau = torch.clamp(new_adaptive_tau, max=0.8)

        # Update state
        state.adaptive_tau = new_adaptive_tau

        # Final threshold = classical + adaptive
        classical_tau = self.tau_0 * (1 + self.alpha * entropy)
        effective_tau = 0.3 * classical_tau + 0.7 * state.adaptive_tau

        # Compute surprise
        surprise = torch.sigmoid((error_norm - effective_tau) / self.gamma)

        return surprise
    
    def update_fast_weights(
        self,
        h_prev: torch.Tensor,
        error: torch.Tensor,
        surprise: torch.Tensor,
        state: DREAMState
    ) -> None:
        """
        Update fast weights U via Hebbian learning with surprise modulation.

        Each batch element has its own U matrix for independent adaptation.
        Uses efficient batch operations via torch.bmm.

        Parameters
        ----------
        h_prev : torch.Tensor
            Previous hidden state (batch, hidden_dim)
        error : torch.Tensor
            Prediction error (batch, input_dim)
        surprise : torch.Tensor
            Surprise values (batch,)
        state : DREAMState
            Current state (U and U_target are updated in-place)
        """
        batch_size = h_prev.shape[0]

        # Hebbian term: outer product projected onto V
        # (batch, hidden, 1) @ (batch, 1, input) = (batch, hidden, input)
        h_outer = h_prev.unsqueeze(2)  # (batch, hidden, 1)
        error_outer = error.unsqueeze(1)  # (batch, 1, input)
        outer_product = h_outer @ error_outer  # (batch, hidden, input)

        # Project onto shared V: (batch, hidden, input) @ (input, rank) = (batch, hidden, rank)
        hebbian = outer_product @ self.V.unsqueeze(0).expand(batch_size, -1, -1)

        # Adaptive forgetting: λ_eff = λ × (1 + k × S_t)
        # High surprise → stale patterns cleared faster before new ones are written
        surprise_expanded = surprise.view(-1, 1, 1)  # (batch, 1, 1)
        lambda_eff = self.lambda_ * (1.0 + self.config.adaptive_forgetting_scale * surprise_expanded)
        dU = (
            -lambda_eff * (state.U - state.U_target) +
            self.eta * surprise_expanded * hebbian
        )

        # Euler integration
        U_new = state.U + dU * self.config.time_step

        # Per-batch normalization to target norm
        U_norm = U_new.norm(dim=(1, 2), keepdim=True)  # (batch, 1, 1)
        scale = (self.target_norm / (U_norm + 1e-6)).clamp(max=1.5)  # Allow some overshoot
        U_new = U_new * scale

        # Update state
        state.U = U_new
    
    def compute_ltc_update(
        self,
        h_prev: torch.Tensor,
        input_effect: torch.Tensor,
        surprise: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute hidden state update with Liquid Time-Constant.
        
        Uses Euler method for continuous-time dynamics:
        dh/dt = (-h + h_target) / tau
        
        With stabilized tau clamping to prevent numerical instability.

        Parameters
        ----------
        h_prev : torch.Tensor
            Previous hidden state (batch, hidden_dim)
        input_effect : torch.Tensor
            Combined input effects (batch, hidden_dim)
        surprise : torch.Tensor
            Surprise values (batch,)
            
        Returns
        -------
        torch.Tensor
            New hidden state (batch, hidden_dim)
        """
        h_target = torch.tanh(input_effect)  # (batch, hidden)

        # If tau_sys ~ 0: classic update (blended via tensor ops, no Python branch)
        tau_sys_clamped = self.tau_sys.clamp(min=self.min_tau)
        tau_dynamic = tau_sys_clamped / (1.0 + surprise * self.ltc_surprise_scale.exp())
        tau_effective = tau_dynamic.clamp(self.min_tau, self.max_tau)  # (batch,)
        dt_over_tau = self.config.time_step / (tau_effective.unsqueeze(1) + self.config.time_step)
        dt_over_tau = dt_over_tau.clamp(0.01, 0.5)

        # LTC update: h_new = (1 - dt/tau) * h_prev + (dt/tau) * h_target
        h_ltc = (1 - dt_over_tau) * h_prev + dt_over_tau * h_target

        # Blend classic update when tau_sys is near zero (no Python sync)
        classic = h_target * 0.95 + h_prev * 0.05
        use_ltc = torch.sigmoid((self.tau_sys - 0.01) * 100.0)  # ~1 when tau>0.01, ~0 otherwise
        return use_ltc * h_ltc + (1 - use_ltc) * classic
    
    def forward(
        self,
        x: torch.Tensor,
        state: DREAMState
    ) -> Tuple[torch.Tensor, DREAMState]:
        """
        Forward pass of DREAM cell.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor (batch, input_dim)
        state : DREAMState
            Current state
            
        Returns
        -------
        h_new : torch.Tensor
            New hidden state (batch, hidden_dim)
        state : DREAMState
            Updated state
        """
        batch_size = x.shape[0]
        
        # ================================================================
        # Normalization
        # ================================================================
        x_norm = x / (x.norm(dim=-1, keepdim=True) + 1e-6)
        x_norm = torch.clamp(x_norm, -1.0, 1.0)  # keep sign — critical for log-mel
        
        # ================================================================
        # Dynamic Prediction — FAST WEIGHTS DISABLED
        # ================================================================
        # Using pure predictive coding without fast weights
        # Fast weights were corrupting the learned C matrix
        # Rely on LTC + surprise gate for adaptation instead

        # Base prediction: x_pred = tanh(h @ C.T)  shape: (batch, input_dim)
        x_pred_raw = state.h @ self.C.T

        # Apply activation and scale
        x_pred = torch.tanh(x_pred_raw) * x.norm(dim=-1, keepdim=True)
        
        # ================================================================
        # Error Computation
        # ================================================================
        error = x - x_pred  # (batch, input_dim)
        error_norm = error.norm(dim=-1)  # (batch,)

        # Normalise error by input scale → surprise in [0,1] regardless of
        # input amplitude or dimension.  Without this, for dim=80 the raw
        # error_norm >> threshold always, surprise ≡ 1, gate never closes.
        x_scale = x.norm(dim=-1) + 1e-6
        rel_error_norm = error_norm / x_scale  # ∈ [0, ~2], 0 = perfect pred

        # ================================================================
        # Surprise Gate
        # ================================================================
        surprise = self.surprise_gate(error, rel_error_norm, state)

        # ================================================================
        # Fast Weights Update — DISABLED
        # ================================================================
        # Fast weights were causing degradation by memorizing wrong patterns
        pass

        # ================================================================
        # State Update with LTC — simplified without fast weights
        # ================================================================
        # Only use base effect and error effect (no fast_effect)
        U_batch = state.U  # (batch, hidden, rank) - kept for compatibility

        base_effect = self.B @ x_norm.T
        base_effect = base_effect.T  # (batch, hidden)
        
        error_effect = self.W @ error.T
        error_effect = error_effect.T  # (batch, hidden)

        # Combine effects: high surprise → error is informative → inject more
        # High surprise = large prediction error = error signal carries real information
        input_effect = (
            state.h * 0.7 +
            base_effect * 0.2 +
            error_effect * surprise.unsqueeze(1) * 0.3
        )

        # LTC update (already performs weighted interpolation internally)
        h_new = self.compute_ltc_update(state.h, input_effect, surprise)

        # ================================================================
        # Update Statistics
        # ================================================================
        alpha = 0.05
        state.error_mean = (1 - alpha) * state.error_mean + alpha * error
        state.error_var = (1 - alpha) * state.error_var + alpha * (error - state.error_mean) ** 2
        state.avg_surprise = (1 - self.beta_s) * state.avg_surprise + self.beta_s * surprise

        # ================================================================
        # Sleep Consolidation
        # ================================================================
        # Consolidate during calm periods (low surprise = rest/sleep state)
        # Transfers accumulated fast weights U → long-term U_target
        # Sleep consolidation: runs as weighted blend — no Python branch, no GPU sync
        avg_surprise_mean = state.avg_surprise.mean()
        sleep_gate = torch.sigmoid((self.S_min - avg_surprise_mean) * 50.0)  # ~1 when asleep

        dU_target = self.sleep_rate * (state.U - state.U_target)
        U_target_new = state.U_target + sleep_gate * dU_target
        U_target_norm = U_target_new.norm(dim=(1, 2), keepdim=True)
        scale = (self.target_norm / (U_target_norm + 1e-6)).clamp(max=1.5)
        state.U_target = U_target_new * scale

        state.h = h_new

        return h_new, state
    
    def forward_sequence(
        self,
        x_seq: torch.Tensor,
        state: Optional[DREAMState] = None,
        return_all: bool = False
    ) -> Tuple[torch.Tensor, DREAMState]:
        """
        Process a full sequence through the cell.
        
        Parameters
        ----------
        x_seq : torch.Tensor
            Input sequence (batch, time, input_dim)
        state : DREAMState, optional
            Initial state. If None, initialized from config.
        return_all : bool, default=False
            If True, return all hidden states. Otherwise, return only final.
            
        Returns
        -------
        output : torch.Tensor
            If return_all: (batch, time, hidden_dim)
            Otherwise: (batch, hidden_dim) (final state only)
        state : DREAMState
            Final state after processing sequence
        """
        batch_size, time_steps, _ = x_seq.shape
        
        if state is None:
            state = self.init_state(batch_size, device=x_seq.device, dtype=x_seq.dtype)
        
        if return_all:
            all_h = []
        
        for t in range(time_steps):
            x_t = x_seq[:, t, :]  # (batch, input_dim)
            h, state = self(x_t, state)
            
            if return_all:
                all_h.append(h.unsqueeze(1))
        
        if return_all:
            output = torch.cat(all_h, dim=1)  # (batch, time, hidden_dim)
        else:
            output = h  # (batch, hidden_dim)
        
        return output, state
