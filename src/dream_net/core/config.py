"""Configuration for DREAM cells."""

from dataclasses import dataclass


@dataclass
class DREAMConfig:
    """
    Configuration for DREAM (Dynamic Recall and Elastic Adaptive Memory) cell.
    
    DREAM is a continuous-time RNN cell with surprise-driven plasticity
    and liquid time-constants for adaptive integration speeds.
    
    Parameters
    ----------
    input_dim : int, default=39
        Dimension of input features. For ASR with MFCC: 39 (13 + 13Δ + 13ΔΔ).
        
    hidden_dim : int, default=256
        Dimension of hidden state (d_state). Controls model capacity.
        
    rank : int, default=16
        Rank of fast weights decomposition (r). Lower = more compression.
        
    time_step : float, default=0.1
        Integration time step (dt) for continuous-time dynamics.
        
    forgetting_rate : float, default=0.01
        Lambda (λ) - forgetting/decay rate for fast weights.
        Higher = faster forgetting of old patterns.
        
    base_plasticity : float, default=0.1
        Base plasticity coefficient for Hebbian learning.
        
    base_threshold : float, default=0.5
        Base surprise threshold (τ₀). Controls sensitivity to novelty.
        Lower = more sensitive to prediction errors.
        
    entropy_influence : float, default=0.2
        Alpha (α) - entropy influence on surprise threshold.
        Higher = more uncertainty-aware threshold.
        
    surprise_temperature : float, default=0.1
        Gamma (γ) - surprise temperature/scaling parameter.
        Higher = smoother surprise curve.
        
    error_smoothing : float, default=0.01
        Beta (β) - exponential smoothing coefficient for error statistics.
        
    surprise_smoothing : float, default=0.01
        Beta_s - exponential smoothing for average surprise.
        
    target_norm : float, default=2.0
        Target norm for fast weights (W_target). Prevents explosion.
        
    kappa : float, default=0.5
        Homeostasis coefficient (κ) for weight normalization.
        
    ltc_enabled : bool, default=True
        Enable Liquid Time-Constant dynamics. If False, uses classic update.
        
    ltc_tau_sys : float, default=10.0
        Base system time constant for LTC. Controls integration speed.
        Higher = slower integration (more memory), Lower = faster response.
        
    ltc_surprise_scale : float, default=10.0
        Scaling factor for surprise modulation of tau.
        Higher = more dynamic time constant adaptation.
        
    sleep_rate : float, default=0.005
        Sleep consolidation rate (ζ_sleep).
        
    min_surprise_for_sleep : float, default=0.2
        Minimum surprise threshold for sleep activation (S_min).
        
    Examples
    --------
    >>> from dream import DREAMConfig, DREAMCell
    >>> config = DREAMConfig(input_dim=39, hidden_dim=256, rank=16)
    >>> cell = DREAMCell(config)
    """
    
    # =====================================================================
    # Model Dimensions
    # =====================================================================
    input_dim: int = 39
    """Dimension of input features. For ASR: 39 = 13 MFCC + 13Δ + 13ΔΔ"""
    
    hidden_dim: int = 256
    """Dimension of hidden state (d_state)"""
    
    rank: int = 16
    """Rank of fast weights decomposition (r)"""
    
    # =====================================================================
    # Time Parameters
    # =====================================================================
    time_step: float = 0.1
    """Integration time step (dt) for continuous-time dynamics"""
    
    # =====================================================================
    # Plasticity Parameters
    # =====================================================================
    forgetting_rate: float = 0.01
    """Lambda (λ) - forgetting/decay rate for fast weights"""

    base_plasticity: float = 0.1
    """Base plasticity coefficient for Hebbian learning"""
    
    # =====================================================================
    # Surprise Parameters
    # =====================================================================
    base_threshold: float = 0.5
    """Base surprise threshold (τ₀) - controls sensitivity to novelty"""

    entropy_influence: float = 0.2
    """Alpha (α) - entropy influence on surprise threshold"""

    surprise_temperature: float = 0.1
    """Gamma (γ) - surprise temperature/scaling parameter"""
    
    # =====================================================================
    # Smoothing Parameters
    # =====================================================================
    error_smoothing: float = 0.01
    """Beta (β) - exponential smoothing for error statistics"""
    
    surprise_smoothing: float = 0.01
    """Beta_s - exponential smoothing for average surprise"""
    
    # =====================================================================
    # Homeostasis Parameters
    # =====================================================================
    target_norm: float = 2.0
    """Target norm for fast weights (W_target)"""
    
    kappa: float = 0.5
    """Homeostasis coefficient (κ) for weight normalization"""
    
    # =====================================================================
    # Adaptive Forgetting
    # =====================================================================
    adaptive_forgetting_scale: float = 8.0
    """k — scales how much surprise boosts the forgetting rate.
    λ_eff = λ × (1 + k × S_t).
    High surprise → fast clearing of stale fast weights."""

    # =====================================================================
    # Sleep Consolidation Parameters
    # =====================================================================
    sleep_rate: float = 0.005
    """Sleep consolidation rate (ζ_sleep)"""
    
    min_surprise_for_sleep: float = 0.2
    """Minimum surprise threshold for sleep activation (S_min)"""
    
    # =====================================================================
    # Liquid Time-Constant (LTC) Parameters
    # =====================================================================
    ltc_enabled: bool = True
    """Enable Liquid Time-Constant dynamics"""

    ltc_tau_sys: float = 10.0
    """Base system time constant for LTC"""

    ltc_surprise_scale: float = 10.0
    """Scaling factor for surprise modulation of tau"""
