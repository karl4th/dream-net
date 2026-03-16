"""State container for DREAM cell."""

import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from dream_net.core.config import DREAMConfig


@dataclass
class DREAMState:
    """
    State container for DREAM cell.
    
    Contains all state variables that need to be tracked across time steps
    for proper RNN operation with batch support.
    
    Attributes
    ----------
    h : torch.Tensor
        Hidden state of shape (batch, hidden_dim)
        
    U : torch.Tensor
        Fast weights (left factor) of shape (hidden_dim, rank)
        Updated via Hebbian learning with surprise modulation.
        
    U_target : torch.Tensor
        Target fast weights for sleep consolidation.
        Shape: (hidden_dim, rank)
        
    adaptive_tau : torch.Tensor
        Adaptive surprise threshold (habituation).
        Shape: (batch,) or scalar
        
    error_mean : torch.Tensor
        Exponential moving average of prediction error.
        Shape: (batch, input_dim)
        
    error_var : torch.Tensor
        Exponential moving variance of prediction error.
        Shape: (batch, input_dim)
        
    avg_surprise : torch.Tensor
        Exponential moving average of surprise.
        Shape: (batch,) or scalar
        
    Examples
    --------
    >>> from dream import DREAMConfig, DREAMCell, DREAMState
    >>> config = DREAMConfig()
    >>> cell = DREAMCell(config)
    >>> state = cell.init_state(batch_size=32)
    >>> h_new, state_new = cell(x, state)
    """
    
    h: torch.Tensor
    """Hidden state: (batch, hidden_dim)"""
    
    U: torch.Tensor
    """Fast weights (left factor): (batch, hidden_dim, rank)"""
    
    U_target: torch.Tensor
    """Target fast weights: (batch, hidden_dim, rank)"""
    
    adaptive_tau: torch.Tensor
    """Adaptive surprise threshold: (batch,) or scalar"""
    
    error_mean: torch.Tensor
    """Error mean: (batch, input_dim)"""
    
    error_var: torch.Tensor
    """Error variance: (batch, input_dim)"""
    
    avg_surprise: torch.Tensor
    """Average surprise: (batch,) or scalar"""
    
    @classmethod
    def init_from_config(
        cls,
        config: "DREAMConfig",
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ) -> "DREAMState":
        """
        Initialize state from configuration.
        
        Parameters
        ----------
        config : DREAMConfig
            Model configuration
        batch_size : int
            Batch size
        device : torch.device, optional
            Device for tensors
        dtype : torch.dtype, optional
            Data type for tensors
            
        Returns
        -------
        DREAMState
            Initialized state with all zeros
        """
        # Initialize h with small random values for stability
        h = torch.randn(batch_size, config.hidden_dim, device=device, dtype=dtype) * 0.01

        # U is per-batch for independent adaptation
        U = torch.zeros(batch_size, config.hidden_dim, config.rank, device=device, dtype=dtype)
        U_target = torch.zeros(batch_size, config.hidden_dim, config.rank, device=device, dtype=dtype)
        
        return cls(
            h=h,
            U=U,
            U_target=U_target,
            adaptive_tau=torch.full(
                (batch_size,),
                config.base_threshold,
                device=device,
                dtype=dtype
            ),
            error_mean=torch.zeros(batch_size, config.input_dim, device=device, dtype=dtype),
            error_var=torch.ones(batch_size, config.input_dim, device=device, dtype=dtype),
            avg_surprise=torch.zeros(
                (batch_size,),
                device=device,
                dtype=dtype
            ),
        )
    
    def detach(self) -> "DREAMState":
        """
        Detach all tensors from computation graph.
        
        Returns
        -------
        DREAMState
            New state with detached tensors
        """
        return DREAMState(
            h=self.h.detach(),
            U=self.U.detach(),
            U_target=self.U_target.detach(),
            adaptive_tau=self.adaptive_tau.detach(),
            error_mean=self.error_mean.detach(),
            error_var=self.error_var.detach(),
            avg_surprise=self.avg_surprise.detach(),
        )
