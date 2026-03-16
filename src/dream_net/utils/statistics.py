"""Running statistics for DREAM cell."""

import torch
import torch.nn as nn
from typing import Optional


class RunningStatistics(nn.Module):
    """
    Running statistics tracker with exponential smoothing.
    
    Tracks mean and variance of prediction errors and surprise values
    using exponential moving averages. Used by DREAM cell for
    adaptive surprise computation.
    
    Parameters
    ----------
    input_dim : int
        Dimension of input features
    error_smoothing : float
        Smoothing coefficient for error statistics (beta)
    surprise_smoothing : float
        Smoothing coefficient for surprise (beta_s)
        
    Examples
    --------
    >>> stats = RunningStatistics(input_dim=39)
    >>> error = torch.randn(32, 39)  # batch of errors
    >>> surprise = torch.rand(32)     # batch of surprise values
    >>> stats.update(error, surprise)
    >>> print(stats.error_mean.shape)  # (32, 39)
    """
    
    def __init__(
        self,
        input_dim: int,
        error_smoothing: float = 0.01,
        surprise_smoothing: float = 0.01
    ):
        super().__init__()
        self.input_dim = input_dim
        self.error_smoothing = error_smoothing
        self.surprise_smoothing = surprise_smoothing
        
        # Running statistics (updated per batch)
        self.register_buffer('error_mean', torch.zeros(input_dim))
        self.register_buffer('error_var', torch.ones(input_dim))
        self.register_buffer('avg_surprise', torch.tensor(0.0))
    
    def update(
        self,
        prediction_error: torch.Tensor,
        surprise: torch.Tensor
    ) -> None:
        """
        Update statistics with new observations.
        
        Parameters
        ----------
        prediction_error : torch.Tensor
            Error tensor of shape (batch, input_dim) or (input_dim,)
        surprise : torch.Tensor
            Surprise scalar tensor of shape (batch,) or ()
        """
        # Handle batched vs non-batched input
        if prediction_error.dim() == 1:
            # Single sample
            self.error_mean = (
                (1 - self.error_smoothing) * self.error_mean +
                self.error_smoothing * prediction_error
            )
            
            squared_diff = (prediction_error - self.error_mean) ** 2
            self.error_var = (
                (1 - self.error_smoothing) * self.error_var +
                self.error_smoothing * squared_diff
            )
            
            self.avg_surprise = (
                (1 - self.surprise_smoothing) * self.avg_surprise +
                self.surprise_smoothing * surprise
            )
        else:
            # Batched: compute batch statistics
            batch_mean = prediction_error.mean(dim=0)
            batch_var = prediction_error.var(dim=0)
            
            self.error_mean = (
                (1 - self.error_smoothing) * self.error_mean +
                self.error_smoothing * batch_mean
            )
            
            self.error_var = (
                (1 - self.error_smoothing) * self.error_var +
                self.error_smoothing * batch_var
            )
            
            self.avg_surprise = (
                (1 - self.surprise_smoothing) * self.avg_surprise +
                self.surprise_smoothing * surprise.mean()
            )
    
    def reset(self) -> None:
        """Reset all statistics to initial values."""
        self.error_mean.zero_()
        self.error_var.fill_(1.0)
        self.avg_surprise.zero_()
    
    def forward(
        self,
        prediction_error: torch.Tensor,
        surprise: torch.Tensor
    ) -> None:
        """Alias for update() to support nn.Module interface."""
        self.update(prediction_error, surprise)
