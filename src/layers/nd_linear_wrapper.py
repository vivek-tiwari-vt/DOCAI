import torch
import torch.nn as nn
from NdLinear.ndlinear import NdLinear

class NdLinearFFN(nn.Module):
    """
    A custom implementation of the Feed-Forward Network for LayoutLM using NdLinear.
    Preserves multi-dimensional structure while maintaining compatibility with LayoutLM.
    """
    def __init__(self, input_dim, hidden_dim=3072, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Intermediate layer components using NdLinear
        self.dense = NdLinear(input_dims=(input_dim,), hidden_size=(hidden_dim,))
        self.intermediate_act_fn = nn.GELU()
        
        # Output layer components using NdLinear
        self.out_dense = NdLinear(input_dims=(hidden_dim,), hidden_size=(input_dim,))
        self.LayerNorm = nn.LayerNorm(input_dim, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, attention_output=None):
        """
        Forward pass with NdLinear transformations.
        Handles both standalone FFN and feed_forward_chunk cases.
        """
        # Case 1: Called as part of feed_forward_chunk in LayoutLM
        if attention_output is not None:
            hidden_states = self.out_dense(hidden_states)
            hidden_states = self.dropout(hidden_states)
            hidden_states = self.LayerNorm(hidden_states + attention_output)
            return hidden_states
        
        # Case 2: Called as standalone FFN module
        # Reshape for NdLinear if needed (preserve batch dimension)
        original_shape = hidden_states.shape
        if len(original_shape) > 2:
            hidden_states = hidden_states.reshape(-1, original_shape[-1])
        
        intermediate_output = self.dense(hidden_states)
        intermediate_output = self.intermediate_act_fn(intermediate_output)
        
        output = self.out_dense(intermediate_output)
        output = self.dropout(output)
        
        # Reshape back if needed
        if len(original_shape) > 2:
            output = output.reshape(*original_shape[:-1], -1)
        
        output = self.LayerNorm(output + hidden_states)
        return output