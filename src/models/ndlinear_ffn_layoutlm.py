import torch
import torch.nn as nn
from transformers import LayoutLMModel
from src.layers.nd_linear_wrapper import NdLinearFFN

class NdFFNLayoutLMForTokenClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        base = LayoutLMModel.from_pretrained(config['model_checkpoint'])
        hidden = base.config.hidden_size
        intermediate = config.get('ndlinear_params', {}).get('hidden_dim', hidden)
        
        # Get the actual intermediate size from the model
        if hasattr(base.encoder.layer[0].intermediate, 'dense') and hasattr(base.encoder.layer[0].intermediate.dense, 'out_features'):
            actual_intermediate = base.encoder.layer[0].intermediate.dense.out_features
            if actual_intermediate != intermediate:
                print(f"Warning: Adjusting intermediate size from {intermediate} to {actual_intermediate}")
                intermediate = actual_intermediate
        
        # Create a separate NdLinearFFN instance for each layer
        for layer in base.encoder.layer:
            # Get the actual intermediate size from this specific layer
            if hasattr(layer.intermediate, 'dense') and hasattr(layer.intermediate.dense, 'out_features'):
                layer_intermediate = layer.intermediate.dense.out_features
            else:
                layer_intermediate = intermediate
                
            # Create a new NdLinearFFN with the correct dimensions
            ffn = NdLinearFFN(hidden, layer_intermediate)
            
            # Create a custom forward function for the intermediate layer
            def create_intermediate_forward(ffn_module):
                def forward_fn(hidden_states):
                    intermediate_output = ffn_module.dense(hidden_states)
                    intermediate_output = ffn_module.intermediate_act_fn(intermediate_output)
                    return intermediate_output
                return forward_fn
            
            # Create a custom forward function for the output layer
            def create_output_forward(ffn_module):
                def forward_fn(hidden_states, input_tensor):
                    return ffn_module(hidden_states, input_tensor)
                return forward_fn
            
            # Replace the intermediate layer's forward method
            layer.intermediate.forward = create_intermediate_forward(ffn)
            
            # Replace the output layer's forward method
            layer.output.forward = create_output_forward(ffn)
            
            # Keep the original dense layers for dimension compatibility
            # This ensures the matrix dimensions match during feed_forward_chunk
        self.base = base
        self.classifier = nn.Linear(hidden, config['num_labels'])

    def forward(self, input_ids, attention_mask, token_type_ids, bbox, labels=None):
        outputs = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            bbox=bbox
        )
        sequence_output = outputs.last_hidden_state
        logits = self.classifier(sequence_output)
        
        # Create output dictionary with logits
        result = {'logits': logits}
        
        # Calculate loss if labels are provided
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, logits.shape[-1])
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(-100).type_as(labels)
                )
                # Ensure labels are within the valid range
                num_labels = logits.shape[-1]
                active_labels = torch.clamp(active_labels, min=-100, max=num_labels-1)
                loss = loss_fct(active_logits, active_labels)
            else:
                # Ensure labels are within the valid range
                num_labels = logits.shape[-1]
                clamped_labels = torch.clamp(labels.view(-1), min=-100, max=num_labels-1)
                loss = loss_fct(logits.view(-1, logits.shape[-1]), clamped_labels)
            
            result['loss'] = loss
        
        return result