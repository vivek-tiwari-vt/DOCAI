import torch.nn as nn
from transformers import LayoutLMModel
from src.layers.moe import MoELayer, TopKGating
from src.layers.nd_linear_wrapper import NdLinearFFN

class LayoutMoNdEForTokenClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        base = LayoutLMModel.from_pretrained(config['model_checkpoint'])
        hidden = base.config.hidden_size
        # create experts
        experts = [nn.Sequential(nn.Linear(hidden, hidden), nn.GELU())]
        for _ in range(config['moe_params']['num_experts']-1):
            experts.append(NdLinearFFN(hidden, config['ndlinear_params']['hidden_dim']))
        gating = TopKGating(hidden, config['moe_params']['num_experts'], config['moe_params']['top_k'])
        # replace FFNs
        for layer in base.encoder.layer:
            layer.intermediate.dense = MoELayer(experts, gating)
        self.base = base
        self.classifier = nn.Linear(hidden, config['num_labels'])

    def forward(self, input_ids, attention_mask, token_type_ids, bbox, labels=None):
        outputs = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            bbox=bbox
        )
        logits = self.classifier(outputs.last_hidden_state)
        return {'logits': logits}