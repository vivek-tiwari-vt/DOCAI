import torch.nn as nn
from transformers import LayoutLMForTokenClassification

class BaselineLayoutLMForTokenClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Get num_labels from config with a default value of 8 if not present
        num_labels = config.get('num_labels', 8)
        self.model = LayoutLMForTokenClassification.from_pretrained(
            config['model_checkpoint'],
            num_labels=num_labels
        )

    def forward(self, **kwargs):
        return self.model(**kwargs)
