import torch
from src.models.baseline_layoutlm import BaselineLayoutLMForTokenClassification
from src.config import load_config

def test_baseline_forward():
    config = load_config('configs/baseline_layoutlm.yaml')
    model = BaselineLayoutLMForTokenClassification(config)
    input_ids = torch.randint(0, 1000, (2, config['max_seq_length']))
    attention_mask = torch.ones_like(input_ids)
    token_type_ids = torch.zeros_like(input_ids)
    bbox = torch.zeros((2, config['max_seq_length'], 4), dtype=torch.long)
    outputs = model(
        input_ids=input_ids, attention_mask=attention_mask,
        token_type_ids=token_type_ids, bbox=bbox
    )
    assert outputs.logits.shape == (2, config['max_seq_length'], config['num_labels'])