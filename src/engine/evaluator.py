import torch
from src.utils.metrics import compute_token_classification_metrics

def evaluate(model, dataloader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            # Handle both dictionary output and object with logits attribute
            if isinstance(outputs, dict):
                logits = outputs['logits'].cpu().numpy()
            else:
                logits = outputs.logits.cpu().numpy()
            preds.extend(logits.argmax(-1))
            labels.extend(batch['labels'].cpu().numpy())
    return compute_token_classification_metrics(preds, labels)
