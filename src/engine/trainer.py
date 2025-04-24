import torch
from tqdm import tqdm

def train_one_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    losses = []
    for batch in tqdm(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        # Handle both dictionary output and HuggingFace output formats
        if isinstance(outputs, dict) and 'loss' in outputs:
            loss = outputs['loss']
        else:
            loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        losses.append(loss.item())
    return sum(losses)/len(losses)


def train(model, train_loader, val_loader, optimizer, scheduler, device, config):
    best_f1 = 0.0
    for epoch in range(config['training_args']['epochs']):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        # eval
        from src.engine.evaluator import evaluate
        metrics = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_f1={metrics['f1']:.4f}")
        # save best
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            torch.save(model.state_dict(), f"{config['output_dir']}/best_model.pt")