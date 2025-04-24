import argparse
import os
import torch
from transformers import LayoutLMTokenizerFast
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import load_config
from src.utils.helpers import set_seed, setup_logging
from src.data_utils.funsd_processor import (load_raw_funsd, preprocess_funsd_for_layoutlm)
from src.data_utils.datasets import FunsdDataset
from torch.utils.data import DataLoader
import wandb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', required=True)
    parser.add_argument('--wandb_project', default='DocAI_NdLinear')
    args = parser.parse_args()

    config = load_config(args.config_path)
    os.makedirs(config['output_dir'], exist_ok=True)
    setup_logging(os.path.join(config['output_dir'], 'run.log'))
    set_seed(config.get('seed', 42))  # Added default seed value
    wandb.init(project=args.wandb_project, config=config)

    # data
    raw = load_raw_funsd(config['data_dir'])
    tokenizer = LayoutLMTokenizerFast.from_pretrained(config['model_checkpoint'])
    train_enc = preprocess_funsd_for_layoutlm(raw['train'], tokenizer, config.get('max_seq_length', 512))
    # Check for 'test' split in the dataset
    if 'test' not in raw:
        raise KeyError("'test' split not found in the dataset. Available splits: {}".format(list(raw.keys())))
    val_enc = preprocess_funsd_for_layoutlm(raw['test'], tokenizer, config.get('max_seq_length', 512))
    train_ds = FunsdDataset(train_enc)
    val_ds = FunsdDataset(val_enc)
    train_loader = DataLoader(train_ds, batch_size=config['training_args']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['training_args']['batch_size'])

    # model selection
    from src.models.baseline_layoutlm import BaselineLayoutLMForTokenClassification
    from src.models.ndlinear_ffn_layoutlm import NdFFNLayoutLMForTokenClassification
    from src.models.layout_monde import LayoutMoNdEForTokenClassification
    variant = os.path.basename(args.config_path).split('.')[0]
    if variant == 'baseline_layoutlm': model = BaselineLayoutLMForTokenClassification(config)
    elif variant == 'ndlinear_ffn_layoutlm': model = NdFFNLayoutLMForTokenClassification(config)
    else: model = LayoutMoNdEForTokenClassification(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training_args']['learning_rate'])
    total_steps = len(train_loader) * config['training_args']['epochs']
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=total_steps//3, gamma=0.1)

    # train
    from src.engine.trainer import train
    train(model, train_loader, val_loader, optimizer, scheduler, device, config)

if __name__ == '__main__':
    main()
