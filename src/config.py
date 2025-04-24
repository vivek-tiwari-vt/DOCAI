import yaml
import re

def load_config(path):
    with open(path) as f:
        config = yaml.safe_load(f)
    
    # Process numeric values that might be loaded as strings
    # Specifically handle the learning_rate in training_args
    if 'training_args' in config and 'learning_rate' in config['training_args']:
        lr = config['training_args']['learning_rate']
        if isinstance(lr, str):
            # Convert scientific notation or numeric strings to float
            config['training_args']['learning_rate'] = float(lr)
    
    return config