
import os
import yaml
import pickle
import argparse


def parse_configure(model=None, dataset=None):
    parser = argparse.ArgumentParser(description='DMRec')
    parser.add_argument('--model', type=str, default='mult_vae_mddm', help='Model name (legacy, e.g., mult_vae_mddm)')
    parser.add_argument('--base_model', type=str, default=None, help='Base model, e.g., mult_vae / cvga / l_diffrec')
    parser.add_argument('--strategy', type=str, default=None, help='Alignment strategy, e.g., mddm / cpdm / godm / rfdm')
    parser.add_argument('--dataset', type=str, default='amazon', help='Dataset name')
    parser.add_argument('--device', type=str, default='cuda', help='cpu or cuda')
    parser.add_argument('--seed', type=int, default=None, help='Device number')
    parser.add_argument('--cuda', type=str, default='0', help='Device number')
    args, _ = parser.parse_known_args()

    # cuda
    if args.device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    # dataset
    if dataset is not None:
        args.dataset = dataset

    # Determine model name: --model and --base_model + --strategy are parallel options
    # Priority: 1) explicit --model (if provided), 2) base_model + strategy (if both provided), 3) default
    if model is not None:
        # Explicit model parameter provided (highest priority)
        model_name = model.lower()
    elif args.model is not None and args.model != 'mult_vae_mddm':
        # Explicit --model provided (not default value)
        model_name = args.model.lower()
    elif args.base_model is not None and args.strategy is not None:
        # Combine base_model and strategy to form model name
        model_name = '{}_{}'.format(args.base_model.lower(), args.strategy.lower())
    elif args.model is not None:
        # Use default --model value
        model_name = args.model.lower()
    else:
        # Fallback to default
        model_name = 'default'

    # Helper function to deep merge dictionaries
    def deep_merge(base_dict, override_dict):
        """Deep merge override_dict into base_dict"""
        result = base_dict.copy()
        for key, value in override_dict.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    # Load default.yml as base configuration
    default_config_path = '../encoder/config/modelconf/default.yml'
    if os.path.exists(default_config_path):
        with open(default_config_path, encoding='utf-8') as f:
            default_config_data = f.read()
            configs = yaml.safe_load(default_config_data)
    else:
        # If default.yml doesn't exist, create minimal config
        configs = {
            'optimizer': {'name': 'adam', 'lr': 1.0e-3, 'weight_decay': 0},
            'train': {'epoch': 1000, 'batch_size': 1024, 'save_model': False, 'test_step': 3, 'reproducible': True, 'seed': 2024, 'patience': 20},
            'test': {'metrics': ['recall', 'ndcg'], 'k': [10, 20], 'batch_size': 1024},
            'data': {'type': 'general_cf'},
            'model': {'name': 'default'}
        }

    # Try to load model-specific config file and merge with default
    model_config_path = '../encoder/config/modelconf/{}.yml'.format(model_name)
    if os.path.exists(model_config_path) and model_name != 'default':
        with open(model_config_path, encoding='utf-8') as f:
            model_config_data = f.read()
            model_configs = yaml.safe_load(model_config_data)
            # Merge model-specific config into default config (model config takes precedence)
            configs = deep_merge(configs, model_configs)
            print("[INFO] Loaded config from '{}'".format(model_config_path))
    elif not os.path.exists(model_config_path) and model_name != 'default':
        # If model-specific config doesn't exist, use default.yml and log a warning
        print("[WARNING] Config file '{}' not found. Using default.yml as fallback.".format(model_config_path))

    # Ensure model.name is set
    if 'model' not in configs:
        configs['model'] = {}
    if 'name' not in configs['model']:
        configs['model']['name'] = model_name.lower()
    else:
        configs['model']['name'] = configs['model']['name'].lower()

    # Set default values for missing sections
    if 'tune' not in configs:
        configs['tune'] = {'enable': False}
    configs['device'] = args.device
    if args.dataset is not None:
        configs['data']['name'] = args.dataset
    if args.seed is not None:
        configs['train']['seed'] = args.seed

    # Fill/override base + strategy
    legacy_name = configs['model'].get('name', '')
    # infer from legacy name if needed
    if '_' in legacy_name and ('base' not in configs['model'] or 'strategy' not in configs['model']):
        parts = legacy_name.split('_')
        if len(parts) >= 2:
            configs['model'].setdefault('base', '_'.join(parts[:-1]))
            configs['model'].setdefault('strategy', parts[-1])
    # override via CLI
    if args.base_model is not None:
        configs['model']['base'] = args.base_model.lower()
    if args.strategy is not None:
        configs['model']['strategy'] = args.strategy.lower()
    
    # normalize name for logging
    if 'base' in configs['model'] and 'strategy' in configs['model']:
        configs['model']['name'] = '{}_{}'.format(configs['model']['base'], configs['model']['strategy'])

    # semantic embeddings
    usrprf_embeds_path = "../data/{}/usr_emb_np.pkl".format(configs['data']['name'])
    itmprf_embeds_path = "../data/{}/itm_emb_np.pkl".format(configs['data']['name'])
    with open(usrprf_embeds_path, 'rb') as f:
        configs['usrprf_embeds'] = pickle.load(f)
    with open(itmprf_embeds_path, 'rb') as f:
        configs['itmprf_embeds'] = pickle.load(f)

    return configs

configs = parse_configure()
