
import os
import yaml
import pickle
import argparse


def parse_configure(model=None, dataset=None):
    parser = argparse.ArgumentParser(description='DMRec')
    parser.add_argument('--model', type=str, default='mult_vae_mddm', help='Model name (legacy, e.g., mult_vae_mddm)')
    parser.add_argument('--base_model', type=str, default=None, help='Base model, e.g., mult_vae / cvga / l_diffrec')
    parser.add_argument('--strategy', type=str, default=None, help='Alignment strategy, e.g., mddm / cpdm / godm')
    parser.add_argument('--dataset', type=str, default='amazon', help='Dataset name')
    parser.add_argument('--device', type=str, default='cuda', help='cpu or cuda')
    parser.add_argument('--seed', type=int, default=None, help='Device number')
    parser.add_argument('--cuda', type=str, default='0', help='Device number')
    args, _ = parser.parse_known_args()

    # cuda
    if args.device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    # model name
    if model is not None:
        model_name = model.lower()
    elif args.model is not None:
        model_name = args.model.lower()
    else:
        model_name = 'default'
        # print("Read the default (blank) configuration.")

    # dataset
    if dataset is not None:
        args.dataset = dataset

    # find yml file
    if not os.path.exists('../encoder/config/modelconf/{}.yml'.format(model_name)):
        raise Exception("Please create the yaml file for your model first.")

    # read yml file
    with open('../encoder/config/modelconf/{}.yml'.format(model_name), encoding='utf-8') as f:
        config_data = f.read()
        configs = yaml.safe_load(config_data)
        configs['model']['name'] = configs['model']['name'].lower()
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
