from config.configurator import configs
import importlib
from models.builder import build_base_model, build_strategy, compose_model

def build_model(data_handler):
    model_type = configs['data']['type']
    model_cfg = configs['model']
    # 新路径：base + strategy 组合（优先）
    base_name = model_cfg.get('base', None)
    strategy_name = model_cfg.get('strategy', None)
    if base_name and strategy_name:
        base_model = build_base_model(data_handler, base_name)
        strategy = build_strategy(strategy_name)
        return compose_model(base_model, strategy)

    # 兼容旧路径：model.name，比如 mult_vae_mddm
    model_name = model_cfg['name']
    # 尝试自动拆分
    if '_' in model_name:
        parts = model_name.split('_')
        # 经验：最后一段为策略（godm/cpdm/mddm），其余为基座名
        strat = parts[-1]
        base = '_'.join(parts[:-1])
        try:
            base_model = build_base_model(data_handler, base)
            strategy = build_strategy(strat)
            return compose_model(base_model, strategy)
        except Exception:
            pass

    # 回退到旧的按模块名直载（保留原行为）
    module_path = ".".join(['models', model_type, model_name])
    if importlib.util.find_spec(module_path) is None:
        raise NotImplementedError('Model {} is not implemented'.format(model_name))
    module = importlib.import_module(module_path)
    for attr in dir(module):
        if attr.lower() == model_name.lower():
            return getattr(module, attr)(data_handler)
    else:
        raise NotImplementedError('Model Class {} is not defined in {}'.format(model_name, module_path))



