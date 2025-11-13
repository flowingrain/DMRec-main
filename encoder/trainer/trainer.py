import os
import time
import random
import numpy as np
from numpy import random
from copy import deepcopy
import torch
import torch.optim as optim
from trainer.metrics import Metric
from models.bulid_model import build_model
from config.configurator import configs
from .utils import DisabledSummaryWriter, log_exceptions


def init_seed():
    if 'reproducible' in configs['train']:
        if configs['train']['reproducible']:
            seed = configs['train']['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def naive_sparse2tensor(data):
    return torch.FloatTensor(data.toarray())

update_counts = 0


class Trainer(object):
    def __init__(self, data_handler, logger):
        self.data_handler = data_handler
        self.logger = logger
        self.metric = Metric()

    def create_optimizer(self, model):
        optim_config = configs['optimizer']
        if optim_config['name'] == 'adam':
            self.optimizer = optim.Adam(model.parameters(), lr=optim_config['lr'],
                                        weight_decay=optim_config['weight_decay'])
        
        # 学习率调度器（可选，针对 CVGA 等模型）
        use_scheduler = optim_config.get('use_scheduler', False)
        if use_scheduler:
            scheduler_type = optim_config.get('scheduler_type', 'plateau')
            if scheduler_type == 'plateau':
                # ReduceLROnPlateau: 当验证指标不提升时降低学习率
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode='max',  # 监控 recall（越大越好）
                    factor=optim_config.get('scheduler_factor', 0.5),
                    patience=optim_config.get('scheduler_patience', 10),
                    verbose=True
                )
            elif scheduler_type == 'cosine':
                # CosineAnnealingLR: 余弦退火
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=configs['train']['epoch'],
                    eta_min=optim_config.get('min_lr', 1e-6)
                )
            else:
                self.scheduler = None
        else:
            self.scheduler = None

    def train_epoch(self, model, epoch_idx):
        global update_counts

        train_list = list(range(configs['data']['user_num']))
        np.random.shuffle(train_list)

        # for recording loss
        loss_log_dict = {}
        ep_loss = 0
        # start this epoch
        model.train()
        for batch_id, start_id in enumerate(range(0, configs['data']['user_num'], configs['train']['batch_size'])):
            self.optimizer.zero_grad()
            end_id = min(start_id + configs['train']['batch_size'], configs['data']['user_num'])
            batch_data = self.data_handler.train_data[train_list[start_id:end_id]]
            data = naive_sparse2tensor(batch_data).to(configs['device'])

            # Pass epoch information for adaptive sampling (if supported)
            max_epoch = configs['train']['epoch']
            loss, loss_dict = model.cal_loss(train_list[start_id:end_id], data, epoch_idx=epoch_idx, max_epoch=max_epoch)
            ep_loss += loss.item()
            loss.backward()
            
            # 梯度裁剪（针对 CVGA 等 GNN 模型，可选）
            clip_grad = configs['optimizer'].get('clip_grad', None)
            if clip_grad is not None and clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            
            self.optimizer.step()

            update_counts += 1

            # record loss
            for loss_name in loss_dict:
                _loss_val = float(loss_dict[loss_name])
                if loss_name not in loss_log_dict:
                    loss_log_dict[loss_name] = _loss_val
                else:
                    loss_log_dict[loss_name] += _loss_val

        if 'log_loss' in configs['train'] and configs['train']['log_loss']:
            self.logger.log(loss_log_dict, save_to_log=False, print_to_console=True)

    @log_exceptions
    def train(self, model):
        now_patience = 0
        best_epoch = 0
        best_recall = -1e9
        self.create_optimizer(model)
        train_config = configs['train']

        time_list = []

        for epoch_idx in range(train_config['epoch']):
            # train
            start_time = time.time()
            self.train_epoch(model, epoch_idx)
            end_time = time.time()
            time_list.append(end_time - start_time)

            # evaluate
            if epoch_idx % train_config['test_step'] == 0:
                eval_result = self.evaluate(model, epoch_idx)
                
                # 更新学习率调度器（如果使用 ReduceLROnPlateau）
                if self.scheduler is not None:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        # ReduceLROnPlateau 需要传入监控的指标
                        self.scheduler.step(eval_result['recall'][-1])
                    else:
                        # 其他调度器（如 CosineAnnealingLR）直接 step
                        self.scheduler.step()

                if eval_result['recall'][-1] > best_recall:
                    now_patience = 0
                    best_epoch = epoch_idx
                    best_recall = eval_result['recall'][-1]
                    best_state_dict = deepcopy(model.state_dict())
                else:
                    now_patience += 1

                # early stop
                if now_patience == configs['train']['patience']:
                    break

        # evaluation again
        model = build_model(self.data_handler).to(configs['device'])
        model.load_state_dict(best_state_dict)
        self.evaluate(model)

        # final test
        model = build_model(self.data_handler).to(configs['device'])
        model.load_state_dict(best_state_dict)
        test_result = self.test(model)

        # save result
        self.save_model(model)
        self.logger.log("Best Epoch {}. Final test result: {}.".format(best_epoch, test_result))

    @log_exceptions
    def evaluate(self, model, epoch_idx=None):
        model.eval()
        eval_result = self.metric.eval(model, self.data_handler.valid_dataloader)
        self.logger.log_eval(eval_result, configs['test']['k'], data_type='Validation set', epoch_idx=epoch_idx)
        return eval_result

    @log_exceptions
    def test(self, model):
        model.eval()
        eval_result = self.metric.eval(model, self.data_handler.test_dataloader)
        self.logger.log_eval(eval_result, configs['test']['k'], data_type='Test set')
        return eval_result

    @log_exceptions
    def test_save(self, model):
        model.eval()
        eval_result, candidate_set = self.metric.eval_save(model, self.data_handler.test_dataloader)
        self.logger.log_eval(eval_result, configs['test']['k'], data_type='Test set')
        return eval_result, candidate_set

    def save_model(self, model):
        if configs['train']['save_model']:
            model_state_dict = model.state_dict()
            model_name = configs['model']['name']
            if not configs['tune']['enable']:
                save_dir_path = './encoder/checkpoint/{}'.format(model_name)

                if not os.path.exists(save_dir_path):
                    os.makedirs(save_dir_path)
                torch.save(model_state_dict,
                           '{}/{}-{}-{}.pth'.format(save_dir_path, model_name, configs['data']['name'],
                                                    configs['train']['seed']))
                self.logger.log("Save model parameters to {}".format(
                    '{}/{}-{}-{}.pth'.format(save_dir_path, model_name, configs['data']['name'],
                                             configs['train']['seed'])))
            else:
                save_dir_path = './encoder/checkpoint/{}/tune'.format(model_name)

                if not os.path.exists(save_dir_path):
                    os.makedirs(save_dir_path)
                now_para_str = configs['tune']['now_para_str']
                torch.save(
                    model_state_dict, '{}/{}-{}.pth'.format(save_dir_path, model_name, now_para_str))
                self.logger.log("Save model parameters to {}".format(
                    '{}/{}-{}.pth'.format(save_dir_path, model_name, now_para_str)))

    def load_model(self, model):
        if 'pretrain_path' in configs['train']:
            pretrain_path = configs['train']['pretrain_path']
            model.load_state_dict(torch.load(pretrain_path))
            self.logger.log(
                "Load model parameters from {}".format(pretrain_path))
