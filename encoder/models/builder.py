from config.configurator import configs
import importlib
from torch import nn


class CompositeModel(nn.Module):
	def __init__(self, base_model, strategy):
		super().__init__()
		self.base = base_model
		self.strategy = strategy
		
		# Register any nn.Module components in strategy as submodules
		# This ensures they are included in model.parameters() for training
		for name, module in strategy.__dict__.items():
			if isinstance(module, nn.Module):
				self.add_module(f'strategy_{name}', module)

	def cal_loss(self, batch_users, data):
		inter = self.base.forward_for_loss(batch_users, data)
		loss, loss_dict = self.strategy.compute_loss(self.base.decode, inter, data)
		
		# Add diffusion loss if present (for L-DiffRec)
		if 'diffusion_loss' in inter and inter['diffusion_loss'] is not None:
			diff_loss = inter['diffusion_loss'].mean()
			loss = loss + diff_loss
			loss_dict['diffusion_loss'] = diff_loss.item()
		
		return loss, loss_dict

	def _mask_predict(self, full_preds, train_mask):
		return self.base._mask_predict(full_preds, train_mask)

	def full_predict(self, batch_data):
		pck_users, train_mask = batch_data
		inter = self.base.forward_for_predict(pck_users, train_mask)

		# CVGA: use z_cvga if available (already decoded in forward_for_predict)
		if 'full_preds' in inter:
			return inter['full_preds']
		
		# L-DiffRec: use diffused/sampled z
		if 'z_sampled' in inter:
			z = inter['z_sampled']
		# CVGA: use z_cvga
		elif 'z_cvga' in inter:
			z = inter['z_cvga']
		# Standard VAE: use mean
		else:
			# 与训练路径保持一致：相加后解码
			mu = inter['mu_src'] + inter['mu_llm']
			logvar = inter['logvar_src'] + inter['logvar_llm']
			z = mu  # 评测阶段使用均值（与原实现一致）

		# CVGA's decode needs user_indices, others don't
		if 'z_cvga' in inter:
			recon_x = self.base.decode(z, pck_users)
		else:
			recon_x = self.base.decode(z)
		full_preds = self.base._mask_predict(recon_x, inter['train_mask'])
		return full_preds


def build_base_model(data_handler, base_name):
	module_path = ".".join(['models', 'bases', base_name])
	module = importlib.import_module(module_path)
	# 约定：类名 MultVAEBackbone / CvgABackbone / LDiffRecBackbone 等
	for attr in dir(module):
		if attr.lower().endswith('backbone'):
			backbone_class = getattr(module, attr)
			print(f"[DEBUG] Building base model: {base_name} -> {backbone_class.__name__}")
			return backbone_class(data_handler)
	raise NotImplementedError('Backbone for {} not found'.format(base_name))


def build_strategy(strategy_name):
	module_path = ".".join(['models', 'strategies', strategy_name])
	module = importlib.import_module(module_path)
	for attr in dir(module):
		if attr.lower().endswith('strategy'):
			return getattr(module, attr)()
	raise NotImplementedError('Strategy {} not found'.format(strategy_name))


def compose_model(base_model, strategy):
	return CompositeModel(base_model, strategy)

