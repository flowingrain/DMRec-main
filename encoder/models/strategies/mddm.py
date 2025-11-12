import torch
import torch.nn.functional as F
from config.configurator import configs


class MDDMStrategy(object):
	def __init__(self):
		hyper = configs['model']
		if configs['data']['name'] in hyper:
			hyper = hyper[configs['data']['name']]
		self.beta = hyper.get('beta', 0.6)

	def loss(self, inter, data):
		mu_src = inter['mu_src']
		mu_llm = inter['mu_llm']
		logvar_src = inter['logvar_src']
		logvar_llm = inter['logvar_llm']

		# 统一设计：先相加再做重构与匹配
		mu = mu_src + mu_llm
		logvar = logvar_src + logvar_llm

		std = torch.exp(0.5 * logvar)
		eps = torch.randn_like(std)
		z = eps.mul(std) + mu

		# 解码（注意：由调用方提供 decode）
		return mu, logvar, z

	def compute_loss(self, decode_fn, inter, data):
		mu_src = inter['mu_src']
		mu_llm = inter['mu_llm']
		logvar_src = inter['logvar_src']
		logvar_llm = inter['logvar_llm']

		# CVGA: already has recon_x computed
		if 'recon_x' in inter:
			recon_x = inter['recon_x']
			# CVGA: use mu_src and logvar_src directly (no mu_llm)
			KLD = - 0.5 * torch.mean(torch.sum(1 + logvar_src - mu_src.pow(2) - logvar_src.exp(), dim=1))
		else:
			# Standard VAE path: 相加路径
			mu = mu_src + mu_llm
			logvar = logvar_src + logvar_llm

			# Use diffused z if available (for L-DiffRec), otherwise reparameterize
			if 'z_diffused' in inter:
				z = inter['z_diffused']
			else:
				std = torch.exp(0.5 * logvar)
				eps = torch.randn_like(std)
				z = eps.mul(std) + mu

			# CVGA's decode needs user_indices
			if 'user_indices' in inter:
				recon_x = decode_fn(z, inter['user_indices'])
			else:
				recon_x = decode_fn(z)
			
			# Standard VAE KLD computation
			KLD = - 0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
			KLD_llm = - 0.5 * torch.mean(torch.sum(
				1 + torch.log(logvar.exp()/(logvar_llm.exp() + 10e-8) + 10e-8)
				- (mu - mu_llm).pow(2)/(logvar_llm.exp() + 10e-8)
				- logvar.exp()/(logvar_llm.exp() + 10e-8), dim=1))
			KLD = self.beta * KLD + (1 - self.beta) * KLD_llm
		
		BCE = - torch.mean(torch.sum(F.log_softmax(recon_x, 1) * data, -1))

		loss = BCE + KLD
		losses = {'rec_loss': BCE, 'reg_loss': KLD}
		return loss, losses

