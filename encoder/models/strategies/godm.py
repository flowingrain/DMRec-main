import torch
import torch.nn.functional as F
from config.configurator import configs


class GODMStrategy(object):
	def __init__(self):
		hyper = configs['model']
		if configs['data']['name'] in hyper:
			hyper = hyper[configs['data']['name']]
		# trade-off coefficient
		self.beta = hyper.get('beta', 0.6)

	def compute_loss(self, decode_fn, inter, data):
		# Unpack intermediates
		mu_src = inter['mu_src']
		mu_llm = inter['mu_llm']
		logvar_src = inter['logvar_src']
		logvar_llm = inter['logvar_llm']

		# Addition path then reconstruction
		mu = mu_src + mu_llm
		logvar = logvar_src + logvar_llm

		# Use diffused z if available (for L-DiffRec), otherwise reparameterize
		if 'z_diffused' in inter:
			z = inter['z_diffused']
		else:
			std = torch.exp(0.5 * logvar)
			eps = torch.randn_like(std)
			z = eps.mul(std) + mu

		recon_x = decode_fn(z)
		BCE = - torch.mean(torch.sum(F.log_softmax(recon_x, 1) * data, -1))

		KLD = - 0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

		# Wasserstein-like distance between two Gaussians (mean and std mismatch)
		mean_diff = torch.norm(mu_src - mu_llm, dim=1) ** 2
		var_diff = torch.norm(torch.exp(0.5 * logvar_src) - torch.exp(0.5 * logvar_llm), dim=1) ** 2
		WD = torch.mean(torch.sqrt(mean_diff + var_diff + 10e-8))

		reg = KLD + self.beta * WD
		loss = BCE + reg
		losses = {'rec_loss': BCE, 'reg_loss': reg}
		return loss, losses

