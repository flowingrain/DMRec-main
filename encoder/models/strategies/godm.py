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

		# GODM strategy: compute alignment term (Wasserstein distance)
		# Wasserstein-like distance between two Gaussians (mean and std mismatch)
		mean_diff = torch.norm(mu_src - mu_llm, dim=1) ** 2
		var_diff = torch.norm(torch.exp(0.5 * logvar_src) - torch.exp(0.5 * logvar_llm), dim=1) ** 2
		WD = torch.mean(torch.sqrt(mean_diff + var_diff + 10e-8))

		# Strategy only computes alignment term
		# Total loss combination is done in builder: loss = bce + kld + beta * WD
		alignment_term = self.beta * WD
		
		return alignment_term, {'wd': WD}

