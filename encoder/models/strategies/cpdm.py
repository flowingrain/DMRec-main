import torch
import torch.nn.functional as F
from config.configurator import configs


class CPDMStrategy(object):
	def __init__(self):
		hyper = configs['model']
		if configs['data']['name'] in hyper:
			hyper = hyper[configs['data']['name']]
		self.beta = hyper.get('beta', 0.6)

	def compute_loss(self, decode_fn, inter, data):
		# Unpack intermediates from backbone
		mu_src = inter['mu_src']
		mu_llm = inter['mu_llm']
		logvar_src = inter['logvar_src']
		logvar_llm = inter['logvar_llm']

		# Unified design: addition then downstream tasks
		mu = mu_src + mu_llm
		logvar = logvar_src + logvar_llm

		# Use diffused z if available (for L-DiffRec), otherwise reparameterize
		if 'z_diffused' in inter:
			z = inter['z_diffused']
		else:
			std = torch.exp(0.5 * logvar)
			eps = torch.randn_like(std)
			z = eps.mul(std) + mu

		# Reconstruction
		recon_x = decode_fn(z)
		BCE = - torch.mean(torch.sum(F.log_softmax(recon_x, 1) * data, -1))

		# KL terms
		KLD = - 0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

		mu_mix = (mu + mu_llm) / 2
		logvar_mix = (logvar + logvar_llm) / 2

		KLD_1 = - 0.5 * torch.mean(torch.sum(
			1 + torch.log(logvar.exp()/(logvar_mix.exp() + 10e-8) + 10e-8)
			- (mu - mu_mix).pow(2)/(logvar_mix.exp() + 10e-8)
			- logvar.exp()/(logvar_mix.exp() + 10e-8), dim=1))

		KLD_2 = - 0.5 * torch.mean(torch.sum(
			1 + torch.log(logvar_llm.exp()/(logvar_mix.exp() + 10e-8) + 10e-8)
			- (mu_llm - mu_mix).pow(2)/(logvar_mix.exp() + 10e-8)
			- logvar_llm.exp()/(logvar_mix.exp() + 10e-8), dim=1))

		KLD = KLD + self.beta * (KLD_1 + KLD_2)

		loss = BCE + KLD
		losses = {'rec_loss': BCE, 'reg_loss': KLD}
		return loss, losses

