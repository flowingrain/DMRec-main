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

		# CPDM strategy: compute alignment terms (KLD_1 and KLD_2)
		# 与原始实现保持一致：先相加再做对齐
		mu = mu_src + mu_llm
		logvar = logvar_src + logvar_llm

		# CPDM特定：计算混合分布和两个KL散度
		mu_mix = (mu + mu_llm) / 2
		logvar_mix = (logvar + logvar_llm) / 2

		# KLD_1: 组合分布到混合分布
		KLD_1 = - 0.5 * torch.mean(torch.sum(
			1 + torch.log(logvar.exp()/(logvar_mix.exp() + 10e-8) + 10e-8)
			- (mu - mu_mix).pow(2)/(logvar_mix.exp() + 10e-8)
			- logvar.exp()/(logvar_mix.exp() + 10e-8), dim=1))

		# KLD_2: LLM分布到混合分布
		KLD_2 = - 0.5 * torch.mean(torch.sum(
			1 + torch.log(logvar_llm.exp()/(logvar_mix.exp() + 10e-8) + 10e-8)
			- (mu_llm - mu_mix).pow(2)/(logvar_mix.exp() + 10e-8)
			- logvar_llm.exp()/(logvar_mix.exp() + 10e-8), dim=1))

		# Strategy only computes alignment term
		# Total loss combination is done in builder: loss = bce + kld + beta * (KLD_1 + KLD_2)
		alignment_term = self.beta * (KLD_1 + KLD_2)

		return alignment_term, {'kld_1': KLD_1, 'kld_2': KLD_2}

