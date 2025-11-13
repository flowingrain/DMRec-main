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
		
		# MDDM strategy: compute alignment term (KLD_llm)
		# 与原始实现保持一致：先相加再做对齐
		mu = mu_src + mu_llm
		logvar = logvar_src + logvar_llm
		
		# KLD_llm：KL(q(z|x) || q_llm(z|x))，衡量组合分布与LLM分布的差异
		KLD_llm = - 0.5 * torch.mean(torch.sum(
			1 + torch.log(logvar.exp()/(logvar_llm.exp() + 10e-8) + 10e-8)
			- (mu - mu_llm).pow(2)/(logvar_llm.exp() + 10e-8)
			- logvar.exp()/(logvar_llm.exp() + 10e-8), dim=1))
		
		# Strategy only computes alignment term
		# Total loss combination is done in builder: loss = bce + beta * kld + (1 - beta) * KLD_llm
		alignment_term = (1 - self.beta) * KLD_llm
		
		return alignment_term, {'kld_llm': KLD_llm}

