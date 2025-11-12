import torch
from torch import nn


class DenoiseNet(nn.Module):
	"""Denoising network for diffusion process in latent space"""
	def __init__(self, latent_dim, hidden_dim=512):
		super().__init__()
		self.latent_dim = latent_dim
		# Time embedding
		self.time_embed = nn.Sequential(
			nn.Linear(1, hidden_dim),
			nn.SiLU(),
			nn.Linear(hidden_dim, hidden_dim)
		)
		# Denoising network (DNN)
		self.net = nn.Sequential(
			nn.Linear(latent_dim + hidden_dim, hidden_dim),
			nn.SiLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.SiLU(),
			nn.Linear(hidden_dim, latent_dim)
		)
	
	def forward(self, x, t):
		# t: [batch_size] -> [batch_size, 1]
		t_emb = self.time_embed(t.float().unsqueeze(-1))
		# Concatenate x and time embedding
		x_t = torch.cat([x, t_emb], dim=-1)
		return self.net(x_t)

