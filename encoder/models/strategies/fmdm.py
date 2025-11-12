import torch
import torch.nn as nn
import torch.nn.functional as F
from config.configurator import configs


class VectorFieldNet(nn.Module):
	"""
	Neural network to learn the vector field v_t(z) for flow matching.
	Predicts the velocity field that transports points from p_φ (LLM space) to q_φ (collaborative space).
	"""
	def __init__(self, latent_dim, hidden_dims=[256, 256], time_embed_dim=128):
		super(VectorFieldNet, self).__init__()
		self.latent_dim = latent_dim
		self.time_embed_dim = time_embed_dim
		
		# Time embedding network
		self.time_embed = nn.Sequential(
			nn.Linear(1, time_embed_dim),
			nn.SiLU(),
			nn.Linear(time_embed_dim, time_embed_dim)
		)
		
		# Main network: takes [z_t, time_embed] -> v_t
		dims = [latent_dim + time_embed_dim] + hidden_dims + [latent_dim]
		layers = []
		for i in range(len(dims) - 1):
			layers.append(nn.Linear(dims[i], dims[i+1]))
			if i < len(dims) - 2:
				layers.append(nn.SiLU())
				layers.append(nn.Dropout(0.1))
		self.net = nn.Sequential(*layers)
	
	def forward(self, z_t, t):
		"""
		Args:
			z_t: [batch_size, latent_dim] point at time t
			t: [batch_size, 1] or scalar time value in [0, 1]
		
		Returns:
			v_t: [batch_size, latent_dim] predicted vector field
		"""
		if isinstance(t, (int, float)):
			t = torch.full((z_t.shape[0], 1), t, device=z_t.device, dtype=z_t.dtype)
		elif t.dim() == 0:
			t = t.unsqueeze(0).unsqueeze(0).expand(z_t.shape[0], -1)
		elif t.dim() == 1:
			t = t.unsqueeze(1)
		
		# Time embedding
		time_emb = self.time_embed(t)  # [batch_size, time_embed_dim]
		
		# Concatenate z_t and time embedding
		h = torch.cat([z_t, time_emb], dim=1)  # [batch_size, latent_dim + time_embed_dim]
		
		# Predict vector field
		v_t = self.net(h)  # [batch_size, latent_dim]
		
		return v_t


class FMDMStrategy(object):
	"""
	Flow Matching for Distribution Matching (FMDM) strategy.
	Aligns LLM space (p_φ) and collaborative space (q_φ) using flow matching.
	
	Core idea:
	- Learn a vector field v_t(z) that transports points from p_φ (t=0) to q_φ (t=1)
	- Use straight-line paths: z_t = (1-t) * z_0 + t * z_1
	- Loss: ||v_t(z_t) - (z_1 - z_0)||^2
	
	Note: Flow matching is used as the alignment regularization mechanism.
	Reconstruction uses the standard combined distribution approach for compatibility.
	"""
	def __init__(self):
		hyper = configs['model']
		if configs['data']['name'] in hyper:
			hyper = hyper[configs['data']['name']]
		
		# Hyperparameters
		self.beta = hyper.get('beta', 0.6)  # Weight for flow matching alignment regularization
		self.latent_dim = hyper.get('latent_dim', 200)  # Latent dimension
		
		# Initialize vector field network
		hidden_dims = hyper.get('flow_hidden_dims', [256, 256])
		self.vector_field = VectorFieldNet(
			latent_dim=self.latent_dim,
			hidden_dims=hidden_dims,
			time_embed_dim=hyper.get('flow_time_embed_dim', 128)
		).to(configs['device'])
	
	def sample_path(self, z_0, z_1, t):
		"""
		Sample point on straight-line path between z_0 and z_1 at time t.
		
		Args:
			z_0: [batch_size, latent_dim] point from p_φ (LLM space)
			z_1: [batch_size, latent_dim] point from q_φ (collaborative space)
			t: [batch_size, 1] or scalar time in [0, 1]
		
		Returns:
			z_t: [batch_size, latent_dim] point on path at time t
		"""
		if isinstance(t, (int, float)):
			t = torch.tensor(t, device=z_0.device, dtype=z_0.dtype)
		if t.dim() == 0:
			t = t.unsqueeze(0).unsqueeze(0).expand(z_0.shape[0], -1)
		elif t.dim() == 1:
			t = t.unsqueeze(1)
		
		# Straight-line path: z_t = (1-t) * z_0 + t * z_1
		z_t = (1 - t) * z_0 + t * z_1
		return z_t
	
	def compute_flow_loss(self, mu_src, mu_llm, logvar_src, logvar_llm):
		"""
		Compute flow matching loss.
		
		Args:
			mu_src: [batch_size, latent_dim] mean from collaborative space
			mu_llm: [batch_size, latent_dim] mean from LLM space
			logvar_src: [batch_size, latent_dim] logvar from collaborative space
			logvar_llm: [batch_size, latent_dim] logvar from LLM space
		
		Returns:
			flow_loss: scalar flow matching loss
		"""
		batch_size = mu_src.shape[0]
		device = mu_src.device
		
		# Sample z_0 from p_φ (LLM space)
		std_llm = torch.exp(0.5 * logvar_llm)
		eps_0 = torch.randn_like(std_llm)
		z_0 = eps_0.mul(std_llm) + mu_llm
		
		# Sample z_1 from q_φ (collaborative space)
		std_src = torch.exp(0.5 * logvar_src)
		eps_1 = torch.randn_like(std_src)
		z_1 = eps_1.mul(std_src) + mu_src
		
		# Sample random time t ~ Uniform(0, 1)
		t = torch.rand(batch_size, 1, device=device)
		
		# Compute point on path: z_t = (1-t) * z_0 + t * z_1
		z_t = self.sample_path(z_0, z_1, t)
		
		# True velocity: v_t = z_1 - z_0 (for straight-line paths)
		v_true = z_1 - z_0
		
		# Predicted velocity from vector field network
		v_pred = self.vector_field(z_t, t.squeeze(1))
		
		# Flow matching loss: ||v_pred - v_true||^2
		flow_loss = torch.mean(torch.sum((v_pred - v_true) ** 2, dim=1))
		
		return flow_loss
	
	def transport_sample(self, z_0, num_steps=10):
		"""
		Transport a sample from LLM space (z_0) to collaborative space using learned vector field.
		Uses Euler method to integrate the ODE: dz/dt = v_t(z_t)
		
		Args:
			z_0: [batch_size, latent_dim] starting point in LLM space
			num_steps: number of integration steps
		
		Returns:
			z_1: [batch_size, latent_dim] transported point in collaborative space
		"""
		z_t = z_0
		dt = 1.0 / num_steps
		
		for i in range(num_steps):
			t = i * dt
			v_t = self.vector_field(z_t, t)
			z_t = z_t + dt * v_t
		
		return z_t
	
	def compute_loss(self, decode_fn, inter, data):
		"""
		Compute total loss including flow matching regularization.
		
		Args:
			decode_fn: decoder function
			inter: intermediate outputs from base model
			data: ground truth interaction data
		
		Returns:
			loss: total loss
			losses: dict of individual loss terms
		"""
		# Unpack intermediates
		mu_src = inter['mu_src']
		mu_llm = inter['mu_llm']
		logvar_src = inter['logvar_src']
		logvar_llm = inter['logvar_llm']
		
		# For reconstruction: use combined distribution (standard approach)
		# Flow matching is used for alignment regularization, not for reconstruction
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
		
		# Reconstruction loss
		BCE = - torch.mean(torch.sum(F.log_softmax(recon_x, 1) * data, -1))
		
		# Standard VAE KLD
		mu = mu_src + mu_llm
		logvar = logvar_src + logvar_llm
		KLD = - 0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
		
		# Flow matching loss: learn vector field to transport from p_φ to q_φ
		# This is the alignment regularization term (similar to WD in GODM, KLD_1+KLD_2 in CPDM)
		flow_loss = self.compute_flow_loss(mu_src, mu_llm, logvar_src, logvar_llm)
		
		# Total regularization: KLD + beta * flow_loss (consistent with other strategies)
		# Similar to GODM: reg = KLD + beta * WD
		reg = KLD + self.beta * flow_loss
		
		# Total loss: reconstruction + regularization
		total_loss = BCE + reg
		
		losses = {
			'rec_loss': BCE,
			'reg_loss': reg,
			'kld': KLD,
			'flow_loss': flow_loss
		}
		
		return total_loss, losses

