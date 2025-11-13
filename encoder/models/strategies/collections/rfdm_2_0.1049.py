import torch
import torch.nn as nn
import torch.nn.functional as F
from config.configurator import configs


class VectorFieldNet(nn.Module):
	"""
	Neural network to learn the vector field v_t(z) for Rectified Flow.
	Predicts the velocity field that transports points from p_φ (LLM space) to q_φ (collaborative space).
	"""
	def __init__(self, latent_dim, hidden_dims=[256, 256], time_embed_dim=128, use_residual=False):
		super(VectorFieldNet, self).__init__()
		self.latent_dim = latent_dim
		self.time_embed_dim = time_embed_dim
		self.use_residual = use_residual
		
		# Time embedding network (improved for Rectified Flow)
		self.time_embed = nn.Sequential(
			nn.Linear(1, time_embed_dim),
			nn.SiLU(),
			nn.Linear(time_embed_dim, time_embed_dim)
		)
		
		# Main network: takes [z_t, time_embed] -> v_t
		dims = [latent_dim + time_embed_dim] + hidden_dims + [latent_dim]
		
		if self.use_residual:
			# With residual connections: build layers separately for skip connections
			self.layers = nn.ModuleList()
			self.activations = nn.ModuleList()
			self.residual_flags = []
			for i in range(len(dims) - 1):
				self.layers.append(nn.Linear(dims[i], dims[i+1]))
				if i < len(dims) - 2:
					self.activations.append(nn.SiLU())
					# Can use residual if input and output dims match (after first layer)
					can_residual = (i > 0 and dims[i] == dims[i+1])
					self.residual_flags.append(can_residual)
		else:
			# Without residual: use Sequential for efficiency
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
		if self.use_residual:
			# With residual connections
			x = h
			for i, (layer, activation, can_residual) in enumerate(zip(self.layers[:-1], self.activations, self.residual_flags)):
				out = layer(x)
				if can_residual:
					out = out + x  # Residual connection
				x = activation(out)
			v_t = self.layers[-1](x)  # Final layer without activation
		else:
			# Without residual: use Sequential
			v_t = self.net(h)  # [batch_size, latent_dim]
		
		return v_t


class RectifiedFMDMStrategy(object):
	"""
	Rectified Flow Matching for Distribution Matching (Rectified FMDM) strategy.
	Aligns LLM space (p_φ) and collaborative space (q_φ) using Rectified Flow.
	
	Key improvements over standard Flow Matching:
	1. Straight-line paths: z_t = (1-t) * z_0 + t * z_1 (already in FMDM)
	2. Reflow: Iterative refinement to straighten paths further
	3. More stable training with better initialization
	4. Single-step or few-step inference capability
	
	Core idea:
	- Learn a vector field v_t(z) that transports points from p_φ (t=0) to q_φ (t=1)
	- Use straight-line paths: z_t = (1-t) * z_0 + t * z_1
	- Loss: ||v_t(z_t) - (z_1 - z_0)||^2 (same as FMDM, but with better training)
	- Optional: Reflow to further straighten paths
	"""
	def __init__(self):
		hyper = configs['model']
		if configs['data']['name'] in hyper:
			hyper = hyper[configs['data']['name']]
		
		# Hyperparameters
		self.beta = hyper.get('beta', 0.6)  # Weight for flow matching alignment regularization
		self.latent_dim = hyper.get('latent_dim', 200)  # Latent dimension
		self.use_reflow = hyper.get('use_reflow', True)  # Whether to use reflow refinement
		self.reflow_steps = hyper.get('reflow_steps', 1)  # Number of reflow iterations
		self.use_adaptive_sampling = hyper.get('use_adaptive_sampling', False)
		self.adaptive_epoch_threshold = hyper.get('adaptive_epoch_threshold', 0.3)
		self.use_residual = hyper.get('use_residual', False)
		
		# Initialize vector field network
		hidden_dims = hyper.get('flow_hidden_dims', [256, 256])
		time_embed_dim = hyper.get('flow_time_embed_dim', 128)
		self.vector_field = VectorFieldNet(
			latent_dim=self.latent_dim,
			hidden_dims=hidden_dims,
			time_embed_dim=time_embed_dim,
			use_residual=self.use_residual
		).to(configs['device'])
		
		# Reflow state: track current reflow iteration
		# During training, we can use reflow to refine z_1 using the learned vector field
		# Auto-enable reflow after this fraction of training (0.0 = never auto-enable, 1.0 = always enabled)
		# Set to None to disable auto-enable (requires manual update_reflow_step call)
		self.reflow_auto_enable_threshold = hyper.get('reflow_auto_enable_threshold', None)
		
		# Initialize reflow step:
		# - If use_reflow=True and no auto-enable threshold, enable immediately (step=1)
		# - Otherwise, start with step=0 (no reflow)
		if self.use_reflow and self.reflow_auto_enable_threshold is None:
			self.current_reflow_step = 1  # Enable reflow immediately
		else:
			self.current_reflow_step = 0  # Start without reflow, enable later if threshold is set
		
		print("[DEBUG] RFDM hyper:", {
			'use_reflow': self.use_reflow,
			'reflow_steps': self.reflow_steps,
			'current_reflow_step': self.current_reflow_step,
			'reflow_auto_enable_threshold': self.reflow_auto_enable_threshold,
			'use_adaptive_sampling': self.use_adaptive_sampling,
			'adaptive_epoch_threshold': self.adaptive_epoch_threshold,
			'use_residual': self.use_residual,
			'hidden_dims': hidden_dims,
			'time_embed_dim': time_embed_dim,
			'beta': self.beta,
		})
		# Initialize with better initialization for stability
		self._init_weights()
	
	def _init_weights(self):
		"""Initialize weights for better training stability (Rectified Flow improvement)"""
		for m in self.vector_field.modules():
			if isinstance(m, nn.Linear):
				# Xavier initialization for better stability
				nn.init.xavier_uniform_(m.weight)
				if m.bias is not None:
					nn.init.zeros_(m.bias)
	
	def sample_path(self, z_0, z_1, t):
		"""
		Sample point on straight-line path between z_0 and z_1 at time t.
		This is the core of Rectified Flow: straight paths.
		
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
		# This is the key of Rectified Flow
		z_t = (1 - t) * z_0 + t * z_1
		return z_t
	
	def sample_time_adaptive(self, batch_size, device, epoch_idx=None, max_epoch=None):
		"""
		Adaptive time sampling: focus on endpoints early, uniform later.
		
		Args:
			batch_size: batch size
			device: device
			epoch_idx: current epoch index (0-based)
			max_epoch: total number of epochs
		
		Returns:
			t: [batch_size, 1] time samples in [0, 1]
		"""
		if not self.use_adaptive_sampling or epoch_idx is None or max_epoch is None:
			# Uniform sampling (default)
			return torch.rand(batch_size, 1, device=device)
		
		# Compute progress: 0 at start, 1 at end
		progress = epoch_idx / max_epoch if max_epoch > 0 else 0.0
		threshold = self.adaptive_epoch_threshold
		
		if progress < threshold:
			# Early training: focus on endpoints (t near 0 or 1)
			# Use beta distribution with alpha=beta<1 to concentrate at endpoints
			alpha = 0.3  # Lower alpha = more concentration at endpoints
			beta = 0.3
			t = torch.distributions.Beta(alpha, beta).sample((batch_size, 1)).to(device)
		else:
			# Later training: uniform sampling (optimal for straight paths)
			t = torch.rand(batch_size, 1, device=device)
		
		return t
	
	def compute_flow_loss(self, mu_src, mu_llm, logvar_src, logvar_llm, epoch_idx=None, max_epoch=None):
		"""
		Compute Rectified Flow matching loss.
		Same formulation as FMDM, but with potential improvements for stability.
		
		Args:
			mu_src: [batch_size, latent_dim] mean from collaborative space
			mu_llm: [batch_size, latent_dim] mean from LLM space
			logvar_src: [batch_size, latent_dim] logvar from collaborative space
			logvar_llm: [batch_size, latent_dim] logvar from LLM space
			epoch_idx: current epoch index (for adaptive sampling and auto reflow)
			max_epoch: total number of epochs (for adaptive sampling and auto reflow)
		
		Returns:
			flow_loss: scalar flow matching loss
		"""
		batch_size = mu_src.shape[0]
		device = mu_src.device
		
		# Auto-enable reflow based on training progress (if configured)
		if self.use_reflow and self.reflow_auto_enable_threshold is not None and epoch_idx is not None and max_epoch is not None:
			progress = epoch_idx / max_epoch if max_epoch > 0 else 0.0
			if progress >= self.reflow_auto_enable_threshold and self.current_reflow_step == 0:
				# Automatically enable reflow after reaching threshold
				self.current_reflow_step = 1
		
		# Sample z_0 from p_φ (LLM space)
		std_llm = torch.exp(0.5 * logvar_llm)
		eps_0 = torch.randn_like(std_llm)
		z_0 = eps_0.mul(std_llm) + mu_llm
		
		# Sample z_1 from q_φ (collaborative space)
		std_src = torch.exp(0.5 * logvar_src)
		eps_1 = torch.randn_like(std_src)
		z_1 = eps_1.mul(std_src) + mu_src
		
		# Reflow: refine z_1 using learned vector field (if enabled and reflow step > 0)
		if self.use_reflow and self.current_reflow_step > 0:
			# Use the learned vector field to transport z_0 to get a refined z_1
			# This helps straighten the paths iteratively
			# IMPORTANT: For Rectified Flow, we typically use 1 step for ODE integration
			# reflow_steps here controls the number of reflow iterations, not ODE steps
			# We use a fixed small number of ODE steps (e.g., 1-2) for stability
			with torch.no_grad():
				# Use fixed small number of ODE steps for stability (Rectified Flow works well with 1 step)
				# reflow_steps controls how many times we refine, not ODE integration steps
				ode_steps = min(2, self.reflow_steps)  # Cap at 2 for numerical stability
				z_1_refined = self.transport_sample(z_0, num_steps=ode_steps)
			
			# Mix original z_1 and refined z_1
			# The mixing ratio should be conservative to avoid disrupting training
			# Use a smaller ratio that doesn't depend on reflow_steps to avoid instability
			mix_ratio = 0.2  # Fixed conservative ratio
			z_1 = (1 - mix_ratio) * z_1 + mix_ratio * z_1_refined
		
		# Sample time t (adaptive or uniform)
		t = self.sample_time_adaptive(batch_size, device, epoch_idx, max_epoch)
		
		# Compute point on path: z_t = (1-t) * z_0 + t * z_1
		z_t = self.sample_path(z_0, z_1, t)
		
		# True velocity: v_t = z_1 - z_0 (for straight-line paths)
		# This is constant along the path, which is the key insight of Rectified Flow
		v_true = z_1 - z_0
		
		# Predicted velocity from vector field network
		v_pred = self.vector_field(z_t, t.squeeze(1))
		
		# Flow matching loss: ||v_pred - v_true||^2
		# Rectified Flow uses simple MSE loss, which is more stable
		flow_loss = torch.mean(torch.sum((v_pred - v_true) ** 2, dim=1))
		
		return flow_loss
	
	def update_reflow_step(self, step):
		"""
		Update the current reflow step.
		This can be called externally to control reflow iterations.
		
		Args:
			step: current reflow step (0 = no reflow, 1+ = reflow iterations)
		"""
		self.current_reflow_step = step
	
	def transport_sample(self, z_0, num_steps=1):
		"""
		Transport a sample from LLM space (z_0) to collaborative space using learned vector field.
		Rectified Flow advantage: can use single-step or very few steps due to straight paths.
		
		Uses Euler method to integrate the ODE: dz/dt = v_t(z_t)
		
		Args:
			z_0: [batch_size, latent_dim] starting point in LLM space
			num_steps: number of integration steps (can be 1 for Rectified Flow!)
		
		Returns:
			z_1: [batch_size, latent_dim] transported point in collaborative space
		"""
		z_t = z_0
		dt = 1.0 / num_steps
		
		for i in range(num_steps):
			# Time should go from 0 to 1
			# For step i, time is i * dt, but we need to ensure the last step reaches t=1
			t = min(i * dt, 1.0)  # Clamp to ensure t <= 1.0
			v_t = self.vector_field(z_t, t)
			z_t = z_t + dt * v_t
		
		return z_t
	
	def compute_loss(self, decode_fn, inter, data, epoch_idx=None, max_epoch=None):
		"""
		Compute total loss including Rectified Flow matching regularization.
		
		Args:
			decode_fn: decoder function
			inter: intermediate outputs from base model
			data: ground truth interaction data
			epoch_idx: current epoch index (for adaptive sampling)
			max_epoch: total number of epochs (for adaptive sampling)
		
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
		
		# Rectified Flow matching loss: learn vector field to transport from p_φ to q_φ
		# This is the alignment regularization term (similar to WD in GODM, KLD_1+KLD_2 in CPDM)
		# Pass epoch information for adaptive sampling
		flow_loss = self.compute_flow_loss(mu_src, mu_llm, logvar_src, logvar_llm, epoch_idx, max_epoch)
		
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

