import torch
from torch import nn
import torch.nn.functional as F

from config.configurator import configs
from models.base_model import BaseModel
from models.layers.vae_encoder_decoder import VAEEncoderDecoder


class TimeEmbedding(nn.Module):
	"""
	Simple sinusoidal time embedding for scalar t in [0, 1].
	Used to condition the flow vector field on continuous time.
	"""

	def __init__(self, dim: int):
		super().__init__()
		self.dim = dim

	def forward(self, t: torch.Tensor) -> torch.Tensor:
		"""
		Args:
			t: [batch] or [batch, 1] tensor with values in [0, 1]
		Returns:
			time_emb: [batch, dim]
		"""
		if t.dim() == 1:
			t = t.unsqueeze(-1)  # [B, 1]
		device = t.device
		half = self.dim // 2
		freqs = torch.exp(
			torch.linspace(
				start=0.0,
				end=torch.log(torch.tensor(1000.0)),
				steps=half,
				device=device,
			)
		)  # [half]
		angles = t * freqs  # [B, half]
		emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # [B, 2*half]
		if self.dim % 2 == 1:
			# pad one zero dim if needed
			emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
		return emb


class RectifiedFlowField(nn.Module):
	"""
	Vector field v_theta(z, t) used to implement a rectified flow in latent space.

	This module learns to approximate the constant velocity (z1 - z0) along the
	straight line path:
	    z_t = (1 - t) * z0 + t * z1
	as described in the ReFlowRec paper.
	"""

	def __init__(self, latent_dim: int, hidden_dim: int = 512, time_dim: int = 64):
		super().__init__()
		self.latent_dim = latent_dim
		self.time_mlp = TimeEmbedding(time_dim)

		in_dim = latent_dim + time_dim
		self.net = nn.Sequential(
			nn.Linear(in_dim, hidden_dim),
			nn.SiLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.SiLU(),
			nn.Linear(hidden_dim, latent_dim),
		)

	def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
		"""
		Args:
			z_t: [batch, latent_dim] latent state at time t
			t:   [batch] or [batch, 1] time in [0, 1]
		Returns:
			v_t: [batch, latent_dim] predicted velocity at (z_t, t)
		"""
		time_emb = self.time_mlp(t)  # [B, time_dim]
		h = torch.cat([z_t, time_emb], dim=-1)
		return self.net(h)


class CFFlowBackbone(BaseModel):
	"""
	Collaborative Flow-based backbone (CF-Flow).

	Design goals:
	- Use a standard VAE-style encoder/decoder to model collaborative distribution
	  in a latent space z (like Mult-VAE).
	- Additionally learn a rectified flow vector field v_theta(z, t) in that
	  latent space which can be used to bridge a simple prior N(0, I) and the
	  collaborative posterior q_src(z|x).

	For compatibility with existing DMRec strategies, this backbone exposes the
	same intermediate keys as MultVAEBackbone:
	    mu_src, logvar_src, mu_llm, logvar_llm, recon_x, kld, bce

	The flow-matching loss is folded into the KLD term so that existing
	CompositeModel logic can automatically treat it as a regularization term.
	"""

	def __init__(self, data_handler):
		super().__init__(data_handler)
		self.data_handler = data_handler

		self.user_num = configs['data']['user_num']
		self.item_num = configs['data']['item_num']

		hyper = configs['model']
		if configs['data']['name'] in hyper:
			hyper = hyper[configs['data']['name']]

		self.dropout = hyper.get('dropout', 0.2)

		# Latent dimension (shared with strategies / language space)
		self.latent_dim = hyper.get('latent_dim', 200)

		# VAE-style encoder/decoder for collaborative space
		encoder_dims = hyper.get('encoder_dims', [600])
		decoder_dims = hyper.get('decoder_dims', [600])
		self.vae = VAEEncoderDecoder(
			input_dim=self.item_num,
			latent_dim=self.latent_dim,
			output_dim=self.item_num,
			encoder_dims=encoder_dims,
			decoder_dims=decoder_dims,
			dropout=self.dropout,
		)

		# LLM-based meta embeddings (same as Mult-VAE)
		self.usrprf_embeds = torch.tensor(
			configs['usrprf_embeds'], dtype=torch.float32, device=configs['device']
		)
		self.itmprf_embeds = torch.tensor(
			configs['itmprf_embeds'], dtype=torch.float32, device=configs['device']
		)

		# Meta-network to produce (mu_llm, logvar_llm) from language features
		llm_hidden = hyper.get('llm_hidden', 600)
		self.llm_mlp = nn.Sequential(
			nn.Linear(self.itmprf_embeds.shape[1], llm_hidden),
			nn.Tanh(),
			nn.Linear(llm_hidden, 2 * self.latent_dim),
		)

		self.drop = nn.Dropout(self.dropout)

		# Rectified flow vector field in latent space
		flow_hidden = hyper.get('flow_hidden', 512)
		time_dim = hyper.get('flow_time_dim', 64)
		self.flow_field = RectifiedFlowField(
			latent_dim=self.latent_dim,
			hidden_dim=flow_hidden,
			time_dim=time_dim,
		)
		self.flow_weight = hyper.get('flow_weight', 1.0)

		self.is_training = False

	def encode_collab(self, x: torch.Tensor):
		"""
		Encode collaborative interactions into latent Gaussian parameters.

		Args:
			x: [batch, item_num] multi-hot interaction vectors
		Returns:
			mu_src, logvar_src: [batch, latent_dim]
		"""
		mu_src, logvar_src = self.vae.encode(x)
		return mu_src, logvar_src

	def encode_llm(self, x: torch.Tensor, user_emb: torch.Tensor):
		"""
		Encode LLM-based semantic features into latent Gaussian parameters.

		Args:
			x: [batch, item_num] interaction vectors (used as weights)
			user_emb: [batch, d_text] user text embeddings
		Returns:
			mu_llm, logvar_llm: [batch, latent_dim]
		"""
		h = self.drop(x)
		hidden = torch.matmul(h, self.itmprf_embeds) + user_emb  # [B, d_text]
		hidden = self.llm_mlp[0:2](hidden)  # first Linear + Tanh
		out = self.llm_mlp[2](hidden)  # final Linear -> 2 * latent_dim
		mu_llm = out[:, : self.latent_dim]
		logvar_llm = out[:, self.latent_dim :]
		return mu_llm, logvar_llm

	def decode(self, z: torch.Tensor) -> torch.Tensor:
		"""
		Decode latent codes into interaction logits.
		"""
		return self.vae.decode(z)

	def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
		if self.is_training:
			return self.vae.reparameterize(mu, logvar, is_training=True)
		else:
			# in eval, use mean (consistent with existing baselines)
			return self.vae.reparameterize(mu, logvar, is_training=False)

	def _flow_matching_loss(self, mu_src: torch.Tensor, logvar_src: torch.Tensor):
		"""
		Compute rectified flow matching loss in latent space for collaborative distribution.

		We sample (z0, z1, t), construct z_t = (1-t) z0 + t z1 and train the flow
		field to match the constant velocity v_true = z1 - z0.
		"""
		device = mu_src.device

		# Sample z1 ~ q_src (we use reparameterization to include uncertainty)
		z1 = self.reparameterize(mu_src, logvar_src)  # [B, d]

		# Sample z0 ~ N(0, I)
		z0 = torch.randn_like(z1, device=device)

		# Sample t ~ Uniform(0, 1)
		B = z1.size(0)
		t = torch.rand(B, device=device)  # [B]

		# Straight-line interpolation
		z_t = (1.0 - t).unsqueeze(-1) * z0 + t.unsqueeze(-1) * z1  # [B, d]

		# True constant velocity along rectified path
		v_true = z1 - z0  # [B, d]

		# Predicted velocity from flow field
		v_pred = self.flow_field(z_t, t)  # [B, d]

		loss = F.mse_loss(v_pred, v_true)
		return loss

	def forward_for_loss(self, batch_users, data):
		"""
		Training forward pass.

		Returns:
			dict with keys compatible with strategy and builder:
			  - mu_src, mu_llm, logvar_src, logvar_llm
			  - recon_x, kld, bce
		"""
		self.is_training = True

		device = configs['device']

		# trainer 传入的是 Python list，这里统一转成 LongTensor 放到 device 上
		if isinstance(batch_users, torch.Tensor):
			batch_users = batch_users.long().to(device)
		else:
			batch_users = torch.LongTensor(batch_users).to(device)

		# data 已经是 dense tensor，直接搬到 device
		x = data.to(device)  # [B, item_num]

		user_emb = self.usrprf_embeds[batch_users]  # [B, d_text]

		# Encode collaborative and language spaces
		mu_src, logvar_src = self.encode_collab(x)
		mu_llm, logvar_llm = self.encode_llm(x, user_emb)

		# Combine for base VAE-style reconstruction
		mu = mu_src + mu_llm
		logvar = logvar_src + logvar_llm
		z = self.reparameterize(mu, logvar)
		recon_x = self.decode(z)

		# Base VAE KLD
		kld_vae = -0.5 * torch.mean(
			torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
		)

		# Rectified flow matching loss in latent space
		flow_loss = self._flow_matching_loss(mu_src, logvar_src)

		# Fold flow loss into kld so that existing builder treats it as regularization
		kld = kld_vae + self.flow_weight * flow_loss

		# Reconstruction loss (same as Mult-VAE)
		bce = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, dim=-1))

		return {
			"mu_src": mu_src,
			"mu_llm": mu_llm,
			"logvar_src": logvar_src,
			"logvar_llm": logvar_llm,
			"user_emb": user_emb,
			"recon_x": recon_x,
			"kld": kld,
			"bce": bce,
		}

	def forward_for_predict(self, pck_users, train_mask):
		"""
		Inference forward pass.

		We follow the same pattern as Mult-VAE:
		  - use mean of combined posterior as latent code
		  - optionally, flow field can be used in future for single-step sampling
		"""
		self.is_training = False
		device = configs["device"]

		# pck_users 可能是 numpy 数组或 tensor，这里统一为 LongTensor
		if isinstance(pck_users, torch.Tensor):
			batch_users = pck_users.long().to(device)
		else:
			batch_users = torch.LongTensor(pck_users).to(device)

		batch_data = self.data_handler.train_data[batch_users.cpu()]
		x = torch.FloatTensor(batch_data.toarray()).to(device)

		user_emb = self.usrprf_embeds[batch_users]

		mu_src, logvar_src = self.encode_collab(x)
		mu_llm, logvar_llm = self.encode_llm(x, user_emb)

		mu = mu_src + mu_llm
		logvar = logvar_src + logvar_llm
		z = self.reparameterize(mu, logvar)  # in eval mode this is just mu

		recon_x = self.decode(z)

		return {
			"data": x,
			"mu_src": mu_src,
			"mu_llm": mu_llm,
			"logvar_src": logvar_src,
			"logvar_llm": logvar_llm,
			"train_mask": train_mask,
			"recon_x": recon_x,
		}


