import torch
from torch import nn
import torch.nn.functional as F
from config.configurator import configs
from models.base_model import BaseModel
from models.layers.gaussian_diffusion import GaussianDiffusion, ModelMeanType
from models.layers.denoise_net import DenoiseNet


class LDiffRecBackbone(BaseModel):
	def __init__(self, data_handler):
		super().__init__(data_handler)
		self.data_handler = data_handler

		self.user_num = configs['data']['user_num']
		self.item_num = configs['data']['item_num']

		hyper = configs['model']
		if configs['data']['name'] in hyper:
			hyper = hyper[configs['data']['name']]
		self.dropout = hyper.get('dropout', 0.2)

		# VAE encoder-decoder structure (same as Mult-VAE)
		self.p_dims = [200, 600, self.item_num]
		self.q_dims = [self.item_num, 600, 200]
		temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2]
		self.q_layers = nn.ModuleList(
			[nn.Linear(d_in, d_out) for d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])]
		)

		self.usrprf_embeds = torch.tensor(configs['usrprf_embeds']).float().cuda()
		self.itmprf_embeds = torch.tensor(configs['itmprf_embeds']).float().cuda()

		self.mlp = nn.Sequential(
			nn.Linear(self.itmprf_embeds.shape[1], 600),
			nn.Tanh(),
			nn.Linear(600, 400)
		)

		self.p_layers = nn.ModuleList(
			[nn.Linear(d_in, d_out) for d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])]
		)

		self.drop = nn.Dropout(self.dropout)
		self.is_training = False

		# Diffusion model setup
		latent_dim = self.q_dims[-1]  # 200
		diffusion_steps = hyper.get('diffusion_steps', 1000)
		noise_schedule = hyper.get('noise_schedule', 'linear')
		noise_scale = hyper.get('noise_scale', 0.01)
		noise_min = hyper.get('noise_min', 0.0001)
		noise_max = hyper.get('noise_max', 0.02)
		
		self.denoise_net = DenoiseNet(latent_dim)
		self.diffusion = GaussianDiffusion(
			mean_type=ModelMeanType.EPSILON,
			noise_schedule=noise_schedule,
			noise_scale=noise_scale,
			noise_min=noise_min,
			noise_max=noise_max,
			steps=diffusion_steps,
			device=configs['device'],
			history_num_per_term=10,
			beta_fixed=True
		)
		self.use_diffusion = hyper.get('use_diffusion', True)  # Flag to enable/disable diffusion

	def encode(self, x, user_emb):
		h = self.drop(x)
		hidden = torch.matmul(h, self.itmprf_embeds) + user_emb
		hidden = self.mlp(hidden)
		mu_llm = hidden[:, :200]
		logvar_llm = hidden[:, 200:]

		for i, layer in enumerate(self.q_layers):
			h = layer(h)
			if i != len(self.q_layers) - 1:
				h = torch.tanh(h)
			else:
				mu = h[:, :self.q_dims[-1]]
				logvar = h[:, self.q_dims[-1]:]

		return mu, mu_llm, logvar, logvar_llm

	def decode(self, z):
		h = z
		for i, layer in enumerate(self.p_layers):
			h = layer(h)
			if i != len(self.p_layers) - 1:
				h = torch.tanh(h)
		return h

	def reparameterize(self, mu, logvar):
		if self.is_training:
			std = torch.exp(0.5 * logvar)
			eps = torch.randn_like(std)
			z = eps.mul(std) + mu
		else:
			z = mu
		return z

	def forward_for_loss(self, batch_users, data):
		self.is_training = True
		user_emb = self.usrprf_embeds[batch_users]
		mu_src, mu_llm, logvar_src, logvar_llm = self.encode(data, user_emb)

		# For L-DiffRec: diffusion process in latent space
		# Combine mu_src and mu_llm for diffusion
		mu = mu_src + mu_llm
		logvar = logvar_src + logvar_llm
		
		# Sample z from the combined distribution
		z_0 = self.reparameterize(mu, logvar)
		
		# Apply diffusion process if enabled
		diffusion_loss = None
		if self.use_diffusion:
			# Diffusion training: add noise and denoise
			diffusion_terms = self.diffusion.training_losses(self.denoise_net, z_0, reweight=False)
			# Use the predicted x_start from diffusion
			z_diffused = diffusion_terms["pred_xstart"]
			diffusion_loss = diffusion_terms.get("loss", torch.tensor(0.0).to(data.device))
		else:
			z_diffused = z_0

		return {
			'mu_src': mu_src, 'mu_llm': mu_llm,
			'logvar_src': logvar_src, 'logvar_llm': logvar_llm,
			'user_emb': user_emb,
			'z_diffused': z_diffused,  # Diffused latent for decoding
			'diffusion_loss': diffusion_loss
		}

	def forward_for_predict(self, pck_users, train_mask):
		self.is_training = False
		pck_users = pck_users.long()
		batch_data = self.data_handler.train_data[pck_users.cpu()]
		data = torch.FloatTensor(batch_data.toarray()).to(configs['device'])
		user_emb = self.usrprf_embeds[pck_users]
		mu_src, mu_llm, logvar_src, logvar_llm = self.encode(data, user_emb)

		# For prediction: use diffusion sampling if enabled
		mu = mu_src + mu_llm
		logvar = logvar_src + logvar_llm
		
		if self.use_diffusion:
			# Start from mean (no reparameterization in eval)
			z_start = mu
			# Sample from diffusion process
			z_sampled = self.diffusion.p_sample(self.denoise_net, z_start, steps=self.diffusion.steps, sampling_noise=False)
		else:
			z_sampled = mu

		return {
			'data': data,
			'mu_src': mu_src, 'mu_llm': mu_llm,
			'logvar_src': logvar_src, 'logvar_llm': logvar_llm,
			'train_mask': train_mask,
			'z_sampled': z_sampled  # Sampled latent for decoding
		}
