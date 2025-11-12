import torch
from torch import nn
import torch.nn.functional as F
from config.configurator import configs


class VAEEncoder(nn.Module):
	"""VAE Encoder: maps input to latent distribution parameters"""
	def __init__(self, input_dim, latent_dim, hidden_dims, dropout=0.2):
		super().__init__()
		self.input_dim = input_dim
		self.latent_dim = latent_dim
		self.hidden_dims = hidden_dims
		
		# Build encoder layers: [input_dim, ...hidden_dims, latent_dim*2]
		# Output is split into mu and logvar
		dims = [input_dim] + hidden_dims + [latent_dim * 2]
		self.layers = nn.ModuleList(
			[nn.Linear(d_in, d_out) for d_in, d_out in zip(dims[:-1], dims[1:])]
		)
		self.drop = nn.Dropout(dropout)
	
	def forward(self, x):
		h = self.drop(x)
		for i, layer in enumerate(self.layers):
			h = layer(h)
			if i != len(self.layers) - 1:
				h = torch.tanh(h)
			else:
				# Split into mu and logvar
				mu = h[:, :self.latent_dim]
				logvar = h[:, self.latent_dim:]
		return mu, logvar


class VAEDecoder(nn.Module):
	"""VAE Decoder: maps latent code to reconstruction"""
	def __init__(self, latent_dim, output_dim, hidden_dims):
		super().__init__()
		self.latent_dim = latent_dim
		self.output_dim = output_dim
		self.hidden_dims = hidden_dims
		
		# Build decoder layers: [latent_dim, ...hidden_dims, output_dim]
		dims = [latent_dim] + hidden_dims + [output_dim]
		self.layers = nn.ModuleList(
			[nn.Linear(d_in, d_out) for d_in, d_out in zip(dims[:-1], dims[1:])]
		)
	
	def forward(self, z):
		h = z
		for i, layer in enumerate(self.layers):
			h = layer(h)
			if i != len(self.layers) - 1:
				h = torch.tanh(h)
		return h


class VAEEncoderDecoder(nn.Module):
	"""Complete VAE Autoencoder: Encoder + Decoder"""
	def __init__(self, input_dim, latent_dim, output_dim, encoder_dims, decoder_dims, dropout=0.2):
		super().__init__()
		self.encoder = VAEEncoder(input_dim, latent_dim, encoder_dims, dropout)
		self.decoder = VAEDecoder(latent_dim, output_dim, decoder_dims)
	
	def encode(self, x):
		return self.encoder(x)
	
	def decode(self, z):
		return self.decoder(z)
	
	def reparameterize(self, mu, logvar, is_training=True):
		"""Reparameterization trick"""
		if is_training:
			std = torch.exp(0.5 * logvar)
			eps = torch.randn_like(std)
			z = eps.mul(std) + mu
		else:
			z = mu
		return z

