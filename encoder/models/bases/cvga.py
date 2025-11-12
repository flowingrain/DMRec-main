import torch
from torch import nn
import torch.nn.functional as F
from config.configurator import configs
from models.base_model import BaseModel
from models.layers.gcn_layer import MultiLayerGCN, MultiLayerLightGCN


class CVGABackbone(BaseModel):
	"""
	Collaborative Variational Graph Auto-Encoder (CVGA)
	
	Key differences from traditional VAE:
	1. Uses GNN to encode user-item collaborative relationships on bipartite graph
	2. Does NOT learn user/item embeddings - focuses on user behavior distribution
	3. Uses variational inference to approximate posterior distribution
	4. Reconstructs the entire user-item interaction graph
	5. Training is interaction-agnostic with near-linear time complexity
	
	GNN Implementation:
	- Default: LightGCN (simplified GCN without activation and transformation)
	- Can fallback to standard GCN by setting use_lightgcn=False in config
	- LightGCN is more efficient and often performs better for recommendation
	
	LLM Alignment Support:
	- Added LLM embedding support to enable distribution matching with strategies (MDDM/CPDM/GODM)
	- mu_src/logvar_src: from GNN encoding (graph structure)
	- mu_llm/logvar_llm: from LLM embeddings (text representations)
	- Both are combined in strategy modules for distribution alignment
	"""
	def __init__(self, data_handler):
		super().__init__(data_handler)
		self.data_handler = data_handler

		self.user_num = configs['data']['user_num']
		self.item_num = configs['data']['item_num']
		self.total_nodes = self.user_num + self.item_num

		hyper = configs['model']
		if configs['data']['name'] in hyper:
			hyper = hyper[configs['data']['name']]
		self.dropout = hyper.get('dropout', 0.2)
		
		# GNN encoder dimensions
		# Input: one-hot or interaction features (no learnable embeddings!)
		# According to CVGA, we don't learn user/item embeddings
		input_dim = hyper.get('input_dim', 1)  # Simple one-hot or interaction indicator
		latent_dim = hyper.get('latent_dim', 200)  # Latent space dimension
		self.latent_dim = latent_dim  # Store for use in encode_llm
		
		# Choose GNN type: default is LightGCN
		use_lightgcn = hyper.get('use_lightgcn', True)
		gcn_n_layers = hyper.get('gcn_n_layers', 3)  # Number of LightGCN layers
		
		if use_lightgcn:
			# LightGCN Encoder: simplified GCN without activation and transformation
			# All layers use the same dimension (input_dim), final projection to latent_dim * 2
			self.gcn_encoder = MultiLayerLightGCN(
				dim=input_dim,
				n_layers=gcn_n_layers,
				dropout=self.dropout,
				final_proj_dim=latent_dim * 2  # Project to [mu, logvar] dimensions
			)
		else:
			# Standard GCN Encoder (backward compatibility)
			hidden_dims = hyper.get('gcn_hidden_dims', [64, 32])  # GCN hidden dimensions
			encoder_dims = [input_dim] + hidden_dims + [latent_dim * 2]
			self.gcn_encoder = MultiLayerGCN(
				dims=encoder_dims,
				dropout=self.dropout,
				activation='tanh'
			)
		
		# Decoder: reconstructs user-item interaction graph
		# Structure: [latent_dim] -> [hidden_dims] -> [item_num] (for each user)
		decoder_hidden_dims = hyper.get('decoder_hidden_dims', [32, 64])
		decoder_dims = [latent_dim] + decoder_hidden_dims + [self.item_num]
		self.decoder = nn.ModuleList([
			nn.Linear(d_in, d_out) 
			for d_in, d_out in zip(decoder_dims[:-1], decoder_dims[1:])
		])
		
		self.drop = nn.Dropout(self.dropout)
		self.is_training = False
		
		# LLM embeddings for alignment with strategies (MDDM/CPDM/GODM)
		# Load pre-trained LLM text representations
		self.usrprf_embeds = torch.tensor(configs['usrprf_embeds']).float().cuda()
		self.itmprf_embeds = torch.tensor(configs['itmprf_embeds']).float().cuda()
		
		# MLP to generate mu_llm and logvar_llm from LLM embeddings
		# Structure matches Mult-VAE: [itmprf_dim] -> [600] -> [400] -> split to [200, 200]
		self.llm_mlp = nn.Sequential(
			nn.Linear(self.itmprf_embeds.shape[1], 600),
			nn.Tanh(),
			nn.Linear(600, 400)
		)
		
		# Initialize node features (simple one-hot or interaction-based, NOT learnable embeddings)
		# CVGA uses the interaction graph itself as input features
		self._init_node_features(input_dim)
	
	def _init_node_features(self, input_dim):
		"""
		Initialize node features based on interaction graph.
		CVGA does NOT use learnable embeddings - features are derived from graph structure.
		"""
		if input_dim == 1:
			# Simple: use degree or constant features
			# In practice, CVGA may use interaction indicators
			self.node_features = torch.ones(self.total_nodes, input_dim).to(configs['device'])
		else:
			# Could use more sophisticated features (e.g., interaction patterns)
			self.node_features = torch.randn(self.total_nodes, input_dim).to(configs['device'])
	
	def encode(self, adj):
		"""
		Encode user-item collaborative relationships using GNN.
		
		Args:
			adj: [N, N] sparse adjacency matrix (N = user_num + item_num)
		
		Returns:
			mu: [N, latent_dim] mean of latent distribution
			logvar: [N, latent_dim] log variance of latent distribution
		"""
		# GNN encoding: propagate information through graph
		# This captures collaborative relationships via information aggregation
		encoded = self.gcn_encoder(self.node_features, adj)  # [N, latent_dim * 2]
		
		# Split into mu and logvar for variational inference
		mu = encoded[:, :encoded.shape[1] // 2]  # [N, latent_dim]
		logvar = encoded[:, encoded.shape[1] // 2:]  # [N, latent_dim]
		
		return mu, logvar
	
	def decode(self, z, user_indices):
		"""
		Decode latent codes to reconstruct user-item interactions.
		
		Args:
			z: [batch_size, latent_dim] latent codes (sampled for users in batch)
			user_indices: [batch_size] indices of users in the batch
		
		Returns:
			recon: [batch_size, item_num] reconstructed interaction probabilities
		"""
		h = z
		for i, layer in enumerate(self.decoder):
			h = layer(h)
			if i != len(self.decoder) - 1:
				h = torch.tanh(h)
				h = self.drop(h)
		return h
	
	def encode_llm(self, x, user_emb):
		"""
		Encode LLM embeddings to generate mu_llm and logvar_llm.
		This follows the same approach as Mult-VAE for consistency.
		
		Args:
			x: [batch_size, item_num] user-item interaction data
			user_emb: [batch_size, usrprf_dim] user LLM embeddings
		
		Returns:
			mu_llm: [batch_size, latent_dim] mean from LLM
			logvar_llm: [batch_size, latent_dim] logvar from LLM
		"""
		h = self.drop(x)
		# Aggregate item embeddings weighted by interactions, then add user embedding
		hidden = torch.matmul(h, self.itmprf_embeds) + user_emb
		hidden = self.llm_mlp(hidden)
		# Split into mu_llm and logvar_llm (each 200-dim for latent_dim=200)
		mu_llm = hidden[:, :self.latent_dim]
		logvar_llm = hidden[:, self.latent_dim:]
		return mu_llm, logvar_llm
	
	def reparameterize(self, mu, logvar, user_indices):
		"""
		Reparameterization trick: sample from latent distribution.
		Only sample for users in the batch (not all nodes).
		
		Args:
			mu: [N, latent_dim] mean (all nodes)
			logvar: [N, latent_dim] logvar (all nodes)
			user_indices: [batch_size] user indices in batch
		
		Returns:
			z: [batch_size, latent_dim] sampled latent codes
		"""
		# Extract mu and logvar for users in batch
		mu_batch = mu[user_indices]  # [batch_size, latent_dim]
		logvar_batch = logvar[user_indices]  # [batch_size, latent_dim]
		
		if self.is_training:
			std = torch.exp(0.5 * logvar_batch)
			eps = torch.randn_like(std)
			z = eps.mul(std) + mu_batch
		else:
			z = mu_batch
		return z
	
	def forward_for_loss(self, batch_users, data):
		"""
		Forward pass for training.
		
		Args:
			batch_users: [batch_size] user indices
			data: [batch_size, item_num] user-item interaction data
		
		Returns:
			dict with intermediate values for loss computation
		"""
		self.is_training = True
		
		# Get adjacency matrix (user-item bipartite graph)
		adj = self.data_handler.torch_adj  # [N, N] sparse tensor
		
		# Encode collaborative relationships using GNN
		mu, logvar = self.encode(adj)  # [N, latent_dim] each
		
		# For CVGA, we focus on user behavior distribution
		# Extract user nodes (first user_num nodes in the graph)
		user_mu = mu[:self.user_num]  # [user_num, latent_dim]
		user_logvar = logvar[:self.user_num]  # [user_num, latent_dim]
		
		# Sample latent codes for users in batch
		z = self.reparameterize(mu, logvar, batch_users)  # [batch_size, latent_dim]
		
		# Decode to reconstruct interactions
		recon_x = self.decode(z, batch_users)  # [batch_size, item_num]
		
		# Generate mu_llm and logvar_llm from LLM embeddings for strategy alignment
		user_emb = self.usrprf_embeds[batch_users]  # [batch_size, usrprf_dim]
		mu_llm, logvar_llm = self.encode_llm(data, user_emb)  # [batch_size, latent_dim] each
		
		return {
			'mu_src': user_mu[batch_users],  # [batch_size, latent_dim] from GNN encoding
			'mu_llm': mu_llm,  # [batch_size, latent_dim] from LLM embeddings
			'logvar_src': user_logvar[batch_users],  # [batch_size, latent_dim] from GNN encoding
			'logvar_llm': logvar_llm,  # [batch_size, latent_dim] from LLM embeddings
			'z': z,  # Sampled latent codes
			'recon_x': recon_x,  # Reconstructed interactions
			'user_indices': batch_users
		}
	
	def forward_for_predict(self, pck_users, train_mask):
		"""
		Forward pass for prediction/evaluation.
		
		Args:
			pck_users: [batch_size] user indices
			train_mask: [batch_size, item_num] mask for training items
		
		Returns:
			dict with intermediate values for prediction
		"""
		self.is_training = False
		pck_users = pck_users.long()
		
		# Get adjacency matrix
		adj = self.data_handler.torch_adj
		
		# Encode using GNN
		mu, logvar = self.encode(adj)
		
		# Extract user nodes
		user_mu = mu[:self.user_num]
		user_logvar = logvar[:self.user_num]
		
		# Generate mu_llm and logvar_llm from LLM embeddings for strategy alignment
		# Need to get user interaction data for LLM encoding
		batch_data = self.data_handler.train_data[pck_users.cpu()]
		data = torch.FloatTensor(batch_data.toarray()).to(configs['device'])
		user_emb = self.usrprf_embeds[pck_users]  # [batch_size, usrprf_dim]
		mu_llm, logvar_llm = self.encode_llm(data, user_emb)  # [batch_size, latent_dim] each
		
		return {
			'data': None,  # Not needed for CVGA
			'mu_src': user_mu[pck_users],
			'mu_llm': mu_llm,
			'logvar_src': user_logvar[pck_users],
			'logvar_llm': logvar_llm,
			'train_mask': train_mask,
			'z_cvga': user_mu[pck_users]  # Latent codes for decoding
		}
