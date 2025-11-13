import torch
from torch import nn
import torch.nn.functional as F
from config.configurator import configs
from models.base_model import BaseModel
from models.layers.gcn_layer import MultiLayerGCN, MultiLayerLightGCN


class CVGABackbone(BaseModel):
	"""
	Collaborative Variational Graph Auto-Encoder (CVGA)
	
	According to paper "Revisiting Graph-based Recommender Systems from the Perspective of Variational Auto-Encoder":
	
	Key differences from traditional VAE:
	1. Uses single-layer GCN to encode user-item collaborative relationships on bipartite graph
	2. Does NOT learn user/item embeddings - focuses on user behavior distribution
	3. Uses variational inference to approximate posterior distribution
	4. Reconstructs the entire user-item interaction graph
	5. Training is interaction-agnostic with near-linear time complexity
	
	GNN Implementation:
	- Uses single-layer GCN (as per paper) to learn distribution parameters (μ, σ²)
	- User nodes use their interaction vectors x_u (multi-hot) as input features
	- Item nodes use constant features (one-hot or constant)
	- GCN aggregates one-hop neighbors to learn μ and log(σ²) for each user
	- Default: LightGCN (simplified GCN without activation and transformation)
	- Can fallback to standard GCN by setting use_lightgcn=False in config
	
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
		# Input: user interaction vectors x_u (multi-hot, dimension = item_num)
		# According to CVGA paper, users use their interaction vectors as input features
		# For CVGA, input_dim should match item_num to use interaction vectors directly
		input_dim = hyper.get('input_dim', self.item_num)  # Default to item_num to use interaction vectors
		latent_dim = hyper.get('latent_dim', 200)  # Latent space dimension
		self.latent_dim = latent_dim  # Store for use in encode_llm
		self.input_dim = input_dim  # Store for node feature updates
		
		# According to CVGA paper: use single-layer GCN
		# Paper states that multi-layer GCN causes dimension mismatch issues
		use_lightgcn = hyper.get('use_lightgcn', True)
		gcn_n_layers = hyper.get('gcn_n_layers', 1)  # CVGA uses single-layer GCN by default
		
		if use_lightgcn:
			# LightGCN Encoder: simplified GCN without activation and transformation
			# CVGA paper uses single-layer GCN to learn distribution parameters
			# All layers use the same dimension (input_dim), final projection to latent_dim * 2
			self.gcn_encoder = MultiLayerLightGCN(
				dim=input_dim,
				n_layers=gcn_n_layers,
				dropout=self.dropout,
				final_proj_dim=latent_dim * 2  # Project to [mu, logvar] dimensions
			)
		else:
			# Standard GCN Encoder: single-layer as per CVGA paper
			# Structure: [input_dim] -> [latent_dim * 2] (single layer)
			encoder_dims = [input_dim, latent_dim * 2]
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
		
		# Initialize node features placeholder
		# CVGA uses the interaction graph itself as input features
		# Note: We'll update this with actual interaction data in forward_for_loss
		self._init_node_features(input_dim)
		self.use_interaction_features = hyper.get('use_interaction_features', True)  # Use actual x_u as features
		
		# Pre-initialize projection layers to avoid state_dict loading issues
		# Always create them as identity layers if not needed, to ensure consistent state_dict
		# This prevents "Unexpected key" errors when loading saved models
		if input_dim != self.item_num:
			# Need projection from item_num to input_dim
			self._user_feature_proj = nn.Linear(self.item_num, input_dim, bias=False).to(configs['device'])
			if input_dim < self.item_num:
				# If input_dim < item_num, items also need projection
				self._item_feature_proj = nn.Linear(self.item_num, input_dim, bias=False).to(configs['device'])
			else:
				# If input_dim > item_num, items can use padding or projection
				self._item_feature_proj = None
		else:
			# No projection needed, but create identity-like layers for state_dict consistency
			# Use a dummy layer that will be bypassed in forward pass
			self._user_feature_proj = nn.Identity()  # Always present in state_dict
			self._item_feature_proj = None
	
	def _init_node_features(self, input_dim):
		"""
		Initialize node features placeholder.
		Actual features will be updated from interaction data in forward pass.
		"""
		# Initialize placeholder (will be updated with actual interaction data)
		self.node_features = torch.ones(self.total_nodes, input_dim).to(configs['device'])
	
	def _update_node_features_from_data(self):
		"""
		Update node features using actual interaction data.
		According to CVGA paper, user nodes should use their interaction vectors x_u.
		This should be called when training data is available.
		"""
		# Get training data matrix
		train_data = self.data_handler.train_data  # [user_num, item_num] sparse matrix
		
		# For users: use their interaction vectors (multi-hot) x_u
		# According to CVGA paper, x_u is a multi-hot vector of dimension item_num
		user_features = torch.FloatTensor(train_data.toarray()).to(configs['device'])  # [user_num, item_num]
		
		# For items: use one-hot identity matrix (each item is represented by its own one-hot vector)
		# This matches the CVGA paper's approach where items use simple features
		if self.input_dim == self.item_num:
			# If input_dim matches item_num, use identity matrix for items
			item_features = torch.eye(self.item_num).to(configs['device'])  # [item_num, item_num]
		else:
			# If input_dim != item_num, use projection (already initialized in __init__)
			if self._item_feature_proj is not None:
				item_onehot = torch.eye(self.item_num).to(configs['device'])
				item_features = self._item_feature_proj(item_onehot)
			else:
				# Fallback: use constant features
				item_features = torch.ones(self.item_num, self.input_dim).to(configs['device'])
		
		# Project user features if needed
		# _user_feature_proj is always present (either Linear or Identity)
		if isinstance(self._user_feature_proj, nn.Linear):
			user_features = self._user_feature_proj(user_features)
		# If Identity, no projection needed (already correct dimension)
		
		# Concatenate: [user_features; item_features]
		self.node_features = torch.cat([user_features, item_features], dim=0)  # [total_nodes, input_dim]
	
	def encode(self, adj, node_features=None):
		"""
		Encode user-item collaborative relationships using GNN.
		According to CVGA paper: single-layer GCN aggregates neighbors to learn distribution parameters.
		
		Args:
			adj: [N, N] sparse adjacency matrix (N = user_num + item_num)
			node_features: [N, input_dim] node features (if None, use self.node_features)
		
		Returns:
			mu: [N, latent_dim] mean of latent distribution
			logvar: [N, latent_dim] log variance of latent distribution
		"""
		if node_features is None:
			node_features = self.node_features
		
		# GNN encoding: propagate information through graph
		# Single-layer GCN: aggregates one-hop neighbors to learn distribution parameters
		# This captures collaborative relationships via information aggregation
		encoded = self.gcn_encoder(node_features, adj)  # [N, latent_dim * 2]
		
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
		According to CVGA paper: uses user interaction vectors x_u as node features.
		
		Args:
			batch_users: [batch_size] user indices
			data: [batch_size, item_num] user-item interaction data (x_u for users in batch)
		
		Returns:
			dict with intermediate values for loss computation
		"""
		self.is_training = True
		
		# Get adjacency matrix (user-item bipartite graph)
		adj = self.data_handler.torch_adj  # [N, N] sparse tensor
		
		# According to CVGA paper: update node features with actual interaction data
		# Users use their interaction vectors x_u, items use constant features
		if self.use_interaction_features:
			self._update_node_features_from_data()
		
		# Encode collaborative relationships using GNN
		# Single-layer GCN aggregates neighbors to learn distribution parameters
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
		
		# Compute model-specific losses for strategy modules
		# CVGA: use mu_src and logvar_src directly (no mu_llm addition)
		mu_src_batch = user_mu[batch_users]  # [batch_size, latent_dim]
		logvar_src_batch = user_logvar[batch_users]  # [batch_size, latent_dim]
		kld = - 0.5 * torch.mean(torch.sum(1 + logvar_src_batch - mu_src_batch.pow(2) - logvar_src_batch.exp(), dim=1))
		bce = - torch.mean(torch.sum(F.log_softmax(recon_x, 1) * data, -1))
		
		return {
			'mu_src': mu_src_batch,  # [batch_size, latent_dim] from GNN encoding
			'mu_llm': mu_llm,  # [batch_size, latent_dim] from LLM embeddings
			'logvar_src': logvar_src_batch,  # [batch_size, latent_dim] from GNN encoding
			'logvar_llm': logvar_llm,  # [batch_size, latent_dim] from LLM embeddings
			'z': z,  # Sampled latent codes
			'recon_x': recon_x,  # Reconstructed interactions
			'user_indices': batch_users,
			'kld': kld,  # Model-specific KLD
			'bce': bce  # Model-specific reconstruction loss
		}
	
	def forward_for_predict(self, pck_users, train_mask):
		"""
		Forward pass for prediction/evaluation.
		According to CVGA paper: uses user interaction vectors x_u as node features.
		
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
		
		# According to CVGA paper: update node features with actual interaction data
		if self.use_interaction_features:
			self._update_node_features_from_data()
		
		# Encode using GNN (single-layer GCN)
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
