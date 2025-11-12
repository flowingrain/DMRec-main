import torch
from torch import nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
	"""
	Graph Convolutional Network layer for bipartite user-item graph.
	Implements information propagation and aggregation as described in CVGA.
	"""
	def __init__(self, in_dim, out_dim, dropout=0.0, activation='tanh'):
		super(GCNLayer, self).__init__()
		self.in_dim = in_dim
		self.out_dim = out_dim
		self.dropout = nn.Dropout(dropout)
		
		# Linear transformation
		self.weight = nn.Linear(in_dim, out_dim, bias=False)
		
		# Activation function
		if activation == 'tanh':
			self.activation = nn.Tanh()
		elif activation == 'relu':
			self.activation = nn.ReLU()
		elif activation == 'sigmoid':
			self.activation = nn.Sigmoid()
		else:
			self.activation = None
	
	def forward(self, features, adj):
		"""
		Args:
			features: [N, in_dim] node features (N = user_num + item_num)
			adj: [N, N] sparse adjacency matrix (normalized)
		
		Returns:
			output: [N, out_dim] aggregated node features
		"""
		# Apply dropout
		features = self.dropout(features)
		
		# Graph convolution: A * X * W
		# adj is already normalized (symmetric normalization)
		support = self.weight(features)  # [N, out_dim]
		output = torch.sparse.mm(adj, support)  # [N, out_dim]
		
		# Apply activation if specified
		if self.activation is not None:
			output = self.activation(output)
		
		return output


class MultiLayerGCN(nn.Module):
	"""
	Multi-layer GCN for encoding user-item collaborative relationships.
	"""
	def __init__(self, dims, dropout=0.0, activation='tanh'):
		"""
		Args:
			dims: list of dimensions, e.g., [input_dim, hidden_dim1, hidden_dim2, ...]
			dropout: dropout rate
			activation: activation function
		"""
		super(MultiLayerGCN, self).__init__()
		self.layers = nn.ModuleList()
		for i in range(len(dims) - 1):
			self.layers.append(
				GCNLayer(dims[i], dims[i+1], dropout=dropout, activation=activation)
			)
	
	def forward(self, features, adj):
		"""
		Args:
			features: [N, input_dim] initial node features
			adj: [N, N] sparse adjacency matrix
		
		Returns:
			output: [N, output_dim] final node features after multi-layer propagation
		"""
		h = features
		for layer in self.layers:
			h = layer(h, adj)
		return h


class LightGCNLayer(nn.Module):
	"""
	LightGCN layer: simplified GCN without activation and feature transformation.
	Only performs neighbor aggregation: e^(k+1) = A * e^(k)
	"""
	def __init__(self, dropout=0.0):
		super(LightGCNLayer, self).__init__()
		self.dropout = nn.Dropout(dropout)
	
	def forward(self, features, adj):
		"""
		Args:
			features: [N, dim] node features
			adj: [N, N] sparse adjacency matrix (normalized)
		
		Returns:
			output: [N, dim] aggregated node features (same dimension)
		"""
		# Apply dropout
		features = self.dropout(features)
		
		# LightGCN: only neighbor aggregation, no transformation, no activation
		output = torch.sparse.mm(adj, features)
		
		return output


class MultiLayerLightGCN(nn.Module):
	"""
	Multi-layer LightGCN for encoding user-item collaborative relationships.
	
	LightGCN characteristics:
	1. No nonlinear activation
	2. No feature transformation (weight matrix)
	3. Final output is weighted average of all layer embeddings
	"""
	def __init__(self, dim, n_layers=3, dropout=0.0, final_proj_dim=None):
		"""
		Args:
			dim: feature dimension (input and output are the same)
			n_layers: number of LightGCN layers
			dropout: dropout rate
			final_proj_dim: if specified, add a final linear projection to this dimension
		"""
		super(MultiLayerLightGCN, self).__init__()
		self.dim = dim
		self.n_layers = n_layers
		
		# LightGCN layers (no transformation, just propagation)
		self.layers = nn.ModuleList()
		for _ in range(n_layers):
			self.layers.append(LightGCNLayer(dropout=dropout))
		
		# Optional final projection (e.g., to map to mu/logvar dimensions)
		if final_proj_dim is not None:
			self.final_proj = nn.Linear(dim, final_proj_dim, bias=False)
		else:
			self.final_proj = None
	
	def forward(self, features, adj):
		"""
		Args:
			features: [N, dim] initial node features
			adj: [N, N] sparse adjacency matrix
		
		Returns:
			output: [N, output_dim] final node features
				- If final_proj_dim is None: [N, dim] (average of all layers)
				- If final_proj_dim is set: [N, final_proj_dim]
		"""
		# Store embeddings from all layers
		embeddings = [features]  # e^(0)
		
		# Propagate through layers
		h = features
		for layer in self.layers:
			h = layer(h, adj)
			embeddings.append(h)  # e^(1), e^(2), ..., e^(K)
		
		# LightGCN: final embedding is average of all layers
		# e = (e^(0) + e^(1) + ... + e^(K)) / (K+1)
		final_emb = torch.stack(embeddings, dim=0).mean(dim=0)  # [N, dim]
		
		# Optional final projection
		if self.final_proj is not None:
			final_emb = self.final_proj(final_emb)
		
		return final_emb

