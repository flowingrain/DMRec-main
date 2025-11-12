import torch
from torch import nn
import torch.nn.functional as F
from config.configurator import configs
from models.base_model import BaseModel


class MultVAEBackbone(BaseModel):
	def __init__(self, data_handler):
		super().__init__(data_handler)
		self.data_handler = data_handler

		self.user_num = configs['data']['user_num']
		self.item_num = configs['data']['item_num']

		# 超参（含数据集特定覆盖）
		hyper = configs['model']
		if configs['data']['name'] in hyper:
			hyper = hyper[configs['data']['name']]
		self.dropout = hyper.get('dropout', 0.2)

		# 结构（与现有 mult_vae_* 一致）
		self.p_dims = [200, 600, self.item_num]
		self.q_dims = [self.item_num, 600, 200]
		temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2]
		self.q_layers = nn.ModuleList(
			[nn.Linear(d_in, d_out) for d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])]
		)

		# 语义嵌入
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
		return {
			'mu_src': mu_src, 'mu_llm': mu_llm,
			'logvar_src': logvar_src, 'logvar_llm': logvar_llm,
			'user_emb': user_emb
		}

	def forward_for_predict(self, pck_users, train_mask):
		self.is_training = False
		pck_users = pck_users.long()
		batch_data = self.data_handler.train_data[pck_users.cpu()]
		data = torch.FloatTensor(batch_data.toarray()).to(configs['device'])
		user_emb = self.usrprf_embeds[pck_users]
		mu_src, mu_llm, logvar_src, logvar_llm = self.encode(data, user_emb)
		return {
			'data': data,
			'mu_src': mu_src, 'mu_llm': mu_llm,
			'logvar_src': logvar_src, 'logvar_llm': logvar_llm,
			'train_mask': train_mask
		}

