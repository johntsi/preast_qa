import torch
from torch import nn
import torch.nn.functional as F


class HighwayNetwork(nn.Module):
	"""
	Highway network as implemented in Srivastava et al 2015
	It is used to fuse the embeddings of GloVe and ELMo
	before entering the Reader or Decoder modules
	"""
	def __init__(self, d_emb, dropout_rate, non_lin = "relu", num_layers = 2):
		super(HighwayNetwork, self).__init__()

		non_lin = non_lin.lower()
		assert non_lin in ["relu", "tanh"]

		self.HighwayLayers = nn.Sequential(*[HighwayLayer(d_emb, dropout_rate) for _ in range(num_layers)])
		
	def forward(self, x):
		"""
		Args
			x: (3d float tensor)
		Returns
			y: (3d float tensor), same dmensionality as x
		"""

		y = self.HighwayLayers(x)

		return y


class HighwayLayer(nn.Module):
	"""
	Highway layer module for use in the Highway network module
	"""
	def __init__(self, d_emb, dropout_rate, non_lin = "relu"):
		super(HighwayLayer, self).__init__()
		
		self.H = nn.Sequential(nn.Linear(d_emb, d_emb),
							nn.ReLU() if non_lin == "relu" else nn.Tanh())

		self.T = nn.Sequential(nn.Linear(d_emb, d_emb),
							nn.Sigmoid())

		self.dropout = nn.Dropout(dropout_rate)

	def forward(self, x):
		"""
		Args
			x: (3d float tensor)
		Returns
			y: (3d float tensor), same dmensionality as x
		"""

		T = self.T(x)
		y = self.dropout(torch.mul(self.H(x), T) + torch.mul(x, 1 - T))

		return y


class FeedForwardNetwork(nn.Module):
	"""
	2-layer feed-forward network module with d_inner != d_model
	and a GELu activation in between. It is used as the last sublayer of
	the decoder module
	"""
	def __init__(self, d_model, d_inner):
		super(FeedForwardNetwork, self).__init__()

		self.lin1 = nn.Linear(d_model, d_inner)
		self.lin2 = nn.Linear(d_inner, d_model)

	def forward(self, x):
		"""
		Args
			x: (3d float tensor)
		Returns
			y: (3d float tensor), same dmensionality as x
		"""

		y = self.lin2(F.gelu(self.lin1(x)))

		return y