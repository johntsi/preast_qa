import torch
from torch import nn
import math


class Embedder(nn.Module):
	"""
	Projects the indices of a sequnce using a linear layer and adds positional embeddings
	"""

	def __init__(self, glove_vectors, pad_idx, dropout_rate, max_seq_len = 100):
		"""
		glove vectors: the pre-trained glove embeddings of all the tokens in the fixed vocabulary
			(2d float tensor) [d_fixed x d_emb]
		pad_idx: int
		dropout_rate: float
		max_seq_len: int
		"""
		
		super(Embedder, self).__init__()

		d_emb = glove_vectors.size(1)

		# initialize embeddings matrix
		self.embedding_layer = nn.Embedding.from_pretrained(
			glove_vectors, padding_idx=pad_idx, freeze=False
		)

		# initialize positional embeddings matrix
		positional_encoder = torch.zeros(max_seq_len, d_emb)
		position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_emb, 2).float() * (-math.log(10000.0) / d_emb))
		positional_encoder[:, 0::2] = torch.sin(position * div_term)
		positional_encoder[:, 1::2] = torch.cos(position * div_term)
		positional_encoder = positional_encoder.unsqueeze(0)
		self.register_buffer("positional_encoder", positional_encoder)

		# initialie dropout layer
		self.dropout = nn.Dropout(dropout_rate)

	def forward(self, x):
		"""
		Args:
			x The indices of a sequence in the fixed vocabulary
				(2d long tensor) [batch_size x sequence_length]
		Returns:
			Embeddings for sequence x
				(3d float tensor) [batch_size x sequence_length x d_emb]
		"""

		seq_len = x.size(1)
		emb = self.embedding_layer(x)

		return self.dropout(emb + self.positional_encoder[0, :seq_len, :])