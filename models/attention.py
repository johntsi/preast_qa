import torch
from torch import nn
import torch.nn.functional as F
from math import sqrt


class MultiHeadAttention(nn.Module):
	"""
	Multi Head Attention module as described in "Attention is all you need"
	After applying a linear transformation splits keys, values and queries in n heads
	Then calculates the scaled similarity scores between queries and keys and
	normallizes them along the dimension corresponding to the keys
	Finally transforms the values according to the normallized scores and applies
	another linear transformation

	"""

	def __init__(self, heads, d_model, dropout_rate):
		super(MultiHeadAttention, self).__init__()

		self.d_model = d_model
		self.d_k = d_model // heads
		self.h = heads

		self.q_linear = nn.Linear(d_model, d_model, bias=False)
		self.v_linear = nn.Linear(d_model, d_model, bias=False)
		self.k_linear = nn.Linear(d_model, d_model, bias=False)

		self.dropout = nn.Dropout(dropout_rate)
		self.out_linear = nn.Linear(d_model, d_model, bias=False)

		self.temperature = sqrt(self.d_k)

	def forward(self, k, v, q, mask=None):
		"""
		k == v always
		k: (3d, float tensor), [bs x seq_len_k x d_model]
		v: (3d, float tensor), [bs x seq_len_v x d_model]
		q: (3d, float tensor), [bs x seq_len_q x d_model]
		the mask corresponds to k and is (3d) [bs x seq_len_k x 1] or [bs x seq_len_k x seq_len_k] (decoder)
		if k is passages(encoder) dim(0) = bs x K
		if k is passages(decoder) dim(1) = K x seq_len_passages
		if k is the concatinated passages and questions then dim(0) = bs x (K + 1)

		Args:
			k, v, q: (3d float tensors)
			mask: (3d long tensor) corresponding to k

		Returns: (3d float tensor), same shape as k and v

		"""

		# get batch size and sequence lengths
		bs, seq_len_k, _ = k.size()
		seq_len_q = q.size(1)

		# perform linear operation
		k = self.k_linear(k)
		v = self.v_linear(v)
		q = self.q_linear(q)

		# split into h heads
		k = k.view(bs, seq_len_k, self.h, self.d_k).transpose(
			1, 2
		)  # [bs x heads x seq_len_k x d_k]
		v = v.view(bs, seq_len_k, self.h, self.d_k).transpose(
			1, 2
		)  # [bs x heads x seq_len_k x d_k]
		q = q.view(bs, seq_len_q, self.h, self.d_k).transpose(
			1, 2
		)  # [bs x heads x seq_len_q x d_k]

		# calculate the scaled similarity scores between queries and keys
		scores = (
			torch.matmul(q, k.transpose(-2, -1)) / self.temperature
		)  # (4d) [bs x heads x seq_len_q x seq_len_k]

		# apply the key mask
		if mask is not None:
			mask = mask.unsqueeze(1).transpose(-2, -1)  # (4d) [bs x 1 x 1 x seq_len_k]
			scores = scores.masked_fill(
				mask == 0, -1e9
			)  # (4d) [bs x heads x seq_len_q x seq_len_k]

		# normallize scores along the seq_len_k dimension and apply dropout
		scores = self.dropout(F.softmax(scores, dim=-1))

		# transform the values by multiplying them with the normallized scores
		attn = torch.matmul(scores, v)  # (4d) [bs x heads x seq_len_q x d_k]

		# concatinate the heads along the last dimension
		attn_concat = attn.transpose(1, 2).reshape(bs, seq_len_q, self.d_model)

		# apply final linear transformation
		out = self.out_linear(attn_concat)  # (3d): [bs x seq_len_q x d_model]

		return out


class DualAttention(nn.Module):
	"""
	Dual Attention Module as implemented in Xiong et al 2017
	First calculates a similarity matrix between Ex and Ey
	Then gets A and B by normallizing along the last dimensions
	And finally obtains dual representations Gx and Gy by a series
		of matrix multiplications and concatinations
	"""

	def __init__(self, d_model, dropout_rate=0):
		super(DualAttention, self).__init__()

		self.similarity_linear = nn.Linear(3 * d_model, 1, bias=False)
		self.dropout = nn.Dropout(dropout_rate)
		self.d_model = d_model

	def forward(self, Ex, Ey, mask_x, mask_y):
		"""
		Given 2 sequence representations X and Y and their masks
		produces bidirectionally informed representations
		Gx (Y-->X) and Gy (X-->Y)

		The X-sequence can be 4d, where some extra steps are applied

		Args
			Ex: 4d float tensor [bs x K x len_x x d_model] or
				3d float tensor [bs x len_x x d_model]
			Ey: 3d float tensor [bs x len_y x d_model]
			mask_x: 4d bool tensor [bs x K x len_x x 1] or
					3d bool tensor [bs x len_x x 1]
			mask_y: 3d bool tensor [bs x len_y x 1]
		Returns
			Gx: 4d float tensor [bs x K x len_x x 5 * d_model] or
				3d float tensor [bs x len_x x 5 * d_model]
			Gy: 3d float tensor [bs x len_y x 5 * d_model]
		"""

		if len(Ex.size()) == 3:

			bs, len_x, _ = Ex.size()
			len_y = Ey.size(1)

			Ex = Ex.view(bs, len_x, 1, self.d_model).expand(bs, len_x, len_y, self.d_model)
			Ey = Ey.view(bs, 1, len_y, self.d_model).expand(bs, len_x, len_y, self.d_model)

			mask_x = mask_x.view(bs, len_x, 1)
			mask_y = mask_y.view(bs, 1, len_y)

			# 4d float tensor [bs x len_x x len_y x 3 * d_model]
			E_cat = torch.cat([Ex, Ey, torch.mul(Ex, Ey)], dim = -1)

			# 3d float tensor [bs x len_x x len_y]
			U = self.similarity_linear(E_cat).squeeze(-1)

			U = U.masked_fill(mask_x * mask_y == 0, -1e9)
			A = self.dropout(F.softmax(U, dim = 2))
			B = self.dropout(F.softmax(U, dim = 1).transpose(-2, -1))

			# reduce the extra dimension
			Ex = torch.narrow(Ex, 2, 0, 1).squeeze(2)
			Ey = torch.narrow(Ey, 1, 0, 1).squeeze(1)

			A_1bar = torch.matmul(A, Ey)  # [bs x len_x x d_model]
			B_1bar = torch.matmul(B, Ex)   # [bs x len_y x d_model]
			A_2bar = torch.matmul(A, B_1bar)  # [bs x len_x x d_model]
			B_2bar = torch.matmul(B, A_1bar)   # [bs x len_y x d_model]

			Gx = torch.cat([Ex, A_1bar, A_2bar, torch.mul(Ex, A_1bar), torch.mul(Ex, A_2bar)], dim = -1)
			Gy = torch.cat([Ey, B_1bar, B_2bar, torch.mul(Ey, B_1bar), torch.mul(Ey, B_2bar)], dim = -1)

		else:

			bs, K, len_x, _ = Ex.size()
			len_y = Ey.size(1)

			# create an extra dimension in Ey
			Ey = Ey.unsqueeze(1).expand(bs, K, len_y, self.d_model)  # (4d): [bs x K x len_y x d_model]

			# concatinate Ex, Ey and their element-wise multiplication along the last dimension
			# (5d): [bs x K x len_x x len_y x 3 * d_model]
			E_cat = torch.cat(
				[
					Ex.unsqueeze(3).expand(bs, K, len_x, len_y, self.d_model),
					Ey.unsqueeze(2).expand(bs, K, len_x, len_y, self.d_model),
					torch.mul(
						Ex.unsqueeze(3).expand(bs, K, len_x, len_y, self.d_model),
						Ey.unsqueeze(2).expand(bs, K, len_x, len_y, self.d_model),
					),
				],
				dim=-1,
			)

			# similarity between embeddings of the X and Y sequences
			# reduces the last dimension by multiplying E_cat with a vector
			U = self.similarity_linear(E_cat).squeeze(-1)  # (4d): [bs x K x len_x x len_y]

			# apply the two masks
			U = U.masked_fill(mask_y.unsqueeze(1).transpose(-1, -2) == 0, -1e9).masked_fill(
				mask_x == 0, -1e9
			)

			# normallize along the len_y to get weights A and along the len_x dimension to get weights B
			A = self.dropout(F.softmax(U, dim=-1))  # (4d): [bs x K x len_x x len_y]
			B = self.dropout(F.softmax(U, dim=-2)).transpose(-2, -1)  # (4d): [bs x K x len_y x len_x]

			# get updated representations
			A_1bar = torch.matmul(A, Ey)  # (4d) [bs x K x len_x x d_model]
			B_1bar = torch.matmul(B, Ex)  # (4d): [bs x K x len_y x d_model]
			A_2bar = torch.matmul(A, B_1bar)  # (4d): [bs x K x len_x x d_model]
			B_2bar = torch.matmul(B, A_1bar)  # (4d): [bs x K x len_y x d_model]

			# reduce the extra dimension in the y-sequence
			Ey = torch.narrow(Ey, 1, 0, 1).squeeze(1)  # [bs x len_y x d_model]

			# Get a unique representation for question by taking the max along the K dimension
			B_1bar_m, _ = torch.max(B_1bar, dim=1)  # [bs x len_y x d_model]
			B_2bar_m, _ = torch.max(B_2bar, dim=1)  # [bs x len_y x d_model]

			# DCN for dual attention representations
			# Gp (4d): [bs x K x len_x x 5 * d_model]
			# Gq (3d): [bs x len_y x 5 * d_model]
			Gx = torch.cat([Ex, A_1bar, A_2bar, torch.mul(Ex, A_1bar), torch.mul(Ex, A_2bar)], dim=-1)
			Gy = torch.cat([Ey, B_1bar_m, B_2bar_m, torch.mul(Ey, B_1bar_m), torch.mul(Ey, B_2bar_m)], dim=-1)

		return Gx, Gy


class TripleAttention(nn.Module):
	"""
	Extension of the DualAttention but for three sequences
	Obtains globally updated representations for the three sequences
	"""

	def __init__(self, d_model, dropout_rate = 0):
		super(TripleAttention, self).__init__()
	
		self.pq_similarity_linear = nn.Linear(3 * d_model, 1, bias = False)
		self.pa_similarity_linear = nn.Linear(3 * d_model, 1, bias = False)
		self.qa_similarity_linear = nn.Linear(3 * d_model, 1, bias = False)
		self.dropout = nn.Dropout(dropout_rate)
		self.d_model = d_model

	def forward(self, Ep, Eq, Ea, mask_p, mask_q, mask_a):
		"""
		Args
			Ep: 3d float tensor [bs x L x d_model]
			Eq: 3d float tensor [bs x J x d_model]
			Ea: 3d float tensor [bs x N x d_model]
			mask_p: 3d bool tensor [bs x L x 1]
			mask_q: 3d bool tensor [bs x J x 1]
			mask_a: 3d bool tensor [bs x J x 1]
		Returns
			Gp:	3d float tensor [bs x L x 9 * d_model]
			Gy: 3d float tensor [bs x J x 9 * d_model]
			Ga: 3d float tensor [bs x N x 9 * d_model]
		"""

		bs = len(Ep)
		L, J, N = Ep.size(1), Eq.size(1), Ea.size(1)

		Epq = Ep.view(bs, L, 1, self.d_model).expand(bs, L, J, self.d_model)
		Eqp = Eq.view(bs, 1, J, self.d_model).expand(bs, L, J, self.d_model)

		Epa = Ep.view(bs, L, 1, self.d_model).expand(bs, L, N, self.d_model)
		Eap = Ea.view(bs, 1, N, self.d_model).expand(bs, L, N, self.d_model)

		Eqa = Eq.view(bs, J, 1, self.d_model).expand(bs, J, N, self.d_model)
		Eaq = Ea.view(bs, 1, N, self.d_model).expand(bs, J, N, self.d_model)

		Upq = self.pq_similarity_linear(torch.cat([Epq, Eqp, torch.mul(Epq, Eqp)], dim = -1)).squeeze(-1)  # [bs x L x J]
		Upa = self.pa_similarity_linear(torch.cat([Epa, Eap, torch.mul(Epa, Eap)], dim = -1)).squeeze(-1)  # [bs x L x N]
		Uqa = self.qa_similarity_linear(torch.cat([Eqa, Eaq, torch.mul(Eqa, Eaq)], dim = -1)).squeeze(-1)  # [bs x J x N]

		Upq = Upq.masked_fill(mask_p * mask_q.transpose(1, 2) == 0, -1e9)
		Upa = Upa.masked_fill(mask_p * mask_a.transpose(1, 2) == 0, -1e9)
		Uqa = Uqa.masked_fill(mask_q * mask_a.transpose(1, 2) == 0, -1e9)

		Apq = self.dropout(F.softmax(Upq, dim = 2))                  # [bs x L x J]
		Bpq = self.dropout(F.softmax(Upq, dim = 1).transpose(1, 2))  # [bs x J x L]
		Apa = self.dropout(F.softmax(Upa, dim = 2))			         # [bs x L x N]
		Bpa = self.dropout(F.softmax(Upa, dim = 1).transpose(1, 2))  # [bs x N x L]
		Aqa = self.dropout(F.softmax(Uqa, dim = 2))                  # [bs x J x N]
		Bqa = self.dropout(F.softmax(Uqa, dim = 1).transpose(1, 2))  # [bs x N x J]

		Apq_bar = torch.matmul(Apq, Eq)  # [bs x L x d_model]
		Bpq_bar = torch.matmul(Bpq, Ep)  # [bs x J x d_model]
		Apa_bar = torch.matmul(Apa, Ea)  # [bs x L x d_model]
		Bpa_bar = torch.matmul(Bpa, Ep)  # [bs x N x d_model]
		Aqa_bar = torch.matmul(Aqa, Ea)  # [bs x J x d_model]
		Bqa_bar = torch.matmul(Bqa, Eq)  # [bs x N x d_model]

		Gp = torch.cat([Ep, Apq_bar, Apa_bar, torch.mul(Ep, Apq_bar), torch.mul(Ep, Apa_bar),
						torch.matmul(Apq, Bpq_bar), torch.matmul(Apq, Aqa_bar),
						torch.matmul(Apa, Bpa_bar), torch.matmul(Apa, Bqa_bar)], dim = -1)
		Gq = torch.cat([Eq, Bpq_bar, Aqa_bar, torch.mul(Eq, Bpq_bar), torch.mul(Eq, Aqa_bar),
						torch.matmul(Bpq, Apq_bar), torch.matmul(Bpq, Apa_bar),
						torch.matmul(Aqa, Bpa_bar), torch.matmul(Aqa, Bqa_bar)], dim = -1)
		Ga = torch.cat([Ea, Bpa_bar, Bqa_bar, torch.mul(Ea, Bpa_bar), torch.mul(Ea, Bqa_bar),
						torch.matmul(Bpa, Apq_bar), torch.matmul(Bpa, Apa_bar),
						torch.matmul(Bqa, Bpq_bar), torch.matmul(Bqa, Aqa_bar)], dim = -1)

		return Gp, Gq, Ga


class AdditiveAttention(nn.Module):
	"""
	Attention module that computes a distribution over the tokens in the key,
	as well as a context vector, given the tokens in the query. The implementation
	is very similar to Bahdanau et al 2015.
	Here it is used only to compute the additive attention between the questions
	and the answers at the last part of the decoder. The context vector is used to
	estimate the lambdas, which are the weights of the distributions in the copying
	mechanism and the alphas are used to define the distribution over the tokens
	in the question.
	"""

	def __init__(self, d_model):
		super(AdditiveAttention, self).__init__()

		self.d_model = d_model

		self.q_linear = nn.Linear(d_model, d_model)
		self.k_linear = nn.Linear(d_model, d_model, bias=False)
		self.energy_layer = nn.Sequential(nn.Tanh(), nn.Linear(d_model, 1, bias=False))

	def forward(self, k, v, q, mask):
		"""
		Args
			k: output of the encoder for question representations (Mq)
				(3d float tensor) [bs x J x d_model]
			v: is the same as k
			q: output of the decoder for answer representation (Ma)
				(3d float tensor) [bs x T x d_model]
			mask: corresponding mask for the keys
				(3d long tensor) [bs x J x 1]
		Returns
			context: a vector representation for the answers informed by the questions
				(3d float tensor) [bs x T x d_model]
			alphas: a distribution over the tokens in the questions
				(3d float tensor) [bs x T x J]
		"""

		J = k.size(-2)  # seq len for questions
		T = q.size(-2)  # seq len for ansers
		bs = k.size(0)  # batch size

		# linear transformation of key and query
		k = self.k_linear(k)
		q = self.q_linear(q)

		# create extra dimensions in k and q that corresponds to the seq_len of the other one
		k = k.unsqueeze(2).expand(bs, J, T, self.d_model)  # (4d) [bs x J x T x d_model]
		q = (
			q.unsqueeze(2).expand(bs, T, J, self.d_model).transpose(1, 2)
		)  # (4d) [bs x J x T x d_model]

		# this layer adds the representation of every token in the question (J)
		# to every token in the representation of the answer (T), thats why we expanded the dimensions before
		# then reduces the last dimension of the tokens (d_model) by passing them through a (d_model x 1) weight
		scores = self.energy_layer(k + q).squeeze(-1)  # (3d) [bs x J x T]

		# apply the mask by zeroing all the correspoding positions
		scores = scores.masked_fill(mask == 0, -1e9)

		# normallization across the question tokens dimension (J) since this is the key
		# to get a distribution over the questions tokens for each example and for each decoding step "t"
		alphas = F.softmax(scores.transpose(-2, -1), dim=-1)  # (3d) [bs x T x J]

		# reduce the seq_len_questions dimension to get a context vector
		context = torch.bmm(alphas, v)  # [bs x T x d_model]

		return context, alphas


class CombinedAttention(nn.Module):
	"""
	This is very similar to the Additive Attention module, it computes the context and alphas
	using the passage representations as keys and values and the answer representations as queries.
	The alphas are modified by the betas, which are the passage relevancies, as calculated by the
	ranker module, thus performing a sort of sentence level attention, where the model learns to
	attend only to the relevant passages.

	"""

	def __init__(self, d_model):
		super(CombinedAttention, self).__init__()

		self.d_model = d_model
		self.q_linear = nn.Linear(d_model, d_model)
		self.k_linear = nn.Linear(d_model, d_model, bias=False)
		self.energy_layer = nn.Sequential(nn.Tanh(), nn.Linear(d_model, 1, bias=False))

	def forward(self, k, v, q, mask, betas):
		"""
		Args
			k: output of the encoder for passage representations (Mp)
				(4d float tensor) [bs x K x L x d_model]
			v: is the same as k
			q: output of the decoder for answer representation (Ma)
				(3d float tensor) [bs x T x d_model]
			mask: corresponding mask for the keys
				(4d long tensor) [bs x K x L x 1]
			betas: the relevance scores from the ranker
				(2d float tensor) [n_ans x K]
		Returns
			context: a vector representation for the answers informed by the passages
				(3d float tensor) [bs x T x d_model]
			modified_alphas: a distribution over the tokens in the passages (renormallized by the betas)
				(3d float tensor) [bs x T x K * L]
		"""

		# dimension sizes
		# batch size (bs)
		# num passages (K)
		# seq_len_passages (L)
		# seq_len_answers (T)
		# concatinated length of passages (KL)
		bs, K, L, _ = k.size()
		T = q.size(1)
		KL = int(K * L)

		# merge the K and seq_len_passages
		# basically this concatinates all the passages in an example making one long sequence
		# with length K * L
		k = k.view(bs, KL, self.d_model)
		mask = mask.view(bs, KL, 1)

		# linear transformation on keys and queries
		k = self.k_linear(k)
		q = self.q_linear(q)

		# create extra dimensions in k and q that corresponds to the seq_len of the other one
		# the same is done for the mask
		k = k.unsqueeze(2).expand(
			bs, KL, T, self.d_model
		)  # (4d) [bs x KL x T x d_model]
		q = (
			q.unsqueeze(2).expand(bs, T, KL, self.d_model).transpose(1, 2)
		)  # (4d) [bs x KL x T x d_model]

		# this layer adds the representation of every token in the concatination of the passages (KL)
		# to every token in the representation of the answer (T), thats why we expanded the dimensions bewfore
		# then reduces the last dimension of the tokens (d_model) by passing them through a d_model x 1 weight
		scores = self.energy_layer(k + q).squeeze(-1)  # (3d) [bs x KL x T]

		# apply the mask
		scores = scores.masked_fill(mask == 0, -1e9)

		# normallization across the passage dimension (KL) since this is the key
		alphas = F.softmax(scores, dim=1)  # (3d) [bs x KL x T]

		alphas = alphas.view(bs, K, L, T)

		# renormallize with the relevance scores
		modified_alphas = alphas * betas.view(bs, K, 1, 1)
		modified_alphas /= modified_alphas.sum(dim=(1, 2), keepdims=True)
		modified_alphas = modified_alphas.view(bs, KL, T).transpose(
			-2, -1
		)  # [bs x T x KL]
		v = v.view(bs, KL, self.d_model)

		# reduce the seq_len_questions dimension to get a context vector
		context = torch.bmm(modified_alphas, v)  # (3d) [bs x seq_len_aswers x d_model]

		return context, modified_alphas