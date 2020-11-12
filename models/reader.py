import torch
from torch import nn
import torch.nn.functional as F

from transformer_blocks import TransformerEncoder
from attention import DualAttention, TripleAttention


class QaReader(nn.Module):
	"""
	This is the encoder part of the model, as described in the
	"Nishida et al. 2019"
	This module has a shared part transformer encoder for the passages and questions,
	where they can be concatinated into one tensor and passed through it to reduce operations.
	Then there is dual-attention module where each passage attends to question
	and the question attends to each of the passages.
	Finally, the passages and the questions are passed through dedicated Tranformer encoders.
	"""

	def __init__(self, args, d_emb, pad_idx):
		"""
		Args:
			args: argument parser object
			d_emb: int
			pad_idx: int
		"""
		
		super(QaReader, self).__init__()

		self.d_model = args.d_model
		self.pad_idx = pad_idx

		self.shared_transformer = TransformerEncoder(
			args.num_layers_shared_enc,
			d_emb,
			args.d_inner,
			args.d_model,
			args.heads,
			args.dropout_rate,
		)

		self.dual_attn = DualAttention(args.d_model, args.dropout_rate)

		self.passage_transformer = TransformerEncoder(
			args.num_layers_passage_enc,
			5 * args.d_model,
			args.d_inner,
			args.d_model,
			args.heads,
			args.dropout_rate,
		)

		if args.include_dec:
			self.question_transformer = TransformerEncoder(
				args.num_layers_question_enc,
				5 * args.d_model,
				args.d_inner,
				args.d_model,
				args.heads,
				args.dropout_rate,
			)

	def forward(self, p_emb, q_emb, mask_p, mask_q):
		"""
		Args
			p_emb: the passage embeddings from the embedder module (concatinated along the batch dimension)
				(3d float tensor) [bs * K x L x d_emb]
			q_emb: the question embeddings from the embedder module
				(3d float tensor) [bs x J x d_emb]
			mask_p: the mask that corresponds to the passages
				(4d bool tensor), [bs x K x L x 1]
			mask_q: the mask that corresponds to the questions
				(3d bool tensor), [bs x J x 1]
		Returns
			Mp: Final Representation of the passages
				(4d float tensor), [bs x K x L x d_model]
			Mq: Final Representation of the questions
				(3d float tensor), [bs x L x d_model]
		"""

		# batch size (bs)
		# number of passages (K)
		# sequence length of passages (L)
		# sequence length of questions (J)
		self.bs, self.K, self.L, _ = mask_p.size()
		self.J = mask_q.size(1)

		# concatinate the passage mask along the batch dimension
		# the passage embeddings are already in this form
		mask_p = mask_p.view(self.bs * self.K, self.L, 1)  # (3d) [bs * K x L x 1]

		# pass passage and question through the shared part of the reader
		# Ep: passage representations, (4d), [bs x K x L x d_model]
		# Eq: question representations, (3d), [bs x J x d_model]
		Ep, Eq = self._forward_concat(p_emb, q_emb, mask_p, mask_q)

		# bring the passages and their masks batch to their original 4d view
		Ep = Ep.view(self.bs, self.K, self.L, self.d_model)
		mask_p = mask_p.view(self.bs, self.K, self.L, 1)

		# Dual attention to fuse information from the passages to the qustion and from the question to the passages
		# Gp (4d): [bs x K x L x 5 * d_model]
		# Gq (3d): [bs x J x 5 * d_model]
		Gp, Gq = self.dual_attn(Ep, Eq, mask_p, mask_q)

		# Concat the passages and their masks along the batch dimension to pass them through the transformer
		Gp = Gp.view(self.bs * self.K, self.L, 5 * self.d_model)
		mask_p = mask_p.view(self.bs * self.K, self.L, 1)

		# pass the passages and questions through their corresponding transformer encoder module
		Mp = self.passage_transformer(Gp, mask_p)  # (3d): [bs * K x L x d_model]

		if hasattr(self, "question_transformer"):
			Mq = self.question_transformer(Gq, mask_q)  # (3d): [bs x J x d_model]
		else:
			Mq = None

		# bring passage representation back to its original 4d view
		Mp = Mp.view(self.bs, self.K, self.L, self.d_model)

		return Mp, Mq

	def _forward_concat(self, p_emb, q_emb, mask_p, mask_q):
		"""
		Concatinates passages and questions and passes them together at once
		through the shared part of the Encoder module.
		"""

		# pad second-to-last dimension of questions and their masks if necessary
		# to make their sequences equal in length
		# in order to concatinate their values and their masks
		if self.L > self.J:
			padding = (0, 0, 0, self.L - self.J)
			mask_q = F.pad(mask_q, padding, "constant", 0)
			q_emb = F.pad(q_emb, padding, "constant", self.pad_idx)

		# concatinate passages and questions and their masks
		pq_emb = torch.cat([p_emb, q_emb], dim=0)  # (3d), [bs * (K + 1) x L x d_emb]
		mask_pq = torch.cat([mask_p, mask_q], dim=0)  # (3d), [bs * (K + 1) x L x 1]

		# pass the concatinated embeddings and their mask through the shared Transformer Encoder
		Epq = self.shared_transformer(
			pq_emb, mask_pq
		)  # (3d), [bs * (K + 1) x L x d_model]

		# split the combined representations back to passages and questions
		# Ep: (3d), [bs * K  x L x d_model]
		# Eq: (3d), [bs x L x d_model]
		Ep, Eq = torch.split(Epq, [self.bs * self.K, self.bs], dim=0)

		# unpad the difference L - J from the sequence length dimension
		Eq = torch.narrow(Eq, 1, 0, self.J)  # (3d), [bs x J x d_model]

		return Ep, Eq


class StReader(nn.Module):
	"""
	Similar to the QAReader but for three source sequences
	"""

	def __init__(self, args, d_emb, pad_idx):
		"""
		Args:
			args: argument parser object
			d_emb: int
			pad_idx: int
		"""
		super(StReader, self).__init__()
		
		self.d_model = args.d_model
		self.d_emb = d_emb
		self.pad_idx = pad_idx
		self.coattention = args.coattention

		self.d_after_dual = 9 * args.d_model

		self.shared_transformer = TransformerEncoder(args.num_layers_shared_enc, d_emb, args.d_inner,
													args.d_model, args.heads, args.dropout_rate)

		if self.coattention == "dual":
			self.dual_attn_pq = DualAttention(args.d_model, args.dropout_rate)
			self.dual_attn_pa = DualAttention(args.d_model, args.dropout_rate)
			self.dual_attn_qa = DualAttention(args.d_model, args.dropout_rate)
		else:
			self.triple_attn = TripleAttention(args.d_model, args.dropout_rate)

		self.passage_transformer = TransformerEncoder(args.num_layers_passage_enc, self.d_after_dual,
														args.d_inner, args.d_model, args.heads, args.dropout_rate)

		self.question_transformer = TransformerEncoder(args.num_layers_question_enc, self.d_after_dual,
														args.d_inner, args.d_model, args.heads, args.dropout_rate)

		self.qa_transformer = TransformerEncoder(args.num_layers_qa_enc, self.d_after_dual,
												args.d_inner, args.d_model, args.heads, args.dropout_rate)

		if self.coattention == "dual":
			self.dual_attn_pq_2 = DualAttention(args.d_model, args.dropout_rate)
			self.dual_attn_pa_2 = DualAttention(args.d_model, args.dropout_rate)
			self.dual_attn_qa_2 = DualAttention(args.d_model, args.dropout_rate)
		else:
			self.triple_attn_2 = TripleAttention(args.d_model, args.dropout_rate)

		self.passage_transformer_2 = TransformerEncoder(args.num_layers_passage_enc_2, self.d_after_dual,
														args.d_inner, args.d_model, args.heads, args.dropout_rate)

		self.question_transformer_2 = TransformerEncoder(args.num_layers_question_enc_2, self.d_after_dual,
														args.d_inner, args.d_model, args.heads, args.dropout_rate)

		self.qa_transformer_2 = TransformerEncoder(args.num_layers_qa_enc_2, self.d_after_dual,
												args.d_inner, args.d_model, args.heads, args.dropout_rate)

	def forward(self, p_emb, q_emb, a_emb, mask_p, mask_q, mask_a):
		"""
		Args:
			p_emb: 2d long tensor [bs x L]
			q_emb: 2d long tensor [bs x J]
			a_emb: 2d long tensor [bs x N]
			mask_p: 3d bool tensor [bs x L x 1]
			mask_q: 3d bool tensor [bs x J x 1]
			mask_a: 3d bool tensor [bs x N x 1]
		Returns:
			Ep: 3d float tensor [bs x L x d_model]
			Eq: 3d float tensor [bs x T x d_model]
			Ea: 3d float tensor [bs x N x d_model]
		"""

		# batch_size, seq_len_passage, seq_len_question, seq_len_qa
		self.bs, self.L, _ = mask_p.size()
		self.J = mask_q.size(1)
		self.N = mask_a.size(1)

		# pass the sequence embeddings through the shared part of the reader
		Ep, Eq, Ea = self._forward_concat(p_emb, q_emb, a_emb, mask_p, mask_q, mask_a)

		# inform each representation from the rest
		if self.coattention == "dual":
			Gpq, Gqp = self.dual_attn_pq(Ep, Eq, mask_p, mask_q)
			Gpa, Gap = self.dual_attn_pa(Ep, Ea, mask_p, mask_a)
			Gqa, Gaq = self.dual_attn_qa(Eq, Ea, mask_q, mask_a)

			Gp = torch.cat([Gpq, Gpa[:, :, self.d_model:]], axis = -1)
			Gq = torch.cat([Gqp, Gqa[:, :, self.d_model:]], axis = -1)
			Ga = torch.cat([Gap, Gaq[:, :, self.d_model:]], axis = -1)
		else:
			Gp, Gq, Ga = self.triple_attn(Ep, Eq, Ea, mask_p, mask_q, mask_a)

		# pass each sequence representation through their corresponding transformer encoder modules
		Mp = self.passage_transformer(Gp, mask_p)
		Mq = self.question_transformer(Gq, mask_q)
		Ma = self.qa_transformer(Ga, mask_a)

		# inform each representation from the rest
		if self.coattention == "dual":
			Gpq, Gqp = self.dual_attn_pq_2(Ep, Eq, mask_p, mask_q)
			Gpa, Gap = self.dual_attn_pa_2(Ep, Ea, mask_p, mask_a)
			Gqa, Gaq = self.dual_attn_qa_2(Eq, Ea, mask_q, mask_a)

			Gp = torch.cat([Gpq, Gpa[:, :, self.d_model:]], axis = -1)
			Gq = torch.cat([Gqp, Gqa[:, :, self.d_model:]], axis = -1)
			Ga = torch.cat([Gap, Gaq[:, :, self.d_model:]], axis = -1)
		else:
			Gp, Gq, Ga = self.triple_attn_2(Ep, Eq, Ea, mask_p, mask_q, mask_a)

		# pass each sequence representation through their corresponding transformer encoder modules
		Mp = self.passage_transformer_2(Gp, mask_p)
		Mq = self.question_transformer_2(Gq, mask_q)
		Ma = self.qa_transformer_2(Ga, mask_a)

		return Mp, Mq, Ma

	def _forward_concat(self, p_emb, q_emb, a_emb, mask_p, mask_q, mask_a):
		"""
		Concatinates the embeddings of the three sequences to pass them
		efficiently trough a shared transformer encoder
		"""

		# find max length among the sequences
		max_len = max([self.L, self.N, self.J])
		narrow_p, narrow_q, narrow_a = False, False, False

		# expand to max length
		if self.J < max_len:
			narrow_q = True
			padding = (0, 0, 0, max_len - self.J)
			mask_q = F.pad(mask_q, padding, "constant", 0)
			q_emb = F.pad(q_emb, padding, "constant", self.pad_idx)

		if self.L < max_len:
			narrow_p = True
			padding = (0, 0, 0, max_len - self.L)
			mask_p = F.pad(mask_p, padding, "constant", 0)
			p_emb = F.pad(p_emb, padding, "constant", self.pad_idx)

		if self.N < max_len:
			narrow_a = True
			padding = (0, 0, 0, max_len - self.N)
			mask_a = F.pad(mask_a, padding, "constant", 0)
			a_emb = F.pad(a_emb, padding, "constant", self.pad_idx)

		# concatinate the sequences and their masks
		pqa_emb = torch.cat([p_emb, q_emb, a_emb], dim = 0)
		mask_pqa = torch.cat([mask_p, mask_q, mask_a], dim = 0)

		# pass the concatinated embeddings and their mask through the shared Transformer Encoder
		Epqa = self.shared_transformer(pqa_emb, mask_pqa)

		# split the combined representations back to individual ones
		# Ep: (3d float tensor), [batch_size x L x d_model]
		# Eq: (3d float tensor), [batch_size x J x d_model]
		# Ea: (3d float tensor), [batch_size x N x d_model]
		Ep, Eq, Ea = torch.split(Epqa, [self.bs, self.bs, self.bs], dim = 0)

		# unpad the difference if the representations where expanded
		if narrow_p: Ep = torch.narrow(Ep, 1, 0, self.L)
		if narrow_q: Eq = torch.narrow(Eq, 1, 0, self.J)
		if narrow_a: Ea = torch.narrow(Ea, 1, 0, self.N)

		return Ep, Eq, Ea