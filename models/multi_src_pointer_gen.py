import torch
from torch import nn
import torch.nn.functional as F

from attention import CombinedAttention, AdditiveAttention


class QaMultiSrcPointerGen(nn.Module):
	"""
	Multi-Source Pointer-Generator network for generating the answer sequence.
	The idea is similar to the one introduced in the "Get to the point" paper (2017).
	In this implementation the network learns to generate a token from three different distributions:
	(1) the vocabulary distribution defined by the d_vocab most frequent tokens in the training set
		this is produced by passing the Answer input representation (output of the decoder)
		through a softmax layer that uses the inverse glove embedding matrix (same as the input)
	(2) the passage distribution defined by the attention weights (alphas) of the concatiation of passages
		for each example
		this is produced by the Combined Attention module of the passages and the answer
	(3) the question distribution defined by the attention weights (alphas) of the question
		for each example
		this is produced by the Additive Attention module of the question and the answer
	The network learns how to weight each of these distributions into a final distribution
	by calculating lambda paramters from the decoder states Ma, context of passages and
	context of questions.
	"""

	def __init__(self, d_model, d_fixed_vocab, d_emb, special_idx):
		"""
		Args:
			d_model: int
			d_fixed_vocab: int
			d_emb: int
			special_idx: list[int]
		"""
		super(QaMultiSrcPointerGen, self).__init__()
		
		self.d_model = d_model
		self.d_fixed_vocab = d_fixed_vocab

		# prepare mask for the output distribution of the fixed vocabulary
		self.special_token_mask = torch.ones(self.d_fixed_vocab, dtype = torch.bool)
		for idx in special_idx:
			self.special_token_mask[idx] = 0
		self.special_token_mask = self.special_token_mask.view(1, 1, -1)

		# question and passage attentions
		self.question_attn = AdditiveAttention(d_model)
		self.passage_attn = CombinedAttention(d_model)

		# 2-layer feed-forward network to project scores on the fixed vocabulary
		self.vocab_generator = nn.Sequential(nn.Linear(d_model, d_emb),
											nn.Linear(d_emb, self.d_fixed_vocab, bias = False))

		# feed-forward network to get the mixing paramters of each distribution
		self.ff_mixture = nn.Sequential(nn.Linear(3 * d_model, 3),
									nn.Softmax(dim = -1))

	def forward(self, Mp, Mq, Ma, mask_p, mask_q, betas, pq_ext, device, d_ext_vocab):
		"""
		Args
			Mp: Passage Representation (concatinated, output of the Reader)
				(3d float tensor) [n_ans x K * L x d_model]
			Mq: Question Representation (output of the Reader)
				(3d float tensor) [n_ans x J x d_model]
			Ma: Answer Representation (ouput of the Decoder)
				(3d float tensor) [n_ans x T x d_model]
			mask_p: passage mask
				(3d bool tensor) [n_ans x K * L x 1]
			mask_q: question mask
				(3d bool tensor) [n_ans x J x 1]
			betas: relevance scores of each passage
				(2d float tensor) [n_ans x K]
			pq_ext: representation of the concatination of passages and question in the extended vocabulary
				(2d long tensor), [n_ans x K * L + J]
			device: torch device object
			d_ext_vocab: size of the extended vocabulary, integer
		Returns
			ext_vocab_distr: distribution over the extended vocabulary
				(3d float tensor), [n_ans x T x d_vocab_ext]
				*** the last dimension, d_vocab_ext, is of variable size
			lambdas: weighting factors of each distribution per example per decoding step
				(3d float tensor), [n_ans x T x 3]
		"""

		# context and attention weights for questions
		# contex_q (3d) [n_ans x seq_len_answers x d_model]
		# alphas_q (3d) [n_ans x seq_len_answers x seq_len_questions]
		context_q, alphas_q = self.question_attn(Mq, Mq, Ma, mask_q)

		# context and attention weights for passages
		# contex_q (3d) [n_ans x seq_len_answers x d_model]
		# mod_alphas_p (3d) [n_ans x seq_len_answers x num_passages * seq_len_passages]
		context_p, mod_alphas_p = self.passage_attn(Mp, Mp, Ma, mask_p, betas)

		# distribution over the fixed vocabulary
		# prevent model assigning probability to special tokens from the input side
		fixed_vocab_logits = self.vocab_generator(Ma).masked_fill(self.special_token_mask.to(device) == 0, -1e1)
		fixed_vocab_distr = F.softmax(fixed_vocab_logits, dim = -1)  # (3d) [n_ans x seq_len_answer x d_vocab]

		n_ans, T, _ = fixed_vocab_distr.size()

		# mixture weights for the 3 distributions
		# mixture_input: (3d) [n_ans x seq_len_answers x 3 * d_model]
		lambdas = self.ff_mixture(torch.cat([Ma, context_q, context_p], dim = -1))  # (3d) [n_ans x seq_len_answers x 3]

		lambda_v, lambda_q, lambda_p = torch.split(lambdas, 1, dim = -1)

		# modify each distribution by its weighting factor lambda (dimensions stay the same)
		fixed_vocab_distr = torch.mul(fixed_vocab_distr, lambda_v)
		alphas_q = torch.mul(alphas_q, lambda_q)
		mod_alphas_p = torch.mul(mod_alphas_p, lambda_p)

		# concatinate the passage and question distributions
		alphas_pq = torch.cat([mod_alphas_p, alphas_q], dim = -1)  # (3d) [n_ans x seq_len_answers x (K * L + J)]

		# project the indices of the passages and questions (extended vocabulary) and their attention weights
		# to the extended vocabulary distribution
		# the one-hot encoding is really expensive memory-wise so do the operation in batches
		# to avoid OOM errors
		num = 3
		n = n_ans // num + 1
		ext_vocab_distr = torch.cat([torch.matmul(alphas_pq[n * i: (i + 1) * n],
									torch.nn.functional.one_hot(pq_ext[n * i: (i + 1) * n], d_ext_vocab).float())
									for i in range(num)], dim = 0)

		# combine the extended and fixed distributions
		ext_vocab_distr[:, :, :self.d_fixed_vocab] += fixed_vocab_distr

		return ext_vocab_distr, lambdas


class StMultiSrcPointerGen(nn.Module):
	"""
	Similar to QAMultiSrcPointerGen but for combining 4 distributions
	"""

	def __init__(self, d_model, d_fixed_vocab, d_emb, special_idx):
		"""
		Args:
			d_model: int
			d_fixed_vocab: int
			d_emb: int
			special_idx: list[int]
		"""

		super(StMultiSrcPointerGen, self).__init__()
		
		self.d_model = d_model
		self.d_fixed_vocab = d_fixed_vocab
		self.num_distr = 4

		self.special_token_mask = torch.ones(self.d_fixed_vocab)
		for idx in special_idx:
			self.special_token_mask[idx] = 0
		self.special_token_mask = self.special_token_mask.view(1, 1, -1).bool()

		self.question_attn = AdditiveAttention(d_model)
		self.qa_attn = AdditiveAttention(d_model)
		self.passage_attn = AdditiveAttention(d_model)

		self.vocab_generator = nn.Sequential(nn.Linear(d_model, d_emb),
											nn.Linear(d_emb, self.d_fixed_vocab, bias = False))

		self.ff4_mixture = nn.Sequential(nn.Linear(self.num_distr * d_model, self.num_distr),
										nn.Softmax(dim = -1))

	def forward(self, Mp, Mq, Mqa, Mnlg, mask_p, mask_q, mask_qa, source_ext, device, d_ext_vocab):
		"""
		Args
			Mp: Passage Representation (concatinated, output of the Reader)
				(3d float tensor) [bs x L x d_model]
			Mq: Question Representation (output of the Reader)
				(3d float tensor) [bs x J x d_model]
			Mqa: QA_answer Representation (output of the Reader)
				(3d float tensor) [bs x N x d_model]
			Mnlg: Nlg_Answer Representation (ouput of the Decoder)
				(3d float tensor) [bs x T x d_model]
			mask_p: passage mask
				(3d bool tensor) [bs x L x 1]
			mask_q: question mask
				(3d bool tensor) [bs x J x 1]
			mask_qa: qa_answer mask
				(3d bool tensor) [bs x N x 1]
			betas: relevance scores of each passage
				(2d float tensor) [bs x K]
			source_ext: representation of the concatination of the three source seqiences
						in the extended vocabulary
				(2d long tensor), [bs x L + J + N]
			device: torch device object
			d_ext_vocab: size of the extended vocabulary, integer
		Returns
			ext_vocab_distr: distribution over the extended vocabulary
				(3d float tensor), [bs x T x d_vocab_ext]
				*** the last dimension, d_vocab_ext, is of variable size
			lambdas: weighting factors of each distribution per example per decoding step
				(3d float tensor), [bs x T x 4]
		"""

		# (3d float) [bs x seq_len_passages + seq_len_question + seq_len_qa x d_ext_vocab]
		source_ext_onehot = F.one_hot(source_ext, d_ext_vocab).float()

		context_q, q_distr = self.question_attn(Mq, Mq, Mnlg, mask_q)
		context_qa, qa_distr = self.qa_attn(Mqa, Mqa, Mnlg, mask_qa)
		context_p, p_distr = self.passage_attn(Mp, Mp, Mnlg, mask_p)

		# distribution over the fixed vocabulary
		# prevent model assigning probability to special tokens from the input side
		fixed_vocab_logits = self.vocab_generator(Mnlg).masked_fill(self.special_token_mask.to(device) == 0, -1e9)
		fixed_vocab_distr = F.softmax(fixed_vocab_logits, dim = -1)  # (3d) [bs x seq_len_answer x d_vocab]

		# mixture weights for the 4 distributions
		# mixture_input: (3d) [bs x seq_len_answers x 4 * d_model]
		lambdas = self.ff4_mixture(torch.cat([Mnlg, context_q, context_qa, context_p], dim = -1))  # (3d) [bs x seq_len_answers x 4]

		# split the lambdas and keep them in memory
		lambda_v, lambda_q, lambda_qa, lambda_p = torch.split(lambdas, 1, dim = -1)  # (3d) [bs x seq_len_answers x 1]

		# modify each distribution by its weighting factor lambda (dimensions stay the same)
		fixed_vocab_distr = torch.mul(fixed_vocab_distr, lambda_v)
		p_distr = torch.mul(p_distr, lambda_p)
		q_distr = torch.mul(q_distr, lambda_q)
		qa_distr = torch.mul(qa_distr, lambda_qa)

		# concatinate the passage and question distributions
		source_distr = torch.cat([p_distr, q_distr, qa_distr], dim = -1)  # (3d) [bs x seq_len_answers x (L + J + N)]

		# get distribution over the extended vocabulary
		ext_vocab_distr = torch.matmul(source_distr, source_ext_onehot)  # (3d) [bs x seq_len_answers x d_ext_vocab]

		ext_vocab_distr[:, :, :self.d_fixed_vocab] += fixed_vocab_distr

		return ext_vocab_distr, lambdas