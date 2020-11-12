import torch
from torch import nn
import torch.nn.functional as F
import os

from create_embeddings import create_embeddings
from embedder import Embedder
from reader import QaReader
from ranker import Ranker
from classifier import Classifier
from transformer_blocks import QaTransformerDecoder
from multi_src_pointer_gen import QaMultiSrcPointerGen

import my_constants


class QaModel(nn.Module):
	"""
	Transformer Encoder-Decoder based on the Masque model of Nishida et. al
	("Multi-Style Generative Reading Comprehension")
	"""

	def __init__(self, args, fixed_token2id, device, only_encoder = False):
		"""
		Args:
			args: argument_parser object containing all the necessary configurations of the model
			fixed_token2id: dict[str: int]
				dictionary of the tokens in the fixed vocabulary
			device: torch.device
			only_encoder: bool
				whether to use only the encoder part of the model, which means
				using the embedder, reader, ranker and classifier modules
				and outputing only ranking and classification scores
		"""
		super(QaModel, self).__init__()

		self.device = device
		self.only_encoder = only_encoder

		self.rnk_method = args.rnk_method

		self.fixed_token2id = fixed_token2id
		self.fixed_id2token = {v: k for k, v in self.fixed_token2id.items()}

		self.d_model = args.d_model

		self.tie_embeddings = args.tie_embeddings

		self.pad_idx = self.fixed_token2id[my_constants.pad_token]
		self.cls_idx = self.fixed_token2id[my_constants.cls_token]
		self.eos_idx = self.fixed_token2id[my_constants.eos_token]
		self.unk_idx = self.fixed_token2id[my_constants.unk_token]
		self.qa_idx = self.fixed_token2id[my_constants.qa_token]
		self.nlg_idx = self.fixed_token2id[my_constants.nlg_token]

		# load glove embeddings for the fixed vocabulary
		data_path = "./../data"
		if "embeddings.pt" not in os.listdir(data_path):
			glove_path = os.path.join(data_path, args.embeddings_name)
			create_embeddings(glove_path)
		glove_vectors = torch.load(os.path.join(data_path, "embeddings.pt"))

		self.max_seq_len_passage = args.seq_len_passage
		self.max_seq_len_question = args.seq_len_question
		self.max_seq_len_answer = args.seq_len_answer
		self.max_seq_len_dec = args.max_seq_len_dec
		self.max_seq_len = max([self.max_seq_len_dec, self.max_seq_len_answer, self.max_seq_len_question, self.max_seq_len_passage])

		self.d_vocab, self.d_emb = glove_vectors.size()

		self.embedder = Embedder(glove_vectors, self.pad_idx, args.emb_dropout_rate, self.max_seq_len)

		self.reader = QaReader(args, self.d_emb, self.pad_idx)

		self.ranker = Ranker(args)
		
		self.classifier = Classifier(args)

		if not self.only_encoder:

			self.decoder = QaTransformerDecoder(args.num_layers_dec, self.d_emb, args.d_model,
				args.d_inner, args.heads, args.dropout_rate)

			# special token mask (exclude this from output vocabulary distribution)
			special_mask_idx = [self.pad_idx, self.cls_idx, self.unk_idx, self.qa_idx, self.nlg_idx]
			self.multi_source_pointer_gen = QaMultiSrcPointerGen(args.d_model, self.d_vocab, self.d_emb, special_mask_idx)

			# whether to share the weights of input and output embeddings
			if self.tie_embeddings:
				self.multi_source_pointer_gen.vocab_generator[1].weight = nn.Parameter(self.embedder.embedding_layer.weight.data)

		self.print_model_params()

	def forward(self, passage_fixed_vectors, query_fixed_vectors, passage_query_ext_vectors = None, is_answerable = None,
					answer_src_vectors = None, maximum_vocab_id = None, autoregressive = False, style = "nlg"):
		"""
		Args:
			passage_fixed_vectors: The indices of the tokens in the passages
				(3d long tensor) [bs x K x L]
			query_fixed_vectors: The indices of the tokens in the questions
				(2d long tensor) [bs x J]
			passage_query_ext_vectors: The indicies of the tokens of the concatination of passages and questions
				in the extended vocab, the non-answrable examples are just padding vectors
				(2d long tesnor) [bs x K * L + J]
			is_answerable: Indicates whether the question is anserable given the set of passages (ground-truth)
				(1d bool tensor) [bs]
			answer_src_vectors: The indices of the input answer where the first idx is the style idx
				non-answerable examples are padding vectors
				(2d long tensor) [bs x T]
			maximum_vocab_id: The highest index in the fixed vocabulary for this batch (int)
			autoregressive: Whether the forward pass is autoregressive (bool)
			style: the style of the answer for the autoregressive forward pass (str)

		Returns:
			In normal mode (training):
				dec_scores_batch: probabilities over the extended vocabulary for the shifted right positions in the answer
					non-answerable examples are zero vectors
					(3d float tensor) [bs x T x d_vocab_ext]
				cls_scores: the answer possibility score for each example (before sigmoid)
					(1d float tensor) [bs]
				rnk_scores: the output of the ranker (to calculate the ranking loss)
					for pointwise: (2d float tensor) [bs x K]
					for pairwie: (4d float tensor) [bs x K x K x 3]
				betas: the relevance probabilities for each passage
					(2d float tensor) [bs x K]
				lambdas_batch: the weighting factors for each distribution for each answer for each step
					non-answerable examples are zero vectors
					(3d float tensor) [bs x T x 3]

			In autoregressive mode (evaluation):
				preds: the predicted indices of the answer in the extended vocabulary for each example
					(2d long tensor) [bs x T]
				possibilities: the probability that each example is answerable
					(1d float tensor) [bs]
				avg_lambdas: the average weight of each distribution of the multi-source-pointer-generator
					[generate_from_fix_vocab, copy_query, copy_passage]
					(2d float tensor) [bs x 3]
				betas: the relevance probabilities for each example for each passage
					(2d float tensor) [bs x K]
				lengths: the length of each answer
					(1d long tensor) [bs]
		"""

		bs = passage_fixed_vectors.size(0)  # batch_size
		K = passage_fixed_vectors.size(1)  # num_passages
		L = passage_fixed_vectors.size(2)  # seq_len_passages

		current_device = torch.device(f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_available() else torch.device("cpu")

		# boolean masks, where 1 indicates answer_src_vectors valid position
		# (4d bool) [batch_size x num_passages x seq_len_passage x 1]
		# (3d bool) [batch_size x seq_len_question x 1]
		# (3d bool) [batch_size x seq_len_answer x seq_len_answer]
		mask_p, mask_q, mask_a = self.create_masks(passage_fixed_vectors, query_fixed_vectors, answer_src_vectors, current_device)

		p_emb = self.embedder(passage_fixed_vectors.view(bs * K, L))
		q_emb = self.embedder(query_fixed_vectors)

		# pass the passages and the questions through the reader
		# Mp: concatinated representations of the passages (4d) [batch_size x num_passages x seq_len_passages x d_model]
		# Mq: reprentation of questions (3d) [batch_size x seq_len_question x d_model]
		Mp, Mq = self.reader(p_emb, q_emb, mask_p, mask_q)

		# get the cls token representations
		Mp_cls = Mp[:, :, 0, :]  # (3d): [batch_size x num_passages x d_model]

		# classification scores for each example
		cls_scores = self.classifier(Mp_cls)  # (1d): [batch_size]

		# ranking scores
		# for pointwise ranker: (2d float tensor) [bs x K]
		# for pairwise ranker: (4d float tensor) [bs x K x K x 3]
		rnk_scores = self.ranker(Mp_cls, mask_p)  # (2d): [batch_size x num_passages]

		# relevance probabilities
		betas = self.calc_betas(rnk_scores)  # (2d float tensor) [bs x K]

		# end here if only encoder is active or non of the examples are answerable
		if (not hasattr(self, "decoder")) or (not is_answerable.any().item()):
			return None, rnk_scores, cls_scores, betas, None

		d_ext_vocab = max(maximum_vocab_id, self.d_vocab)

		if not autoregressive:
			T = answer_src_vectors.size(1)

			# to store outputs related to the decoder (maintain consistency with dataparallel)
			dec_scores_batch = torch.zeros([bs, T, d_ext_vocab], dtype = torch.float, device = current_device)
			lambdas_batch = torch.zeros([bs, T, 3], dtype = torch.float, device = current_device)

			# continue only with the examples that are answerable
			# the first dimension of every one of the following tensors is now <n_ans>
			Mp = Mp[is_answerable]
			Mq = Mq[is_answerable]
			mask_p = mask_p[is_answerable]
			mask_q = mask_q[is_answerable]
			mask_a = mask_a[is_answerable]
			passage_query_ext_vectors = passage_query_ext_vectors[is_answerable]

			a_emb = self.embedder(answer_src_vectors[is_answerable])

			# pass the outputs of the encoder and the answer input tokens through the decoder
			Ma = self.decoder(Mp, Mq, a_emb, mask_p, mask_q, mask_a)  # (3d): [n_ans x seq_len_answer x d_model]

			# pass the betas, the outputs of the encoder and the decoder through the multi-source pointer generator
			# to get the predictions for the answer sequence, along with the source tokens for estimating the final distr
			# (3d float): [n_ans x seq_len_answer x d_ext_vocab]
			dec_scores_ans, lambdas_ans = self.multi_source_pointer_gen(Mp, Mq, Ma, mask_p, mask_q, betas[is_answerable],
															passage_query_ext_vectors, current_device, d_ext_vocab)

			# put the predictions of the answerable examples to answer_src_vectors tensor of dimensionality batch_size
			# so that the concatination of DataParallel works properly
			dec_scores_batch[is_answerable] = dec_scores_ans
			lambdas_batch[is_answerable] = lambdas_ans

			return dec_scores_batch, rnk_scores, cls_scores, betas, lambdas_batch

		else:

			possibilities = torch.sigmoid(cls_scores)

			is_answerable = possibilities > -1

			style_idx = self.qa_idx if style == "qa" else self.nlg_idx
			
			# prepare initial inputs (exclude non-answerable examples)
			# the first dimension now becomes equal to the number of answerable examples
			# batch_size >= n_ans > 1
			Mp_ = Mp[is_answerable]
			Mq_ = Mq[is_answerable]
			mask_p_ = mask_p[is_answerable]
			mask_q_ = mask_q[is_answerable]
			betas_ = betas[is_answerable]
			pq_ext_ = passage_query_ext_vectors[is_answerable]

			# initialize <answer_src_vectors> with eos_idx for non-answerable and with style_idx for the answerable ones
			style_init = torch.ones(bs, dtype = torch.long, device = current_device) * self.eos_idx
			style_init[is_answerable] = style_idx
			a_ = style_init[is_answerable]

			not_completed = is_answerable

			# initialize stopping conditions
			cond1, cond2 = True, True

			# initialize predictions and lambdas
			preds, sum_lambdas, lengths = None, None, None

			decoding_step = 1

			while cond1 and cond2:

				num_not_completed = not_completed.long().sum()

				a_ = a_.view(num_not_completed, decoding_step)

				a_emb = self.embedder(a_)  # (3d): [num_not_completed x decoding_step x d_emb]

				# pass the outputs of the encoder and the answer input tokens through the decoder
				Ma = self.decoder(Mp_, Mq_, a_emb, mask_p_, mask_q_)  # (3d): [num_not_completed x decoding_step x d_model]

				# pass the betas, the outputs of the encoder and the decoder through the multi-source pointer generator
				# to get the predictions for the answer sequence, along with the source tokens for estimating the final distr
				# dec_scores: (3d): [num_not_completed x decoding_step x d_ext_vocab]
				# lambdas (3d): [num_not_completed x decoding_step x 3]
				dec_scores, lambdas = self.multi_source_pointer_gen(Mp_, Mq_, Ma, mask_p_, mask_q_,
																	betas_, pq_ext_, current_device, d_ext_vocab)

				# initialize the predictions of this step with answer_src_vectors padding vector
				# (2d):  [batch_size x d_ext_vocab]
				dec_scores_i = torch.zeros([bs, d_ext_vocab], dtype = torch.float, device = current_device)
				dec_scores_i[:, self.pad_idx] = 1

				# fill the not_completed examples with the last step predictions of the multi-source-pointer-gen
				dec_scores_i[not_completed, :] = dec_scores[:, -1, :]

				preds_i = torch.argmax(dec_scores_i, dim = -1)

				# append to the predicitons and lambdas of this decoding step to parent tensors
				if preds is not None:
					preds = torch.cat([preds, preds_i.unsqueeze(1)], dim = 1)  # [bs x decoding_step]
					sum_lambdas[not_completed] += lambdas[:, -1, :]  # [bs x 3]
					lengths += not_completed.long()  # [bs]

				# initialize the parents tensors with the first decoding step predictions and lambdas
				else:
					preds = preds_i.contiguous().unsqueeze(1)  # [bs x 1]
					sum_lambdas = lambdas[:, -1, :]  # [bs x 3]
					lengths = torch.ones(bs, dtype = torch.long, device = current_device)  # [bs]

				# prepare the extra input for the next decoding step
				preds_fixed_vocab = preds.clone()
				preds_fixed_vocab[preds_fixed_vocab > self.d_vocab - 1] = self.unk_idx

				# construct the input of the next decoding step by appendinig the this step to the rest
				# (2d): [batch_size x decoding_step + 1]
				answer_src_vectors = torch.cat([style_init.unsqueeze(1), preds_fixed_vocab], dim = -1)

				# get the new not_completed mask
				not_completed = (preds != self.eos_idx).all(dim = 1)

				# prepare inputs for the next decoding step
				a_ = answer_src_vectors[not_completed]
				Mp_ = Mp[not_completed]
				Mq_ = Mq[not_completed]
				mask_p_ = mask_p[not_completed]
				mask_q_ = mask_q[not_completed]
				betas_ = betas[not_completed]
				pq_ext_ = passage_query_ext_vectors[not_completed]

				# update conditions for stopping
				cond1 = not_completed.any().item()
				cond2 = answer_src_vectors.size(-1) < self.max_seq_len_dec

				# increase the decoding step
				decoding_step += 1

			avg_lambdas = sum_lambdas / lengths.unsqueeze(1)

			padding = self.max_seq_len_dec - preds.size(1) - 1

			if padding > 0:
				preds = F.pad(preds, (0, padding), "constant", self.pad_idx)

			return preds, possibilities, avg_lambdas, betas, lengths


	def print_model_params(self):

		total_num_params = 0
		total_memory = 0
		for n, p in self.named_parameters():
			if "embedder" in n:
				num_params_emb = p.nelement()
				memory_emb = p.element_size() * num_params_emb
			total_num_params += p.nelement()
			total_memory += p.element_size() * p.nelement()

		if hasattr(self, "decoder") and self.tie_embeddings:
			total_num_params -= num_params_emb
			total_memory -= memory_emb

		print(f"Total number of parameters: {total_num_params // 1e3}k and total memory: {total_memory // 1e6}MB")


	def calc_betas(self, rnk_scores):

		if self.rnk_method == "pointwise":
			betas = torch.sigmoid(rnk_scores)

		else:
			K = rnk_scores.size(1)

			# average net advantage of each passage
			S = (rnk_scores[:, :, :, 2] - rnk_scores[:, :, :, 0] - rnk_scores[:, :, :, 1]).sum(dim = -1) / K
			betas = F.softmax(S, dim = -1)
			
		return betas

	def create_masks(self, passage_fixed_vectors, query_fixed_vectors, answer_src_vectors = None, device = None):

		# create standard masks
		mask_p = (passage_fixed_vectors != self.pad_idx).unsqueeze(-1)  # (4d): [batch_size x num_passages x seq_len_passages x 1]
		mask_q = (query_fixed_vectors != self.pad_idx).unsqueeze(-1)  # (3d): [batch_size x seq_len_question x 1]

		if answer_src_vectors is not None:

			# standard mask for answer
			mask_a_src = (answer_src_vectors != self.pad_idx).unsqueeze(-1)  # (3d): [batch_size x seq_len_answer x 1]

			# no-peak into the future mask for answer
			T = answer_src_vectors.size(1)
			mask_a_trg = torch.tril(torch.ones([1, T, T], dtype = torch.float, device = device))  # (3d) [1 x seq_len_answer x seq_len_answer]

			# combined mask for answer
			mask_a = torch.mul(mask_a_src, mask_a_trg).transpose(-1, -2).bool()  # (3d) [batch_size x seq_len_answer x seq_len_answer]
		else:
			mask_a = None

		return mask_p, mask_q, mask_a