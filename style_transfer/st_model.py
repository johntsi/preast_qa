import torch
from torch import nn
import torch.nn.functional as F
import os

from create_embeddings import create_embeddings
from embedder import Embedder
from reader import StReader
from transformer_blocks import StTransformerDecoder
from multi_src_pointer_gen import StMultiSrcPointerGen
import my_constants


class StModel(nn.Module):
	"""
	Transformer-based encoder-decoder that learns a function
		f(passage, question, extractive answers) --> abstractive answer

	The encoder is made of shared and sequence-specific transformer encoder-blocks
		for the three input sequences and several dual attention modules
		to blend cross-sequence information

	The decoder is modified transformer decoder, where each layers has 2 more encoder-decoder
		attention sublayers to account for the extra input sequences

	A multi-source-pointer-generator is employed to either copy from the three input sequences
		or generate a token from the fixed vocabulary.
	"""
	def __init__(self, args, fixed_token2id):
		"""
		Args:
			args: argument parser object
			fixed_token2id: dict[str: int]
		"""
		
		super(StModel, self).__init__()

		# load glove embeddings for the fixed vocabulary
		data_path = "./../data"
		if "embeddings.pt" not in os.listdir(data_path):
			glove_path = os.path.join(data_path, args.embeddings_name)
			create_embeddings(glove_path)
		glove_vectors = torch.load(os.path.join(data_path, "embeddings.pt"))

		self.pad_idx = fixed_token2id[my_constants.pad_token]
		self.eos_idx = fixed_token2id[my_constants.eos_token]
		self.unk_idx = fixed_token2id[my_constants.unk_token]
		self.bos_idx = fixed_token2id[my_constants.nlg_token]
		self.cls_idx = fixed_token2id[my_constants.cls_token]
		self.qa_idx = fixed_token2id[my_constants.qa_token]

		self.max_seq_len_passage = args.max_seq_len_passage
		self.max_seq_len_question = args.max_seq_len_question
		self.max_seq_len_qa_answer = args.max_seq_len_qa_answer
		self.max_seq_len_nlg_answer = args.max_seq_len_nlg_answer

		max_seq_len = max(self.max_seq_len_passage, self.max_seq_len_question, self.max_seq_len_qa_answer, self.max_seq_len_nlg_answer)

		self.d_vocab, self.d_emb = glove_vectors.size()
		self.d_model = args.d_model

		self.embedder = Embedder(glove_vectors, self.pad_idx, args.emb_dropout_rate, max_seq_len)

		self.reader = StReader(args, self.d_emb, self.pad_idx)

		self.decoder = StTransformerDecoder(args, self.d_emb)

		# special token mask (exclude this from output vocabulary distribution)
		special_mask_idx = [self.pad_idx, self.unk_idx, self.bos_idx, self.cls_idx, self.qa_idx]
		self.multi_source_pointer_gen = StMultiSrcPointerGen(self.d_model, self.d_vocab, self.d_emb, special_mask_idx)

		# whether to share the weights of input and output embeddings
		if args.tie_embeddings:
			self.multi_source_pointer_gen.vocab_generator[1].weight = nn.Parameter(self.embedder.embedding_layer.weight.data)


	def forward(self, p, q, qa, source_ext, d_ext_vocab, nlg = None, autoregressive = False):
		"""
		Inputs
			p: The indices of the tokens in the passages
				(2d long tensor) [batch_size x seq_len_passages]
			q: The indices of the tokens in the question
				(2d long tensor) [batch_size x seq_len_question]
			qa: The indices of the tokens in the QA answer
				(2d long tensor) [batch_size x seq_len_qa]
			source_ext: The indicies of the tokens of the concatination of passages, question and QA answer
				(2d long tesnor) [batch_size x num_passages * seq_len_passages + seq_len_question + seq_len_qa]
			d_ext_vocab:  the size of the extended vocabulary (int)
			nlg: The indices of the tokens in the NLG answer
				(2d long tensor) [batch_size x seq_len_nlg]
			autoregressive: whether the model is in autoregressive mode (bool)
		Returns
			(Regular)
			dec_scores: The probabilities in the extedned vocabulary of the predictions of the NLG answers
				(3d float tensor) [batch_size x seq_len_nlg x d_ext_vocab]

			(Autoregressive)
			preds: Indices in the extended vocabulary of the NLG predictions
				(2d long tensor) [batch_size x seq_len_nlg]
			avg_lambdas: the average weight of each distribution for each answer
				(2d float tensor) [batch_size x 4]
			lengths: the length of each answer
				(1d long tensor) [bs]
		"""

		bs = p.size(0)  # batch_size

		d_ext_vocab = max(d_ext_vocab, self.d_vocab)

		current_device = torch.device(f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_available() else torch.device("cpu")

		mask_p, mask_q, mask_qa, mask_nlg = self.create_masks(p, q, qa, nlg, device = current_device)

		# pass the sequences trough the embedding layer
		# 3d float tensor [bs x len x d_emb]
		p_emb = self.embedder(p)
		q_emb = self.embedder(q)
		qa_emb = self.embedder(qa)

		# pass the sequences through the reader
		# 3d float tensors [bs x len x d_model]
		Mp, Mq, Mqa = self.reader(p_emb, q_emb, qa_emb, mask_p, mask_q, mask_qa)

		if not autoregressive:

			nlg_emb = self.embedder(nlg)

			# pass the outputs of the encoder and the answer input tokens through the decoder
			Mnlg = self.decoder(Mp, Mq, Mqa, nlg_emb, mask_p, mask_q, mask_qa, mask_nlg)  # (3d): [bs x T x d_model]

			# pass the betas, the outputs of the encoder and the decoder through the multi-source pointer generator
			# to get the predictions for the answer sequence, along with the source tokens for estimating the final distr
			# (3d float): [bs x T x d_ext_vocab]
			dec_scores, lambdas = self.multi_source_pointer_gen(Mp, Mq, Mqa, Mnlg, mask_p, mask_q, mask_qa,
															source_ext, current_device, d_ext_vocab)

			return dec_scores, lambdas

		else:

			cond1, cond2 = True, True
			preds, sum_lambdas, lengths = None, None, None
			not_completed = torch.ones(bs, dtype = torch.bool, device = current_device)
			decoding_step = 1

			bos_init = torch.ones(bs, dtype = torch.long, device = current_device) * self.bos_idx

			seq_indices = bos_init.clone()

			while cond1 and cond2:

				num_not_completed = not_completed.long().sum()

				# (3d): [num_not_completed x decoding_step x d_emb]
				nlg_emb = self.embedder(seq_indices[not_completed].view(num_not_completed, decoding_step))

				# pass the outputs of the encoder and the answer input tokens through the decoder
				Mnlg = self.decoder(Mp[not_completed], Mq[not_completed], Mqa[not_completed],
									nlg_emb, mask_p[not_completed], mask_q[not_completed], mask_qa[not_completed])  # (3d): [num_not_completed x decoding_step x d_model]

				# pass the betas, the outputs of the encoder and the decoder through the multi-source pointer generator
				# to get the predictions for the answer sequence, along with the source tokens for estimating the final distr
				# dec_scores: (3d): [num_not_completed x decoding_step x d_ext_vocab]
				dec_scores, lambdas = self.multi_source_pointer_gen(Mp[not_completed], Mq[not_completed],
												Mqa[not_completed], Mnlg, mask_p[not_completed], mask_q[not_completed],
												mask_qa[not_completed], source_ext[not_completed],
												current_device, d_ext_vocab)

				dec_scores_i = torch.zeros((bs, d_ext_vocab), dtype = torch.float, device = current_device)
				dec_scores_i[:, self.pad_idx] = 1.

				dec_scores_i[not_completed, :] = dec_scores[:, -1, :]

				preds_i = torch.argmax(dec_scores_i, dim = -1)

				# append to the predicitons and lambdas of this decoding step to parent tensors
				if preds is not None:
					preds = torch.cat([preds, preds_i.unsqueeze(1)], dim = 1)  # [bs x decoding_step]
					sum_lambdas[not_completed] += lambdas[:, -1, :]  # [bs x 4]
					lengths += not_completed.long()  # [bs]

				# initialize the parents tensors with the first decoding step predictions and lambdas
				else:
					preds = preds_i.contiguous().unsqueeze(1)  # [bs x 1]
					sum_lambdas = lambdas[:, -1, :]  # [bs x 4]
					lengths = torch.ones(bs, dtype = torch.long, device = current_device)  # [bs]

				# prepare the extra input for the next decoding step
				# if the prediction was not in the fixed vocab, we correct it with the unk token
				seq_indices = torch.cat([bos_init.unsqueeze(1), preds], dim = 1)
				seq_indices[(seq_indices > self.d_vocab - 1)] = self.unk_idx

				# get the new not_completed mask
				not_completed = (seq_indices != self.eos_idx).all(dim = 1)

				# update conditions for stopping
				cond1 = not_completed.any().item()
				cond2 = seq_indices.size(-1) < self.max_seq_len_nlg_answer

				# increase the decoding step
				decoding_step += 1

			padding = self.max_seq_len_nlg_answer - preds.size(1) - 1

			if padding > 0:
				preds = F.pad(preds, (0, padding), "constant", self.pad_idx)

			avg_lambdas = sum_lambdas / lengths.unsqueeze(1)

			return preds, avg_lambdas, lengths


	def create_masks(self, p, q, qa, nlg = None, device = None):
		"""
		Args:
			p: 2d long tensor [bs x L]
			q: 2d long tensor [bs x J]
			qa: 2d long tensor [bs x N]
			nlg: 2d long tensor [bs x T]
			device: torch.device
		Returns:
			mask_p: 3d bool tensor [bs x L x 1]
			mask_q: 3d bool tensor [bs x J x 1]
			mask_qa: 3d bool tensor [bs x N x 1]
			mask_nlg: 3d bool tensor [bs x T x T]
		"""

		# create standard masks
		mask_p = (p != self.pad_idx).unsqueeze(-1)
		mask_q = (q != self.pad_idx).unsqueeze(-1)
		mask_a = (qa != self.pad_idx).unsqueeze(-1)

		if nlg is not None:

			# standard mask for answer
			mask_nlg_src = (nlg != self.pad_idx).unsqueeze(-1)  # (3d): [batch_size x seq_len_answer x 1]

			# no-peak into the future mask for answer
			T = nlg.size(1)
			mask_nlg_trg = torch.tril(torch.ones([1, T, T], dtype = torch.float, device = device))  # (3d) [1 x seq_len_answer x seq_len_answer]

			# combined mask for answer
			mask_nlg = torch.mul(mask_nlg_src, mask_nlg_trg).transpose(-1, -2).bool()
		else:
			mask_nlg = None

		return mask_p, mask_q, mask_a, mask_nlg