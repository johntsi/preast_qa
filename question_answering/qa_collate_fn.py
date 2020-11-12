import torch

import my_constants
from my_tokenizer import construct_tokenizer

print("Loading fixed vocab in collate function ...")
fixed_token2id = construct_tokenizer().vocab
fixed_vocab = set(fixed_token2id)
d_fixed_vocab = len(fixed_vocab)


def qa_collate_fn(batch):
	"""
	Combines the individual examples from the dataset into a batch
	Args
		batch: a list of batch_size examples
		each example is a list of (output of the dataset)
			# 0: passage_vectors
			# 1: query_vector
			# 2: answer_vector
			# 3: relevance
			# 4: is_answerable
			# 5: passage_oov_tokens
			# 6: query_oov_tokens
			# 7: answer_oov_tokens
			# 8: answer_style
			# 9: sampled_answers (useful only for eval)
			# 10: query_id
			# 11: is_training
	Returns
		# 0: passage_vectors: 3d long tensor [bs x K x L]
			the indices of the passages (fixed vocab)
		# 1: query_vectors: 2d long tensor [bs x J]
			the indcies of the questions (fixed vocab)
		# 2: answer_src_vectors: 2d long tensor [bs x T - 1]
			the indices of the answers (fixed vocab)
			the style index is also appended in the beginning
			padding vector if the example is not answerable
			(only useful for training)
		# 3: relevancies: 2d float tensor [bs x K]
			the ground-truth relevancies of the passages
		# 4: is_answerable: 1d bool tensor [bs]
			the ground-truth is_answerable
		# 5: answer_trg_vectors: 2d long tensor [bs x T - 1]
			the indices of the answers shifted right (extended vocab)
			the eos token in appended at the end
			padding vector if the example is not answerable
			(only useful for training)
		# 6: passage_query_ext_vectors: 2d long tensor [bs x (K * L + J)]
			the indices of the concatenated passages and question (extended vocab)
			padding vector if the example is not answerable
		# 7: answer_styles: list[str]
			(only useful for training)
		# 8: batch_extended_token2id: dict[str: int]
			this is None if no OOV tokens exist in the passages or queries of the examples
			(only useful for eval)
		# 9: sampled_answers: list[list[str]]
			the ground-truth available answers for each example
			(only useful for eval)
		# 10: query_ids: list[str]
			the unique query identifier of each example
			(only useful for eval)
	"""

	bs = len(batch)

	answer_styles = [example[8] for example in batch]
	sampled_answers = [example[9] for example in batch]
	query_ids = [example[10] for example in batch]
	is_training = batch[0][11]

	# relevancies and answerabilities
	relevancies = torch.tensor([example[3] for example in batch], dtype = torch.float)
	is_answerable = torch.tensor([example[4] for example in batch], dtype = torch.bool)

	# whether each example contributes to the extended vocabulary
	is_extended = [any([bool(k_oov_tokens) for k_oov_tokens in example[5]]) or bool(example[6])
		for example in batch]

	# create and extended vocabulary for the batch only if at least one example (passages or query) has oov tokens
	if any(is_extended):
		batch_extended_token2id = fixed_token2id.copy()
		batch_new_tokens = set().union(*[list(k_oov_tokens.values())
			for example in batch for k_oov_tokens in example[5]])
		batch_new_tokens = batch_new_tokens.union(*[list(example[6].values())
			for example in batch])
		batch_extended_token2id.update({token: d_fixed_vocab + i
			for i, token in enumerate(batch_new_tokens)})
	else:
		batch_extended_token2id = None

	# repeat for each type of sequence (passage, query, answer)
	# concatenate example vectors into batch vectors
	# find maximum length among the batch
	# remove unnecessary padding form the tensors
	# initialize extended vocabulary vectors with the fixed vocabulary ones

	# passage
	passage_fixed_vectors = torch.cat([example[0].unsqueeze(0)
		for example in batch], dim = 0)
	max_L = (passage_fixed_vectors.sum(dim = (0, 1)) != 0).sum()
	passage_fixed_vectors = torch.narrow(passage_fixed_vectors, 2, 0, max_L)
	passage_ext_vectors = passage_fixed_vectors.clone()

	# query
	query_fixed_vectors = torch.cat([example[1].unsqueeze(0)
		for example in batch], dim = 0)
	max_J = (query_fixed_vectors.sum(dim = 0) != 0).sum()
	query_fixed_vectors = torch.narrow(query_fixed_vectors, 1, 0, max_J)
	query_ext_vectors = query_fixed_vectors.clone()

	# answer (only if in training mode)
	if is_training:
		answer_src_vectors = torch.cat([example[2].unsqueeze(0)
			for example in batch], dim = 0)
		max_T = (answer_src_vectors.sum(dim = 0) != 0).sum()
		answer_src_vectors = torch.narrow(answer_src_vectors, 1, 0, max_T)
		answer_trg_vectors = answer_src_vectors.clone()
	else:
		answer_src_vectors, answer_trg_vectors = None, None

	# if necessary adjust extended vectors with the enties from the extended vocabulary
	for i, example in enumerate(batch):

		if is_answerable[i] and is_extended[i]:

			for k, k_oov_tokens in enumerate(example[5]):
				for pos, token in k_oov_tokens.items():
					passage_ext_vectors[i, k, pos] = batch_extended_token2id[token]

			for pos, token in example[6].items():
				query_ext_vectors[i, pos] = batch_extended_token2id[token]

			if is_training:
				for pos, token in example[7].items():
					# only if the unknown token is part of the query or passage oov tokens
					if any([token in k_oov_tokens.values() for k_oov_tokens in example[5]]) \
						or token in example[6].values():
						answer_trg_vectors[i, pos] = batch_extended_token2id[token]

	# combine passages and questions in one tensor (easier handling in multi-source pointer-gen)
	passage_query_ext_vectors = torch.cat([passage_ext_vectors.view(bs, -1),
		query_ext_vectors], dim = -1)

	if is_training:
		eos_idx = fixed_token2id[my_constants.eos_token]
		pad_idx = fixed_token2id[my_constants.pad_token]

		# replace eos token in the source answer with padding
		answer_src_vectors[answer_src_vectors == eos_idx] = pad_idx

		# shift source and target by one position
		answer_src_vectors = answer_src_vectors[:, :-1]
		answer_trg_vectors = answer_trg_vectors[:, 1:]

	return passage_fixed_vectors, query_fixed_vectors, answer_src_vectors, relevancies, is_answerable, answer_trg_vectors, passage_query_ext_vectors, answer_styles, batch_extended_token2id, sampled_answers, query_ids