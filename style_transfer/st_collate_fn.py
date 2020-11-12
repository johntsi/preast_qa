import torch

import my_constants
from my_tokenizer import construct_tokenizer

print("Loading fixed vocab in st collate function ...")
fixed_token2id = construct_tokenizer().vocab
fixed_vocab = set(fixed_token2id)
d_fixed_vocab = len(fixed_vocab)


def combine(batch, i):
	"""
	Combines the vectors for a specific type of sequence
	and removes extra padding

	Args:
		batch: list
		i: int
			indicates the type of sequence
			(4, 5, 6) --> (passage, query, qa_answer)
	Returns:
		vectors: 2d long tensor [bs x max_len]

	"""

	vectors = torch.cat([example[i].unsqueeze(0)
		for example in batch], dim = 0)
	max_len = (vectors.sum(dim = 0) != 0).sum()
	vectors = torch.narrow(vectors, 1, 0, max_len)

	return vectors


def st_collate_fn(batch):
	"""
	Args
		batch: list
		each example is a list of (output of the dataset)
			# 0: passage_vector
			# 1: query_vector
			# 2: qa_answer_vector
			# 3: nlg_answer_vector
			# 4: passage_oov_tokens
			# 5: query_oov_tokens
			# 6: qa_answer_oov_tokens
			# 7: nlg_answer_oov_tokens
			# 8: query_ids
			# 9: nlg_answers
			# 10: mode

	Returns
		# 0: passage_fixed_vectors: 2d torch long [bs x L]
			the indices of the passages (fixed vocab)
		# 1: query_fixed_vectors: 2d torch long [bs x J]
			the indcies of the questions (fixed vocab)
		# 2: qa_answer_fixed_vectors: 2d torch long [bs x N]
			the indcies of the qa_answers (fixed vocab)
		# 3: nlg_answer_src_vectors: 2d torch long [bs x T - 1]
			the indices of the nlg answers (fixed vocab)
			the bos index is appended in the beginning
			(only useful for training)
		# 4: nlg_answer_trg_vectors: 2d torch long [bs x T - 1]
			the indices of the nlg answers shifted right (extended vocab)
			the eos token is appended at the end
			(only useful for training)
		# 5: source_ext_vectors: 2d torch long [bs x (L + J + N)]
			the combined representations of the three source sequences
			(passage, query, qa_answer) (extended vocab)
		# 6: batch_extended_token2id: dict[str: int]
			this is None if no OOV tokens exist in the source sequences
			(only useful for eval)
		# 7: nlg_answers: list[list[str]]
			the ground-truth available nlg answers for each example
			(only useful for eval)
		# 8: query_ids: list[str]
			the unique query identifier of each example
			(only useful for eval)
	"""

	mode = batch[0][10]

	query_ids = [example[8] for example in batch]
	nlg_answers = [example[9] for example in batch]

	# whether each example contributes to the extended vocabulary
	is_extended = [any([bool(example[i]) for i in [4, 5, 6]])
					for example in batch]

	# create and extended vocabulary for the batch only if at least one example
	# (passages or query, or qa_answers) has oov tokens
	if any(is_extended):
		batch_extended_token2id = fixed_token2id.copy()
		batch_new_tokens = set().union(*[list(example[i].values())
			for example in batch for i in [4, 5, 6]])
		batch_extended_token2id.update({token: d_fixed_vocab + i
			for i, token in enumerate(batch_new_tokens)})
	else:
		batch_extended_token2id = None

	passage_fixed_vectors = combine(batch, 0)
	passage_ext_vectors = passage_fixed_vectors.clone()

	query_fixed_vectors = combine(batch, 1)
	query_ext_vectors = query_fixed_vectors.clone()

	qa_answer_fixed_vectors = combine(batch, 2)
	qa_answer_ext_vectors = qa_answer_fixed_vectors.clone()

	if mode == "train":
		nlg_answer_src_vectors = combine(batch, 3)
		nlg_answer_trg_vectors = nlg_answer_src_vectors.clone()
	else:
		nlg_answer_src_vectors = None
		nlg_answer_trg_vectors = None

	# if necessary adjust extended vectors with the enties from the extended vocabulary
	for i, example in enumerate(batch):

		if is_extended[i]:

			for pos, token in example[4].items():
				passage_ext_vectors[i, pos] = batch_extended_token2id[token]

			for pos, token in example[5].items():
				query_ext_vectors[i, pos] = batch_extended_token2id[token]

			for pos, token in example[6].items():
				qa_answer_ext_vectors[i, pos] = batch_extended_token2id[token]

			if mode == "train":
				for pos, token in example[7].items():
					# only if the unknown token is part of the query or passage oov tokens
					if any([token in example[i].values() for i in [4, 5, 6]]):
						nlg_answer_trg_vectors[i, pos] = batch_extended_token2id[token]

	# combine the extended representatons in one tensor
	source_ext_vectors = torch.cat([passage_ext_vectors, query_ext_vectors,
									qa_answer_ext_vectors], dim = -1)

	if mode == "train":
		eos_idx = fixed_token2id[my_constants.eos_token]
		pad_idx = fixed_token2id[my_constants.pad_token]

		# replace eos token in the nlg source answer with padding
		nlg_answer_src_vectors[nlg_answer_src_vectors == eos_idx] = pad_idx

		# shift source and target by one position
		nlg_answer_src_vectors = nlg_answer_src_vectors[:, :-1]
		nlg_answer_trg_vectors = nlg_answer_trg_vectors[:, 1:]

	return passage_fixed_vectors, query_fixed_vectors, qa_answer_fixed_vectors, \
		nlg_answer_src_vectors, nlg_answer_trg_vectors, source_ext_vectors, \
		batch_extended_token2id, nlg_answers, query_ids