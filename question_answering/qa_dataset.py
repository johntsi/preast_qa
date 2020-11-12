import json
import numpy as np
from torch.utils import data
from random import random, choice
import torch
import torch.nn.functional as F
import os

from my_tokenizer import construct_tokenizer
from preprocess_ms_marco import preprocess_ms_marco
import my_constants


class QaDataset(data.Dataset):
	"""
	Dataset object for the MS MARCO data (train or dev)
	"""

	def __init__(self, args, is_training = True):
		super(QaDataset, self).__init__()

		# mode
		self.is_training = is_training

		# dataset name
		self.dataset_name = args.dataset_name

		# tokenizer object
		self.tokenizer = construct_tokenizer()

		# set of known tokens
		self.fixed_vocab = set(self.tokenizer.vocab.keys())

		# size of fixed vocab
		self.d_vocab = len(self.fixed_vocab)

		# mappings from token to ids and from ids to tokens
		self.fixed_token2id = self.tokenizer.vocab
		self.fixed_id2token = self.tokenizer.ids_to_tokens

		# special tokens
		self.pad_token = my_constants.pad_token
		self.eos_token = my_constants.eos_token
		self.unk_token = my_constants.unk_token
		self.cls_token = my_constants.cls_token
		self.qa_token = my_constants.qa_token
		self.nlg_token = my_constants.nlg_token

		# indices of special tokens in the fixed vocabula
		self.pad_idx = self.fixed_token2id[my_constants.pad_token]
		self.unk_idx = self.fixed_token2id[my_constants.unk_token]
		self.cls_idx = self.fixed_token2id[my_constants.cls_token]
		self.eos_idx = self.fixed_token2id[my_constants.eos_token]
		self.qa_idx = self.fixed_token2id[my_constants.qa_token]
		self.nlg_idx = self.fixed_token2id[my_constants.nlg_token]

		# styles
		self.qa_style = "qa"
		self.nlg_style = "nlg"

		# available styles (both, nlg or qa)
		if args.available_styles == "both":
			self.available_styles = [self.qa_style, self.nlg_style]
		else:
			self.available_styles = [args.available_styles]

		# ALL, ANS or NLG
		self.active_subset = args.subset_name

		# max sequence lengths for passages (L), questions (J) and answers (T)
		# max number of passages (K)
		self.max_L = args.seq_len_passage
		self.max_J = args.seq_len_question
		self.max_T = args.seq_len_answer
		self.max_K = args.max_num_passages

		self.shuffle_passage_order = True

		# (optionally cleans and) loads the data
		self._load_data()

		# optionally removes some datapoints
		self.max_data_size = args.max_data_size
		self._filter_data()

		self.indices = list(self.data["query_id"].keys())

		# optionally add to the dataset the generated nlg answers from a style-transfer model
		self._add_generated_nlg_answers(args.use_generated_nlg_answers, args.generated_nlg_answers_path)


	def __len__(self):
		return len(self.data["query_id"])


	def __getitem__(self, i):
		"""
		Args:
			i: int
		Returns:
			passage_vectors: 2d long tensor [K x L]
				The indices of the passages in the fixed vocabulary
			query_vector: 1d long tensor [J]
				The indices of the query in the fixed vocabulary
			answer_vector: 1d long tensor [T]
				The indices of the answer in the fixed vocabulary
				Padding vector if the query is not is_answerable
				None if not in training mode
			rel: 1d bool array [K]
				The ground-truth relevance labels for each of the passages
			is_answerable: bool
				The ground-truth answerability label for the example
			passage_oov_tokens: list[dict[int: str]]
				The out-of-vocabulary tokens for each of the passages
				The k-th dictionary maps the l-th position in the k-th passage to an unknown token
					Empty if no oov tokens
			query_oov_tokens: dict[int: str]
				The out-of-vocabulary tokens for the query
				Empty if no oov tokens
			answer_oov_tokens: dict[int: str]
				The out-of-vocabulary tokens for the answer
				Empty if no oov tokens or not answerable or not training mode
			answer_style: str
				The sampled style of the answer (nlg or qa)
				None if not answerable
			sampled_answers: list[str]
				The available answers for the sampled style for this example
				Empty if not answerable
			query_id: str
				The unique identifier for this example
			is_training: bool
				Whether the dataset is in training mode
		"""

		# index of the example in the data: str
		idx = self.indices[i]

		# unique query identifier
		query_id = self.data["query_id"][idx]

		# query: str
		query = self.data["query"][idx]

		# query_vector: 1d long tensor [max_J]
		# query_oov_tokens: dict[pos: token]
		query_vector, query_oov_tokens = self._process_text(query, self.max_J)

		# sample K passages from the example
		# rel: 1d array[bool]
		# passages: list[str]
		rel, passages = self._sample_passages(idx)

		# passage_vectors: 2d long tensor [max_K x max_L]
		# passage_oov_tokens: list[dict[pos: token]]
		passage_vectors, passage_oov_tokens = self._process_mult_text(passages, self.max_L, bos_token = self.cls_token)

		# the answerability of the example: bool
		is_answerable = (rel > 0).any()

		if is_answerable:
			# sample a style and an answer from the example
			# answer_style: str
			# sampled_answers: list[str]
			answer_style, sampled_answers = self._sample_answers(idx)

			if self.is_training:
				answer_text = choice(sampled_answers)

				# answer_vector: 1d long tensor [max_T]
				# answer_oov_tokens: dict[pos: token]
				answer_vector, answer_oov_tokens = self._process_text(answer_text, self.max_T,
					bos_token = self.qa_token if answer_style == self.qa_style else self.nlg_token,
					eos_token = self.eos_token)

			else:
				answer_vector, answer_oov_tokens = None, {}

		else:
			answer_vector = torch.zeros(self.max_T, dtype = torch.long)
			answer_oov_tokens, answer_style, sampled_answers = {}, None, []

		return passage_vectors, query_vector, answer_vector, rel, is_answerable, passage_oov_tokens, query_oov_tokens, answer_oov_tokens, answer_style, sampled_answers, query_id, self.is_training


	def _load_data(self):
		"""
		Processes ms marco and saves a clean version of the data
		Loads the data from the clean file
		"""

		data_path = "./../data/"
		dataset_path = os.path.join(data_path, self.dataset_name + "_v2.1.json")
		cleaned_dataset_path = os.path.join(data_path, self.dataset_name + "_v2.1_cleaned.json")

		if not os.path.isfile(cleaned_dataset_path):
			preprocess_ms_marco(dataset_path)

		with open(cleaned_dataset_path, "r") as read_file:
			self.data = json.load(read_file)


	def _filter_data(self):
		"""
		Removes examples according to active subset
		Keeps a percentage of the examples if max_data_size != 100 (useful for debugging)
		"""

		if self.active_subset == "ANS":
			for idx in list(self.data["answers"].keys()):
				qa_avail = (self.data["answers"][idx] != ['No Answer Present.']) and (self.data["answers"][idx] != [""])
				if not qa_avail:
					for k in self.data.keys():
						del self.data[k][idx]

		elif self.active_subset == "NLG":
			for idx in list(self.data["answers"].keys()):
				nlg_avail = self.data["wellFormedAnswers"][idx] != "[]"
				if not nlg_avail:
					for k in self.data.keys():
						del self.data[k][idx]

		if self.max_data_size != 100:
			avail_data_size = len(self.data["passages"])

			indices = list(self.data["query_id"].keys())

			shuffled = np.random.permutation(avail_data_size)
			to_keep = shuffled[:int(self.max_data_size / 100 * avail_data_size)]

			for k in self.data.keys():
				self.data[k] = {indices[i]: self.data[k][indices[i]] for i in to_keep}


	def _add_generated_nlg_answers(self, use_generated_nlg_answers, generated_nlg_answers_path):
		"""
		Args:
			use_generated_nlg_answers: bool
			generated_nlg_answers_path: str
		"""

		if self.is_training and use_generated_nlg_answers:

			# query_id --> idx
			id_mapping = {q_id: idx for idx, q_id in self.data["query_id"].items()}

			# load generate nlg answers
			# dict[int: list[str]]
			with open(generated_nlg_answers_path, "r") as f:
				generated_nlg_answers = {id_mapping[str(json.loads(line)["query_id"])]: json.loads(line)["answers"]
											for line in f}

			# add them to the dataset
			# empty list if not available for an example
			self.data["generated_nlg_answers"] = {idx: generated_nlg_answers.get(idx, [])
													for idx in self.indices}

		else:
			self.data["generated_nlg_answers"] = {idx: [] for idx in self.indices}


	def _sample_passages(self, idx):
		"""
		Args:
			idx: str
		Returns:
			rel: 1d bool array [K]
			passages: list[str]

		"""
		
		rel = np.array([p_info["is_selected"] for p_info in self.data["passages"][idx]], dtype = bool)
		K_i = len(rel)

		# if the available passages are more than max_K
		# sample all the relevant ones and randomly sample the rest
		if K_i > self.max_K:
			all_idx = np.arange(K_i)

			relevant_idx = all_idx[rel == 1]
			non_relevant_idx = all_idx[rel == 0]
			n_rel = len(relevant_idx)

			sampled_idx = np.random.choice(non_relevant_idx, size = self.max_K - n_rel, replace = False)
			selected_idx = np.sort(np.append(relevant_idx, sampled_idx))
			
			rel = rel[selected_idx]

		else:
			selected_idx = np.arange(K_i)

		# get the selected passages
		passages = [p_info["passage_text"] for i, p_info in enumerate(self.data["passages"][idx])
											if i in selected_idx]

		# if the available passages are less than max_K
		# add empty strings to the passage list
		# and zeros to the relevance array
		if K_i < self.max_K:
			pad_size = self.max_K - K_i
			passages.extend(["" for i in range(pad_size)])
			rel = np.concatenate([rel, np.zeros(pad_size)])

		# optionally shuffle the order of the passages to remove any positional bias
		if self.shuffle_passage_order:
			shuffled_idx = np.random.permutation(np.arange(len(rel)))
			rel = rel[shuffled_idx]
			passages = [passages[i] for i in shuffled_idx]

		return rel, passages


	def _sample_answers(self, idx):
		"""
		Args:
			idx: str
		Returns:
			sampled_style: str
			sampled_answers: list[str]
		"""

		qa_answers = self.data["answers"][idx]
		nlg_answers = self.data["wellFormedAnswers"][idx]
		gen_nlg_answers = self.data["generated_nlg_answers"][idx]

		nlg_avail = nlg_answers != "[]"
		gen_nlg_avail = bool(gen_nlg_answers)

		if len(self.available_styles) == 2:
			# sample the nlg answers if available
			if nlg_avail:
				sampled_style = self.nlg_style
				sampled_answers = nlg_answers

			# coin toss if generated nlg answers are available
			elif gen_nlg_avail:
				if random() > 0.5:
					sampled_style = self.qa_style
					sampled_answers = qa_answers
				else:
					sampled_style = self.nlg_style
					sampled_answers = gen_nlg_answers

			# sample the qa answers
			else:
				sampled_style = self.qa_style
				sampled_answers = qa_answers

		else:
			sampled_style = self.available_styles[0]
			if sampled_style == self.qa_style:
				sampled_answers = qa_answers
			else:
				sampled_answers = nlg_answers

		return sampled_style, sampled_answers


	def _process_text(self, text, max_tokens, bos_token = None, eos_token = None):
		"""
		Args:
			text: str
			max_tokens: int
			bos_token: str
			eos_token: str
		Returns:
			vector: 1d long tensor [max_tokens]
			oov_tokens: dict[pos: str]
		"""

		# tokenizer text and optionally get the positions of the oov tokens
		tokens, unk_positions = self.tokenizer._unk_tokenize(text)

		# truncate to adjusted maximum tokens
		tokens = tokens[:max_tokens - (bos_token is not None) - (eos_token is not None)]

		# get oov tokens and replace them with UNK in the tokens
		oov_tokens = {}
		if unk_positions:
			for pos in unk_positions:
				if pos >= len(tokens):
					break
				oov_tokens.update({pos: tokens[pos]})
				tokens[pos] = self.unk_token

		if bos_token is not None:
			tokens.insert(0, bos_token)

		if eos_token is not None:
			tokens.append(eos_token)

		# indices of the tokens in the fixed vocabulary
		vector = torch.tensor(self.tokenizer.encode(tokens, add_special_tokens = False), dtype = torch.long)

		# pad to max length
		diff = max_tokens - len(vector)
		if diff > 0:
			vector = F.pad(vector, (0, diff), "constant", 0)

		return vector, oov_tokens


	def _process_mult_text(self, texts, max_tokens, bos_token = None, eos_token = None):
		"""
		Iterativelly calls _process_text()
		Used for passages

		Args:
			text: str
			max_tokens: int
			bos_token: str
			eos_token: str
		Returns:
			vectors: 2d long tensor [max_K, max_tokens]
			oov_tokens: list[dict[pos: str]]
		"""

		# init
		vectors = torch.zeros(size = [self.max_K, max_tokens], dtype = torch.long)
		oov_tokens = []

		# fill by iterating over the K texts
		for i, text in enumerate(texts):
			vector_i, oov_tokens_i = self._process_text(text, max_tokens, bos_token, eos_token)
			vectors[i, :len(vector_i)] = vector_i
			oov_tokens.append(oov_tokens_i)

		return vectors, oov_tokens