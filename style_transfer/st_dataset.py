import json
import numpy as np
from torch.utils import data
import torch
import torch.nn.functional as F
from random import choice
from rouge import Rouge
import os

import my_constants
from preprocess_ms_marco import preprocess_ms_marco
from my_tokenizer import construct_tokenizer


class StDataset(data.Dataset):
	"""
	Loads data and contains the necessary functions to sample and prepare an example for the dataloader
	"""
	
	def __init__(self, args):
		"""
		Args:
			args: argument parses object
		"""
		super(StDataset, self).__init__()

		self.mode = args.mode

		self.dataset_name = args.dataset_name

		self.pad_token = my_constants.pad_token
		self.unk_token = my_constants.unk_token
		self.bos_token = my_constants.nlg_token
		self.eos_token = my_constants.eos_token

		self.tokenizer = construct_tokenizer()
		self.fixed_token2id = self.tokenizer.vocab
		self.d_vocab = len(self.fixed_token2id)

		self.pad_idx = self.fixed_token2id[self.pad_token]
		self.unk_idx = self.fixed_token2id[self.unk_token]
		self.bos_idx = self.fixed_token2id[self.bos_token]
		self.eos_idx = self.fixed_token2id[self.eos_token]

		self.max_L = args.max_seq_len_passage
		self.max_J = args.max_seq_len_question
		self.max_N = args.max_seq_len_qa_answer
		self.max_T = args.max_seq_len_nlg_answer

		# (optionally cleans and) loads the data
		self._load_data()

		# filters data according the active mode and the max_data_size percentage
		self.max_data_size = args.max_data_size
		self._filter_data()

		self.indices = list(self.data["query_id"].keys())

		self.rouge = Rouge()


	def __len__(self):
		return len(self.data["query_id"])


	def __getitem__(self, i):
		
		# index of the example in the data: str
		idx = self.indices[i]

		# unique query identifier
		query_id = self.data["query_id"][idx]

		# query: str
		query = self.data["query"][idx]

		if self.mode in ["train", "eval"]:
			# randomly sample the (target) nlg answer
			nlg_answers = self.data["wellFormedAnswers"][idx]
		else:
			nlg_answers = None

		if self.mode == "train":
			sampled_nlg_answer = choice(nlg_answers)

		# candidate qa answers
		qa_answers = self.data["answers"][idx]

		if self.mode == "train":
			# sample the qa answer that is most similar to the nlg answer
			sampled_qa_answer = self._sample_most_similar(sampled_nlg_answer, qa_answers)
		else:
			sampled_qa_answer = choice(qa_answers)

		# sample the relevant passage that is most similar to the qa answer
		rel_passages = [p_info["passage_text"] for p_info in self.data["passages"][idx]
						if p_info["is_selected"]]
		sampled_rel_passage = self._sample_most_similar(sampled_qa_answer, rel_passages)

		# map the text to vectors and (optionally) get the oov tokens
		passage_vector, passage_oov_tokens = self._process_text(sampled_rel_passage, self.max_L)
		query_vector, query_oov_tokens = self._process_text(query, self.max_J)
		qa_answer_vector, qa_answer_oov_tokens = self._process_text(sampled_qa_answer, self.max_N)

		if self.mode == "train":
			nlg_answer_vector, nlg_answer_oov_tokens = self._process_text(sampled_nlg_answer, self.max_T,
																		bos_token = self.bos_token,
																		eos_token = self.eos_token)
		else:
			nlg_answer_vector, nlg_answer_oov_tokens = None, None

		return passage_vector, query_vector, qa_answer_vector, nlg_answer_vector, \
			passage_oov_tokens, query_oov_tokens, qa_answer_oov_tokens, nlg_answer_oov_tokens, \
			query_id, nlg_answers, self.mode


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
		Removes examples according to the active mode
		Keeps a percentage of the examples if max_data_size != 100 (useful for debugging)
		"""

		for idx in list(self.data["answers"].keys()):

			qa_avail = (self.data["answers"][idx] != ['No Answer Present.']) and (self.data["answers"][idx] != [""])
			nlg_avail = self.data["wellFormedAnswers"][idx] != "[]"

			# keeps the examples that are answerable and have an NLG answer
			if self.mode in ["train", "eval"] and not nlg_avail:
				for k in self.data.keys():
					del self.data[k][idx]

			# keeps the examples that are answerable and do not have an NLG answer
			elif self.mode == "infer" and (nlg_avail or not qa_avail):
				for k in self.data.keys():
					del self.data[k][idx]

		if self.max_data_size != 100:
			avail_data_size = len(self.data["passages"])

			indices = list(self.data["query_id"].keys())

			shuffled = np.random.permutation(avail_data_size)
			to_keep = shuffled[:int(self.max_data_size / 100 * avail_data_size)]

			for k in self.data.keys():
				self.data[k] = {indices[i]: self.data[k][indices[i]] for i in to_keep}


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


	def _sample_most_similar(self, reference, candidates):

		if len(candidates) > 1:
			scores = [self.rouge.get_scores(reference, cand)[0]["rouge-l"]["f"]
						for cand in candidates]
			sampled_candidate = candidates[np.argmax(scores)]
		else:
			sampled_candidate = candidates[0]

		return sampled_candidate
