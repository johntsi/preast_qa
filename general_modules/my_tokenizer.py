from transformers import BertTokenizer
import types
from collections import OrderedDict

import my_constants


def _unk_tokenize(self, text):
	"""
	Tokenizes a string and indicates which positions were not found in the vocabulary (UNK)

	Args:
		text: str.
	Returns:
		split_tokens: the tokenized text (list of str)
		unk_indices: the positions of the split_tokens in which there is an UNK

	"""
	split_tokens = []
	unk_indices = []
	if self.do_basic_tokenize:
		for token in self.basic_tokenizer.tokenize(text, never_split = self.all_special_tokens):
			for sub_token in self.wordpiece_tokenizer.tokenize(token):
				if sub_token == my_constants.unk_token:
					split_tokens.append(token)
					unk_indices.append(len(split_tokens) - 1)
					break
				split_tokens.append(sub_token)
	else:
		split_tokens = self.wordpiece_tokenizer.tokenize(text)
	return split_tokens, unk_indices


def remove_tokens(self, tokens, lower = True):
	"""Removes tokens from the vocabularies.

	Args:
		tokens: Tokens to be removed (list of str)
		lower: whether to lowercase (bool)

	"""
	if lower:
		tokens = [t.lower() for t in tokens]
	ids = [self.added_tokens_encoder[t] for t in tokens]
	for i, t in zip(ids, tokens):
		del self.added_tokens_encoder[t]
		self.unique_added_tokens_encoder.remove(t)
		del self.added_tokens_decoder[i]


def construct_tokenizer(uncased = True):
	"""
	Creates a modified version of the BertTokenizer from the Transformers library
	
	Loads the pre-trained tokenizer
	adds special tokens (CLS, EOS, NLG, QA, UNK, PAD)
	removes "unused" tokens, MASK and SEP

	Args:
		uncased: version of the tokenizer

	Returns:
		bert_tokenizer: The modified tokenizer object with new functions (remove_tokens, _unk_tokenize)

	"""
	
	if uncased:
		name = "bert-base-uncased"
	else:
		name = "bert-base-cased"
		
	bert_tokenizer = BertTokenizer.from_pretrained(name, do_lower_case = uncased,
										cls_token = my_constants.cls_token,
										eos_token = my_constants.eos_token,
										pad_token = my_constants.pad_token,
										unk_token = my_constants.unk_token,
										additional_special_tokens = [my_constants.qa_token, my_constants.nlg_token])

	bert_tokenizer.add_special_tokens({"cls_token": my_constants.cls_token})
	bert_tokenizer.add_special_tokens({"eos_token": my_constants.eos_token})
	bert_tokenizer.add_special_tokens({"pad_token": my_constants.pad_token})
	bert_tokenizer.add_special_tokens({"unk_token": my_constants.unk_token})
	bert_tokenizer.add_special_tokens({"additional_special_tokens": [my_constants.qa_token, my_constants.nlg_token]})

	bert_tokenizer._unk_tokenize = types.MethodType(_unk_tokenize, bert_tokenizer)
	bert_tokenizer.remove_tokens = types.MethodType(remove_tokens, bert_tokenizer)

	bert_tokenizer.remove_tokens([my_constants.eos_token, my_constants.qa_token, my_constants.nlg_token], lower = False)

	bert_tokenizer.unique_added_tokens_encoder = {}

	new_vocab = OrderedDict()
	new_ids_to_tokens = OrderedDict()

	special_tokens = [my_constants.pad_token, my_constants.cls_token, my_constants.unk_token, my_constants.eos_token,
						my_constants.qa_token, my_constants.nlg_token]

	for Id, token in enumerate(special_tokens):
		new_vocab[token] = Id
		new_ids_to_tokens[Id] = token

	for token in bert_tokenizer.vocab.keys():
		if "[unused" in token:
			continue
		if token == "[MASK]" or token == "[SEP]":
			continue
		if token in special_tokens:
			continue
		
		Id = len(new_vocab)
		new_vocab[token] = Id
		new_ids_to_tokens[Id] = token

	bert_tokenizer.vocab = new_vocab
	bert_tokenizer.ids_to_tokens = new_ids_to_tokens
	bert_tokenizer.wordpiece_tokenizer.vocab = new_vocab
	
	return bert_tokenizer