import torch
import os
from tqdm import tqdm

from my_tokenizer import construct_tokenizer


def create_embeddings(glove_path):
	"""
	Creates an embedding matrix for all the tokens in the vocabulary and saves them in a .pt file

	Load 300d cached glove embeddings and
	loop through the vocabulary tokens from the tokenizer to init them with
		(1) zeros if the token is the padding token
		(2) the glove vector if the token is part of glove
		(3) the glove vector plus some guassian noise if the sub-token is part of glove
		(4) random normal vector if completelly unknown

	Args:
		glove_path: str
			The path for the 300d glove emebeddings
			Download from: https://nlp.stanford.edu/projects/glove/

	"""

	glove_vocab, glove_token2id, glove_vectors, d_emb = torch.load(glove_path)
	glove_vocab = set(glove_vocab)

	tokenizer = construct_tokenizer()

	not_found = []

	new_vectors = torch.zeros((len(tokenizer.vocab), d_emb), dtype = torch.float)

	for idx in tqdm(tokenizer.vocab.values()):

		token = tokenizer.ids_to_tokens[idx]

		if (token == tokenizer.pad_token):
			vector = torch.zeros((1, d_emb), dtype = torch.float)

		elif token in glove_vocab:
			vector = glove_vectors[glove_token2id[token]].unsqueeze(0)

		elif "##" in token:
			reduced_token = token[2:]
			if reduced_token in glove_vocab:
				# plus some gaussian noise
				vector = glove_vectors[glove_token2id[reduced_token]].unsqueeze(0) + torch.normal(0, 0.005, size = (1, d_emb))
			else:
				not_found.append(token)
				vector = torch.normal(0, 0.01, size = (1, d_emb))
		else:
			not_found.append(token)
			vector = torch.normal(0, 0.01, size = (1, d_emb))

		new_vectors[int(idx)] = vector

	print(f"{len(not_found)} tokens and subtokens were not found in pre-trained glove")
	
	embeddings_path = os.path.join(os.path.dirname(glove_path), "embeddings.pt")
	torch.save(new_vectors, embeddings_path)

	print(f"Saved embeddings in {embeddings_path}")
