from torch.utils.data import DataLoader
import torch
import numpy as np
import os
import json
from multiprocessing import cpu_count
from tqdm import tqdm
import sys

os.chdir("question_answering")

sys.path.append("./../models")
sys.path.append("./../general_modules")

from qa_dataset import QaDataset
from qa_argument_parser_helper import parse_arguments
from qa_eval_helpers import get_architecture, decode_fn, init_eval_model
from qa_collate_fn import qa_collate_fn
from postprocess_decoded_seq import postprocess_decoded_seq
from my_tokenizer import construct_tokenizer

args = parse_arguments()

tokenizer = construct_tokenizer()


def eval(args):

	# number of cpu and gpu devices
	n_gpu = torch.cuda.device_count()
	n_cpu = cpu_count()
	print(f"Number of cuda devices: {n_gpu} | Number of CPU cores: {n_cpu}")

	device_list = [torch.device(f"cuda:{i}") for i in range(n_gpu)]
	main_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	# Load checkpoint
	checkpoint_path = os.path.join(args.checkpoint_path, args.run_name)
	specific_checkpoint = os.listdir(checkpoint_path)[-1] if args.run_subname == "" else args.run_subname
	specific_checkpoint_path = os.path.join(checkpoint_path, specific_checkpoint)
	checkpoint = torch.load(specific_checkpoint_path, map_location = main_device)
	print(f"Loaded checkpoint: {specific_checkpoint} of run: {args.run_name}")
	
	# Load architecture-specific arguments
	args = get_architecture(checkpoint["args"], args)

	# Specify evaluation path and create it if does not exist
	if args.eval_path == "":
		eval_path = os.path.join("evaluation", args.run_name, specific_checkpoint.split("_")[-1].split(".")[0],
							args.subset_name)
	else:
		eval_path = args.eval_path

	os.makedirs(eval_path, exist_ok = True)

	# initialize dataloder object
	dataloader = DataLoader(QaDataset(args, is_training = False),
							args.batch_size,
							collate_fn = qa_collate_fn,
							shuffle = False,
							num_workers = args.num_workers,
							drop_last = False)
	print(f"Initialized dataset with {len(dataloader.dataset)} examples")

	# initialize model, load checkpoint and set to eval mode
	model = init_eval_model(args, main_device, device_list, dataloader.dataset.fixed_token2id, checkpoint["model"])

	# to store results
	results = {	"lambdas": {"passage": [], "question": [], "vocabulary": []},
				"seq_len": [],
				"query_ids": []}

	style = args.available_styles
	assert style in ["qa", "nlg"]

	# evaluate dataset
	with torch.no_grad():
		for batch in tqdm(iter(dataloader), miniters = 1000):
			take_eval_step(batch, model, main_device, results, style,
							dataloader.dataset.fixed_token2id, eval_path)

	# store lambdas
	with open(os.path.join(eval_path, "avg_lambdas.json"), "w") as f:
		json.dump({"vocabulary": np.mean(results["lambdas"]["vocabulary"]),
					"question": np.mean(results["lambdas"]["question"]),
					"passage": np.mean(results["lambdas"]["passage"])}, f)


def take_eval_step(batch, model, device, results, style, fixed_token2id, eval_path):

	# Indices in the fixed vocabulary for passages and questions
	passage_fixed_vectors = batch[0].to(device)  # Passages (3d long tensor) [batch_size x num_passages x seq_len_passage]
	query_fixed_vectors = batch[1].to(device)  # Questions (2d long tensor) [batch_size x seq_len_question]

	# Indices in the extended vocabulary for the combined passages and questions
	passage_query_ext_vectors = batch[6].to(device)  # (2d long tensor) [batch_size x num_passages x seq_len_passage + seq_len_question]

	# Get extended vocab if there was at least one OOV token in the passages and questions
	# otherwise it is the same as the fixed one
	if batch[8] is not None:
		batch_ext_token2id, is_extended = batch[8], True
	else:
		batch_ext_token2id, is_extended = fixed_token2id, False

	# unique example identifiers
	query_ids = batch[10]

	# Example answerabilites, (1d bool tensor) [batch_size]
	is_answerable = batch[4].to(device)

	# target answers: list[list[str]]
	# can be multiple for each example
	trg_answers = batch[9]

	# the maximum index of the source text
	maximum_vocab_id = passage_query_ext_vectors.max().item() + 1

	batch = None

	# autoregressive forward through the model
	predicted_sequences, answer_possibilities, avg_lambdas, betas, lengths = model(passage_fixed_vectors,
																query_fixed_vectors,
																passage_query_ext_vectors,
																is_answerable,
																maximum_vocab_id = maximum_vocab_id,
																autoregressive = True,
																style = style)

	# add the extra tokens to the tokenizer
	if is_extended:
		extra_tokens = list(batch_ext_token2id.keys())[len(fixed_token2id):]
		tokenizer.add_tokens(extra_tokens)

	# decode indices to strings
	generated_answers = decode_fn(predicted_sequences, tokenizer)
	generated_answers = postprocess_decoded_seq(generated_answers)

	# bring tokenizer back to its original vocab
	if is_extended:
		tokenizer.remove_tokens(extra_tokens)

	# number of tokens in every prediction
	pred_seq_len = lengths.tolist()
	results["seq_len"].extend(pred_seq_len)

	results["query_ids"].extend(query_ids)

	# store lambdas
	results["lambdas"]["vocabulary"].extend(avg_lambdas[:, 0].tolist())
	results["lambdas"]["question"].extend(avg_lambdas[:, 1].tolist())
	results["lambdas"]["passage"].extend(avg_lambdas[:, 2].tolist())

	with open(os.path.join(eval_path, f"lambdas_{style}.json"), "a") as f:
		for i, q_id in enumerate(query_ids):
			json.dump({"lambdas": {"vocabulary": float(avg_lambdas[i, 0]),
									"question": float(avg_lambdas[i, 1]),
									"passage": float(avg_lambdas[i, 2])},
						"query_id": int(q_id)}, f)
			f.write("\n")

	with open(os.path.join(eval_path, f"reference_{style}.json"), "a") as f:
			for trg_a, q_id in zip(trg_answers, query_ids):
				json.dump({"answers": trg_a, "query_id": int(q_id)}, f)
				f.write("\n")

	with open(os.path.join(eval_path, f"predictions_{style}.json"), "a") as f:
			for gen_a, q_id in zip(generated_answers, query_ids):
				json.dump({"answers": [gen_a], "query_id": int(q_id)}, f)
				f.write("\n")

	with open(os.path.join(eval_path, f"sequence_length_{style}.json"), "a") as f:
			for s_len, q_id in zip(pred_seq_len, query_ids):
				json.dump({"answers": s_len, "query_id": int(q_id)}, f)
				f.write("\n")


if __name__ == '__main__':
	eval(args)