from torch.utils.data import DataLoader
import torch
import os
from multiprocessing import cpu_count
import json
from tqdm import tqdm
import sys

os.chdir("style_transfer")

sys.path.append("./../models")
sys.path.append("./../general_modules")

from st_dataset import StDataset
from st_argument_parser_helper import parse_arguments
from st_collate_fn import st_collate_fn
from st_eval_helpers import get_architecture, decode_fn, init_eval_model
from postprocess_decoded_seq import postprocess_decoded_seq
from my_tokenizer import construct_tokenizer

tokenizer = construct_tokenizer()


def eval(args):

	assert args.mode in ["eval", "infer"]

	# number of cpu and gpu devices
	n_gpu = torch.cuda.device_count()
	n_cpu = cpu_count()
	print(f"Number of cuda devices: {n_gpu} | Number of CPU cores: {n_cpu}")

	# specify main device and all devices (if gpu available)
	device_list = [torch.device(f"cuda:{i}") for i in range(n_gpu)]
	main_device = device_list[0] if n_gpu > 0 else torch.device("cpu")
	print(f"Main device: {main_device}")
	print(f"Parallel devices = {device_list}")

	checkpoint_path = os.path.join(args.checkpoint_path, args.run_name)
	specific_checkpoint = os.listdir(checkpoint_path)[-1] if args.run_subname == "" else args.run_subname
	specific_checkpoint_path = os.path.join(checkpoint_path, specific_checkpoint)
	checkpoint = torch.load(specific_checkpoint_path, map_location = main_device)
	print(f"Loaded checkpoint: {specific_checkpoint} of run: {args.run_name}")
	
	args = get_architecture(checkpoint["args"], args)

	if args.eval_path == "":
		eval_path = os.path.join("evaluation", args.run_name, specific_checkpoint.split("_")[-1].split(".")[0])
	else:
		eval_path = args.eval_path

	os.makedirs(eval_path, exist_ok = True)

	dataloader = DataLoader(StDataset(args),
							args.batch_size,
							collate_fn = st_collate_fn,
							shuffle = False,
							num_workers = args.num_workers,
							drop_last = False)
	print(f"Initialized dataset with {len(dataloader.dataset)} examples")

	model = init_eval_model(args, dataloader.dataset.fixed_token2id, main_device, device_list, checkpoint["model"])

	with torch.no_grad():
		for i, batch in tqdm(enumerate(iter(dataloader)), miniters = 1000):
			take_eval_step(batch, model, main_device, eval_path, args.mode)


def take_eval_step(batch, model, main_device, eval_path, mode):

	# Representations of the sequences in the fixed vocabulary (indices)
	passage_fixed_vectors = batch[0].to(main_device)  # 2d long tensor [batch_size x seq_len_passage]
	query_fixed_vectors = batch[1].to(main_device)  # (2d long tensor [batch_size x seq_len_question]
	qa_answer_fixed_vectors = batch[2].to(main_device)  # 2d long tensor [batch_size x seq_len_qa]

	# Representation of the concatination of passage, question and qa_answer in the extended vocabulary
	source_ext_vectors = batch[5].to(main_device)  # (2d long tensor) [batch_size x seq_len_passage + seq_len_question + seq_len_answer]

	if batch[6] is not None:
		batch_ext_token2id, is_extended = batch[6], True
	else:
		batch_ext_token2id, is_extended = tokenizer.vocab, False

	# the target nlg answers for each example (empty if in inference mode)
	trg_nlg_answers = batch[7]

	query_ids = batch[8]

	d_ext_vocab = source_ext_vectors.max().item() + 1

	del batch

	# forward pass
	preds, avg_lambdas, lengths = model(passage_fixed_vectors, query_fixed_vectors, qa_answer_fixed_vectors,
		source_ext_vectors, d_ext_vocab, autoregressive = True)

	# optionally add the extra tokens in the tokenizer
	if is_extended:
		extra_tokens = list(batch_ext_token2id.keys())[len(tokenizer.vocab):]
		tokenizer.add_tokens(extra_tokens)

	# decode the predictions into strings
	pred_nlg_answers = decode_fn(preds, tokenizer)
	pred_nlg_answers = postprocess_decoded_seq(pred_nlg_answers)

	# restore tokenizer to each original vocabulary
	if is_extended:
		tokenizer.remove_tokens(extra_tokens)

	if mode == "infer":

		with open(os.path.join(eval_path, "predictions_infer.json"), "a") as f:
			for pred_nlg, q_id in zip(pred_nlg_answers, query_ids):
				json.dump({"answers": [pred_nlg], "query_id": int(q_id)}, f)
				f.write("\n")

	else:

		with open(os.path.join(eval_path, "predictions_eval.json"), "a") as f:
			for pred_nlg, q_id in zip(pred_nlg_answers, query_ids):
				json.dump({"answers": [pred_nlg], "query_id": int(q_id)}, f)
				f.write("\n")

		with open(os.path.join(eval_path, "reference_eval.json"), "a") as f:
			for trg_nlg, q_id in zip(trg_nlg_answers, query_ids):
				json.dump({"answers": trg_nlg, "query_id": int(q_id)}, f)
				f.write("\n")

		with open(os.path.join(eval_path, "lambdas.json"), "a") as f:
			for l, q_id in zip(avg_lambdas.tolist(), query_ids):
				json.dump({"lambdas": {"vocabulary": l[0], "question": l[1], "qa_answer": l[2], "passage": l[3]},
							"query_id": int(q_id)}, f)
				f.write("\n")

		with open(os.path.join(eval_path, "lengths.json"), "a") as f:
			for length, q_id in zip(lengths.tolist(), query_ids):
				json.dump({"length": length, "query_id": int(q_id)}, f)
				f.write("\n")


if __name__ == '__main__':

	args = parse_arguments()
	eval(args)