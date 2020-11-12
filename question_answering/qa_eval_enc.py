from torch.utils.data import DataLoader
import torch
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
from qa_eval_helpers import get_architecture, init_eval_model, tune_answerability
from qa_collate_fn import qa_collate_fn
from qa_metrics import calculate_ranking_metrics


def eval(args):

	# number of cpu and gpu devices
	n_gpu = torch.cuda.device_count()
	n_cpu = cpu_count()
	print(f"Number of cuda devices: {n_gpu} | Number of CPU cores: {n_cpu}")

	print(f"Using the {args.subset_name} subset")

	main_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	device_list = [torch.device(f"cuda:{i}") for i in range(n_gpu)]

	checkpoint_path = os.path.join(args.checkpoint_path, args.run_name)
	specific_checkpoint = os.listdir(checkpoint_path)[-1] if args.run_subname == "" else args.run_subname
	specific_checkpoint_path = os.path.join(checkpoint_path, specific_checkpoint)
	checkpoint = torch.load(specific_checkpoint_path, map_location = main_device)
	print(f"Loaded checkpoint: {specific_checkpoint} of run: {args.run_name}")
	
	args = get_architecture(checkpoint["args"], args)
	args.num_layers_question_enc = 0
	args.include_dec = False

	if args.eval_path == "":
		eval_path = os.path.join("evaluation", args.run_name, specific_checkpoint.split("_")[-1].split(".")[0],
							args.subset_name)
	else:
		eval_path = args.eval_path
		
	os.makedirs(eval_path, exist_ok = True)

	dataloader = DataLoader(QaDataset(args, is_training = False),
							args.batch_size,
							collate_fn = qa_collate_fn,
							shuffle = False,
							num_workers = args.num_workers,
							drop_last = False)
	print(f"Initialized dataset with {len(dataloader.dataset)} examples")

	# initialize model, load checkpoint and set to eval mode
	model = init_eval_model(args, main_device, device_list, dataloader.dataset.fixed_token2id,
							checkpoint["model"], only_encoder = True)

	# to store results
	results = {"rnk": {"loss": [], "trg": [], "prob": [], "query_id": []},
				"cls": {"loss": [], "trg": [], "prob": [], "query_id": []}}

	# eval files
	file_list = [os.path.join(eval_path, filename)
		for filename in ["reference_rnk.json", "predictions_rnk.json", "reference_cls.json", "predictions_cls.json"]]

	# evaluate dataset
	with torch.no_grad():
		for batch in tqdm(iter(dataloader), miniters = 1000):
			take_eval_step(batch, model, main_device, results, file_list)

	map_score, mrr_score = calculate_ranking_metrics(results["rnk"]["trg"], results["rnk"]["prob"])
	with open(os.path.join(eval_path, "ranking_metrics.json"), "w") as f:
		json.dump({"map": float(map_score), "mrr": float(mrr_score)}, f)

	tune_answerability(eval_path, results["cls"])


def take_eval_step(batch, model, device, results, file_list):

	# Indices in the fixed vocabulary for passages and questions
	passage_fixed_vectors = batch[0].to(device)  # Passages (3d long tensor) [batch_size x num_passages x seq_len_passage]
	query_fixed_vectors = batch[1].to(device)  # Questions (2d long tensor) [batch_size x seq_len_question]

	# ground truth labels for relevance and anserability
	relevancies = batch[3].to(device)  # Passage relevancies, (2d float tensor) [batch_size x num_passages]
	is_answerable = batch[4].to(device)  # Example answerabilites, (1d bool tensor) [batch_size]

	# unique example identifiers
	query_ids = batch[10]

	batch = None

	# forward through the encoder part of the model
	_, _, cls_scores, betas, _ = model(passage_fixed_vectors, query_fixed_vectors)

	# get the ranking probabilities and ground truths for the answerable examples in the batch
	rnk_prob = betas[is_answerable].tolist()
	rnk_trg = relevancies[is_answerable].tolist()
	rnk_query_ids = [q_id for i, q_id in enumerate(query_ids) if is_answerable[i]]
	results["rnk"]["trg"].extend(rnk_trg)
	results["rnk"]["prob"].extend(rnk_prob)
	results["rnk"]["query_id"].extend(rnk_query_ids)

	# get the classification probabilities and ground truths for all the examples in the batch
	possibilities = torch.sigmoid(cls_scores)
	cls_prob = possibilities.tolist()
	cls_trg = is_answerable.tolist()
	results["cls"]["trg"].extend(cls_trg)
	results["cls"]["prob"].extend(cls_prob)
	results["cls"]["query_id"].extend(query_ids)

	# store ranking predictions
	for file_path, result in zip(file_list[:2], [rnk_trg, rnk_prob]):
		with open(file_path, "a") as f:
			for x, q_id in zip(result, rnk_query_ids):
				json.dump({"trg" if "ref" in file_path else "prob": x, "query_id": int(q_id)}, f)
				f.write("\n")

	# store classification predictions
	for file_path, result in zip(file_list[2:], [cls_trg, cls_prob]):
		with open(file_path, "a") as f:
			for x, q_id in zip(result, query_ids):
				json.dump({"trg" if "ref" in file_path else "prob": x, "query_id": int(q_id)}, f)
				f.write("\n")


if __name__ == '__main__':

	args = parse_arguments()
	eval(args)