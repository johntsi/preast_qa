import torch
from collections import OrderedDict
import numpy as np
import os
import json
import matplotlib.pyplot as plt

from qa_model import QaModel
from qa_metrics import calculate_classification_metrics, calculate_auc_score
import my_constants


def decode_fn(seq, tokenizer):
	"""
	Given some indices in the vocabulary of the tokenizer
	decodes a sequence until the EOS token is found
	Args
		seq: 2d long tensor [batch_size x T]
		tokenizer: the tokenizer object
	Returns
		seq_decoded: list[str]
			the decoded answers
			
	"""

	eos_idx = tokenizer.vocab[my_constants.eos_token]

	all_pad = torch.all(seq == 0, dim = -1).tolist()

	seq = torch.cat((seq.cpu(), torch.ones([seq.size(0), 1], dtype = torch.long) * eos_idx), dim = -1).numpy()
	eos_pos = np.argmax(seq == eos_idx, axis = 1)
	seq_decoded = [tokenizer.decode(s[:e]) if (e > 0 and not a) else ["No Answer Present."]
					for s, e, a in zip(seq, eos_pos, all_pad)]

	return seq_decoded


def get_architecture(checkpoint_args, args):
	"""
	Loads architecure-specific arguments from a checkpoint

	Args
		checkpoint_args: the arguments of the checkpoint
		args: the arguments of this run
	Returns
		args: the updated arguments of this run
	"""

	args.num_layers_shared_enc = checkpoint_args.num_layers_shared_enc
	args.num_layers_passage_enc = checkpoint_args.num_layers_passage_enc
	args.num_layers_question_enc = checkpoint_args.num_layers_question_enc
	args.num_layers_dec = checkpoint_args.num_layers_dec
	args.d_model = checkpoint_args.d_model
	args.d_inner = checkpoint_args.d_inner
	args.heads = checkpoint_args.heads
	args.include_dec = checkpoint_args.include_dec
	args.include_rnk = checkpoint_args.include_rnk
	args.include_cls = checkpoint_args.include_cls
	args.rnk_method = checkpoint_args.rnk_method
	args.cls_method = checkpoint_args.cls_method
	args.include_rnk_transformer = checkpoint_args.include_rnk_transformer
	
	return args


def init_eval_model(args, main_device, device_list, fixed_token2id, model_checkpoint, only_encoder = False):
	"""
	Args
		args: the arguments of the run
		device: torch device object
		fixed_token2id: dictionary of the fixed vocab
		model_checkpoint: torch checkpoint object
		only_encoder: bool, whether to initialize only the encoder part of the model
	Returns
		model: intialized model in eval mode
			torch.nn.module
	"""

	model = QaModel(args, fixed_token2id, main_device, only_encoder = only_encoder)

	model.to(main_device)

	if only_encoder:
		model_checkpoint = OrderedDict({key: model_checkpoint[key] for key in model_checkpoint.keys()
									if any(["embedder" in key, "reader" in key, "rnk_transformer" in key,
										"ranker" in key, "classifier" in key]) and "question_transformer" not in key})

	model_checkpoint = OrderedDict({key: model_checkpoint[key] if "positional_encoder" not in key
									else model.state_dict()[key].data for key in model_checkpoint.keys()})

	model.load_state_dict(model_checkpoint)

	# Optionally parallelize model
	if len(device_list) > 1:
		model = torch.nn.DataParallel(model, device_ids = device_list, output_device = main_device)

	model.eval()

	return model


def tune_answerability(eval_path, results):
	"""
	Finds the optimal answerability threshold by progressivelly tuning
	it in smaller areas betweeen 0 and 1
	Then calculates ann saves the classification metrics for it

	Args:
		eval_path: str
		results: dict
	"""

	def tune(start, end, steps, results):
		cls_results = {}
		opt_p = 0
		opt_f1 = 0
		P = np.linspace(start, end, steps)
		for p in P:
			recall, precision, f1 = calculate_classification_metrics(results["trg"], results["prob"], p)
			if f1 > opt_f1:
				opt_f1 = f1
				opt_recall = recall
				opt_precision = precision
				opt_p = p
			cls_results[p] = {"recall": recall, "precision": precision, "f1": f1}

		return cls_results, opt_p, opt_f1, opt_recall, opt_precision

	start, end, steps0 = 0.01, 0.99, 500
	cls_results_lvl0, opt_p_lvl0, _, _, _ = tune(start, end, steps0, results)

	steps1 = 100
	small_step = (end - start) / steps1
	_, opt_p_lvl1, _, _, _ = tune(opt_p_lvl0 - small_step, opt_p_lvl0 + small_step, steps1, results)

	small_step = 2 * small_step / steps1
	_, opt_p_lvl2, opt_f1_lvl2, opt_recall_lvl2, opt_precision_lvl2 = tune(opt_p_lvl1 - small_step, opt_p_lvl1 + small_step, steps1, results)
	
	auc_score = calculate_auc_score(results["trg"], results["prob"])

	with open(os.path.join(eval_path, "classification_metrics.json"), "w") as f:
		json.dump({"threshold": float(opt_p_lvl2), "f1": float(opt_f1_lvl2),
					"recall": float(opt_recall_lvl2), "precision": float(opt_precision_lvl2), "auc": float(auc_score)}, f)

	P = np.linspace(start, end, steps0)

	fig, ax = plt.subplots(3, 1, sharex = True, sharey = True)
	ax[0].plot(P, [cls_results_lvl0[p]["recall"] for p in P], linewidth = 0.5)
	ax[0].set_title("Recall")
	ax[0].xaxis.set_visible(False)
	ax[0].axvline(x = opt_p_lvl2, ymax = 1, ymin = -1.5, c = "red", linewidth = 2, zorder = 0, clip_on = False)
	ax[1].plot(P, [cls_results_lvl0[p]["precision"] for p in P], linewidth = 0.5)
	ax[1].set_title("Precision")
	ax[1].xaxis.set_visible(False)
	ax[1].axvline(x = opt_p_lvl2, ymax = 1, ymin = -1.5, c = "red", linewidth = 2, zorder = 0, clip_on = False)
	ax[2].plot(P, [cls_results_lvl0[p]["f1"] for p in P], linewidth = 0.5)
	ax[2].set_title("F1 Score")
	ax[2].axvline(x = opt_p_lvl2, ymax = 1, ymin = 0, c = "red", linewidth = 2, zorder = 0, clip_on = False)
	plt.savefig(os.path.join(eval_path, "threshold_plot.png"), dpi = 400)