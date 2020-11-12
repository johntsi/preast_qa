import torch
from torch.utils.data import DataLoader, BatchSampler
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import os
from datetime import datetime
import seaborn as sns
sns.set()

from qa_metrics import calculate_ranking_metrics, calculate_auc_score
from qa_dataset import QaDataset
from qa_batch_sampler import QaBatchSampler
from qa_collate_fn import qa_collate_fn
from qa_model import QaModel

from my_losses import DecoderLoss, RankerLoss, ClassifierLoss


def init_dataloader(args, n_gpu):
	"""
	Initializes dataloader for training

	Args:
		args: argument parser object
		n_gpu: int
	Returns:
		dataloader: torch.utils.data.DataLoader
	"""

	if args.custom_batch_sampler:
		dataset = QaDataset(args)
		my_batch_sampler = BatchSampler(QaBatchSampler(dataset,
														args.batch_size,
														max(n_gpu, 1),
														args.num_answerable_per_batch),
										args.batch_size,
										drop_last = True)
		dataloader = DataLoader(dataset,
								batch_sampler = my_batch_sampler,
								collate_fn = qa_collate_fn,
								num_workers = args.num_workers,
								pin_memory = args.pin_memory and n_gpu > 0)

	else:
		dataloader = DataLoader(QaDataset(args),
								args.batch_size,
								collate_fn = qa_collate_fn,
								shuffle = True,
								num_workers = args.num_workers,
								pin_memory = args.pin_memory and n_gpu > 0,
								drop_last = True)

	print(f"Initialized dataset with {len(dataloader.dataset)} examples")
	print(f"Initialized dataloader with {dataloader.num_workers} number of workers")
	print(f"                            pin memory option set to {dataloader.pin_memory}")
	print(f"                            custom batch sampler set to {args.custom_batch_sampler}")

	return dataloader


def init_model(args, dataloader, main_device, device_list, checkpoint = None):
	"""
	Initializes model in training mode

	Args:
		args: argument parser object
		dataloader: torch.utils.data.dataloader
		main_device: torch.device
		device_list: list[torch.device]
		checkpoint: dict (see ssave)
	Returns:
		model: torch.nn.module
	"""

	model = QaModel(args, dataloader.dataset.fixed_token2id, main_device, only_encoder = not args.include_dec)

	if checkpoint:
		model.load_state_dict({k: v.data if ("positional" in k)
						else checkpoint["model"][k] for k, v in model.state_dict().items()})
		print(f"Loaded checkpoint from run: {args.run_name}")

	# Optionally parallelize model
	if len(device_list) > 1:
		model = torch.nn.DataParallel(model, device_ids = device_list, output_device = main_device)

	# send to device and start training mode
	model.to(main_device)

	model.train()

	return model


def init_loss_fns(args, dataloader, main_device, device_list):
	"""
	Intialize the loss functions for the three tasks

	Args:
		args: argument parser object
		dataloader: torch.data.utils.Dataloader
		main_device: torch.device
		device_list: list[torch.device]
	Returns:
		dec_loss_fn: torch.nn.module
		rnk_loss_fn: torch.nn.module
		cls_loss_fn: torch.nn.module
	"""

	# initilaize individual loss functions and total_loss_fn for to combine them
	dec_loss_fn = DecoderLoss(dataloader.dataset.pad_idx, dataloader.dataset.unk_idx)
	rnk_loss_fn = RankerLoss(args, label_smoothing = True)
	cls_loss_fn = ClassifierLoss(args, label_smoothing = True)

	if len(device_list) > 1:
		dec_loss_fn = torch.nn.DataParallel(dec_loss_fn, device_ids = device_list, output_device = main_device)
		rnk_loss_fn = torch.nn.DataParallel(rnk_loss_fn, device_ids = device_list, output_device = main_device)
		cls_loss_fn = torch.nn.DataParallel(cls_loss_fn, device_ids = device_list, output_device = main_device)

	dec_loss_fn = dec_loss_fn.to(main_device)
	rnk_loss_fn = rnk_loss_fn.to(main_device)
	cls_loss_fn = cls_loss_fn.to(main_device)

	return dec_loss_fn, rnk_loss_fn, cls_loss_fn


def plot_grad_flow(named_parameters, global_train_step, writer, gradient_save_path):
	"""
	Plots the gradients flowing through different layers in the net during training.
	Can be used for checking for possible gradient vanishing / exploding problems.
	Taken from https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10

	Args:
		named_parameters: model parameters
		global_train_step: int
		writer: torch writer object
		gradient_save_path: str
	"""
	
	ave_grads = []
	max_grads = []
	layers = []
	for n, p in named_parameters:
		if p.requires_grad and ("bias" not in n) and (p.grad is not None):
			writer.add_histogram(n, p.grad, global_train_step)
			if torch.max(p.grad).item() != .0:
				layers.append(n)
				ave_grads.append(p.grad.abs().mean())
				max_grads.append(p.grad.abs().max())
			else:
				layers.append(n)
				ave_grads.append(-0.5)
				max_grads.append(-1)
	plt.bar(np.arange(len(max_grads)), max_grads, alpha = 0.1, lw = 1, color = "c")
	plt.bar(np.arange(len(max_grads)), ave_grads, alpha = 0.1, lw = 1, color = "b")
	plt.hlines(0, 0, len(ave_grads) + 1, lw = 0.5, color = "k", alpha = 0.1)
	plt.xticks(range(0, len(ave_grads), 1), layers, rotation = 90, fontsize = 2.5)
	plt.yticks(fontsize = 6)
	plt.xlim(left = -0.75, right = len(ave_grads))
	plt.ylim(bottom = -0.001, top = 0.02)  # zoom in on the lower gradient regions
	plt.xlabel("Layers", fontsize = 6)
	plt.ylabel("average gradient", fontsize = 6)
	plt.title("Gradient flow", fontsize = 6)
	plt.grid(True)
	plt.legend([Line2D([0], [0], color = "c", lw = 4),
				Line2D([0], [0], color = "b", lw = 4),
				Line2D([0], [0], color = "k", lw = 4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
	plt.tight_layout()

	file_path = os.path.join(gradient_save_path, str(global_train_step) + ".png")
	plt.savefig(file_path, dpi = 400)


def get_store_dicts():
	"""
	To store stats and metrics during training
	"""

	results_init = {"loss": {"dec": [], "rnk": [], "cls": []},
				"ranking": {"trg": [], "prob": []},
				"classification": {"trg": [], "prob": []},
				"style": [],
				"lambdas": {"passage": [], "question": [], "vocabulary": []}}
	performance = {"loss": {}, "ranking": {}, "classification": {}, "other": {}, "lambdas": {}}
	return results_init, performance


def pprint_and_log(writer, results, performance, global_train_step, lr, args):
	"""
	calculates performance of the sampled iterations during the last <args.print_and_log_every> iterations
	prints performance
	logs performance to the summary writer

	Args:
		writer: torch writer object
		results: dict[dict] (from get_store_dicts)
		performance: dict[dict] (from get_store_dicts)
		global_train_step: int
		lr: float
		args: argument parser object
	"""

	performance["loss"]["rnk"] = np.mean(results["loss"]["rnk"])
	performance["loss"]["cls"] = np.mean(results["loss"]["cls"])

	performance["other"]["learning_rate"] = lr

	performance["ranking"]["map"], performance['ranking']["mrr"] = calculate_ranking_metrics(*results["ranking"].values())

	performance["classification"]["auc"] = calculate_auc_score(*results["classification"].values())

	if args.include_dec:
		mask_qa = np.array([1 if style == "qa" else 0 for style in results["style"]]).astype(np.bool)
		mask_nlg = ~mask_qa

		results["loss"]["dec"] = np.array(results["loss"]["dec"])

		performance["loss"]["dec"] = results["loss"]["dec"].mean()
		performance["loss"]["dec_qa"] = results["loss"]["dec"][mask_qa].mean() if mask_qa.any() else 0
		performance["loss"]["dec_nlg"] = results["loss"]["dec"][mask_nlg].mean() if mask_nlg.any() else 0

		results["lambdas"]["passage"] = np.array(results["lambdas"]["passage"])
		results["lambdas"]["question"] = np.array(results["lambdas"]["question"])
		results["lambdas"]["vocabulary"] = np.array(results["lambdas"]["vocabulary"])

		performance["lambdas"]["passage_all"] = np.mean(results["lambdas"]["passage"])
		performance["lambdas"]["question_all"] = np.mean(results["lambdas"]["question"])
		performance["lambdas"]["vocabulary_all"] = np.mean(results["lambdas"]["vocabulary"])

		performance["lambdas"]["passage_qa"] = results["lambdas"]["passage"][mask_qa].mean() if mask_qa.any() else 0
		performance["lambdas"]["question_qa"] = results["lambdas"]["question"][mask_qa].mean() if mask_qa.any() else 0
		performance["lambdas"]["vocabulary_qa"] = results["lambdas"]["vocabulary"][mask_qa].mean() if mask_qa.any() else 0

		performance["lambdas"]["passage_nlg"] = results["lambdas"]["passage"][mask_nlg].mean() if mask_nlg.any() else 0
		performance["lambdas"]["question_nlg"] = results["lambdas"]["question"][mask_nlg].mean() if mask_nlg.any() else 0
		performance["lambdas"]["vocabulary_nlg"] = results["lambdas"]["vocabulary"][mask_nlg].mean() if mask_nlg.any() else 0

		performance["loss"]["total"] = performance["loss"]["dec"] + args.gamma_rnk * performance["loss"]["rnk"] + \
																	args.gamma_cls * performance["loss"]["cls"]

		performance["other"]["nlg_percent"] = mask_nlg.mean()
		performance["other"]["ans_percent"] = len(results["loss"]["dec"]) / len(results["loss"]["cls"])
	else:
		performance["loss"]["total"] = args.gamma_rnk * performance["loss"]["rnk"] + \
										args.gamma_cls * performance["loss"]["cls"]
		performance["loss"]["dec"] = 0

	dt = datetime.now().time().replace(microsecond = 0)
	print("[{}] Step {}: DEC {:.4f} | RNK {:.4f} | CLS {:.4f} | TTL {:.4f} | MAP {:.4f} | AUC {:.4f} | lr = {:.6f}".format(
		dt, str(global_train_step).zfill(5), performance["loss"]["dec"],
		performance["loss"]["rnk"], performance["loss"]["cls"],
		performance["loss"]["total"], performance["ranking"]["map"],
		performance["classification"]["auc"], performance["other"]["learning_rate"]))

	if writer is not None:
		for field in ["loss", "ranking", "classification", "lambdas", "other"]:
			for metric in performance[field].keys():
				writer.add_scalar(field + "/" + metric, performance[field][metric], global_train_step)
