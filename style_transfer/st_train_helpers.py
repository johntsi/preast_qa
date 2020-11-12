import torch
from torch.utils.data import DataLoader
from collections import OrderedDict
from datetime import datetime
import numpy as np

from st_model import StModel
from st_dataset import StDataset
from st_collate_fn import st_collate_fn


def init_dataloader(args):
	"""
	Initializes dataloder model based on MyDataset class of the MS Marco
	Args:
		args: argument parser object
	Returns:
		dataloader: torch.utils.data.Dataloader
	"""

	dataloader = DataLoader(StDataset(args),
							args.batch_size,
							collate_fn = st_collate_fn,
							shuffle = True,
							num_workers = args.num_workers,
							pin_memory = args.pin_memory,
							drop_last = True)
	print(f"Initialized dataset with {len(dataloader.dataset)} examples")
	print(f"Initialized dataloader with {dataloader.num_workers} number of workers")
	print(f"                       and pin memory option set to {dataloader.pin_memory}")

	return dataloader


def init_model(args, dataloader, main_device, device_list, model_checkpoint):
	"""
	Initializes Style-Transfer model
	Optionally loads checkpoint from either a Question-Answering model or a Style-Transfer one
	Optionally parallelizes module
	Sets to training mode

	Args:
		args: argument parser object
		dataloader: torch.utils.data.dataloader
		main_device: torch.device
		device_list: list[torch.device]
		model_checkpoint: OrderedDict[str: torch.tensor]
	"""

	# initialize model
	model = StModel(args, dataloader.dataset.fixed_token2id)

	if bool(model_checkpoint):

		print("Loading checkpoint from Style-Transfer model")

		model_checkpoint = {k: v.data if ("positional_encoder" in k)
					else model_checkpoint[k] for k, v in model.state_dict().items()}

		model.load_state_dict(model_checkpoint)

	elif bool(args.init_from_question_answering):

		print("Loading checkpoint from Question-Answering model")

		model_checkpoint = torch.load(args.init_from_question_answering, map_location = main_device)["model"]

		invalid_keys = ["positional_encoder", "passage_transformer.lin.weight", "question_transformer.lin.weight"]

		model_checkpoint = OrderedDict({key: model_checkpoint[key]
								if all([inv not in key for inv in invalid_keys]) and
								(key in model_checkpoint.keys())
								else model.state_dict()[key].data
								for key in model.state_dict().keys()})

		model.load_state_dict(model_checkpoint)
	
	if len(device_list) > 1:
		model = torch.nn.DataParallel(model, device_ids = device_list, output_device = main_device)

	# send to device and start training mode
	model.to(main_device)
	model.train()

	return model


def pprint_and_log(writer, results, global_train_step, lr):
	"""
	calculates performance of the sampled iterations during the last <args.print_and_log_every> iterations
	prints performance
	logs performance to the summary writer

	Args:
		writer: torch writer object
		results: dict
		global_train_step: int
		lr: float
	"""


	avg_likelihood_loss = np.mean(results["loss"])
	avg_lambda_v = np.mean(results["lambdas"]["vocab"])
	avg_lambda_q = np.mean(results["lambdas"]["question"])
	avg_lambda_qa = np.mean(results["lambdas"]["qa_answer"])
	avg_lambda_p = np.mean(results["lambdas"]["passage"])

	dt = datetime.now().time().replace(microsecond = 0)
	print("[{}] Step {}: Loss = {:.4f} || Lambdas = (v) {:.3f} (q) {:.3f} (qa) {:.3f} (p) {:.3f} | lr = {:.6f}".format(
		dt, str(global_train_step).zfill(5), avg_likelihood_loss,
		avg_lambda_v, avg_lambda_q, avg_lambda_qa, avg_lambda_p, lr))

	if writer is not None:
		writer.add_scalar("loss_L", avg_likelihood_loss, global_train_step)
		writer.add_scalar("lr", lr, global_train_step)
		writer.add_scalar("lambda_v", avg_lambda_v, global_train_step)
		writer.add_scalar("lambda_q", avg_lambda_q, global_train_step)
		writer.add_scalar("lambda_qa", avg_lambda_qa, global_train_step)
		writer.add_scalar("lambda_p", avg_lambda_p, global_train_step)