import torch
from tensorboardX import SummaryWriter
import numpy as np
import os
from datetime import datetime
from multiprocessing import cpu_count
import gc
import random
import sys

os.chdir("style_transfer")

sys.path.append("./../models")
sys.path.append("./../general_modules")

from st_argument_parser_helper import parse_arguments
from st_train_helpers import init_dataloader, init_model, pprint_and_log
from my_losses import DecoderLoss
from my_train_helpers import init_optimizer, ssave


def train(args):

	args.mode = "train"

	# number of cpu and gpu devices
	n_gpu = torch.cuda.device_count()
	n_cpu = cpu_count()
	print(f"Number of cuda devices: {n_gpu} | Number of CPU cores: {n_cpu}")

	# specify main device and all devices (if gpu available)
	device_list = [torch.device(f"cuda:{i}") for i in range(n_gpu)]
	main_device = device_list[0] if n_gpu > 0 else torch.device("cpu")
	print(f"Main device: {main_device}")
	print(f"Parallel devices = {device_list}")

	if args.deterministic:
		torch.manual_seed(args.seed)
		torch.cuda.manual_seed(args.seed)
		np.random.seed(args.seed)
		random.seed(args.seed)
		torch.backends.cudnn.enabled = False
		torch.backends.cudnn.deterministic = True

	# initialize cuDNN backend
	if args.cudnn_backend and n_gpu > 0:
		torch.backends.cudnn.benchmark = True
		torch.backends.cudnn.enabled = True
		torch.backends.cudnn.deterministic = False

	# paths for saveing gradient flow and checkpoints
	if args.saving:
		checkpoint_path = os.path.join(args.checkpoint_path, args.run_name)
	else:
		checkpoint_path = None

	# load checkpoint and fetch its arguments
	if args.load_checkpoint:
		# most recent checkpoint if not specificed
		specific_checkpoint = os.listdir(checkpoint_path)[-1] if args.run_subname == "" else args.run_subname
		specific_checkpoint_path = os.path.join(checkpoint_path, specific_checkpoint)
		checkpoint = torch.load(specific_checkpoint_path, map_location = main_device)

		args = checkpoint["args"]
		print(f"Loaded arguments from {specific_checkpoint_path}")
	else:
		checkpoint = None


	# initialize dataloader module
	dataloader = init_dataloader(args)

	# initialize masque model, optionally load checkpoint and wrap in DataParallel
	model = init_model(args, dataloader, main_device, device_list, checkpoint)

	# intialize custom optimizer, optionally load state from checkpoint
	optimizer, current_epoch, current_train_step, global_train_step = init_optimizer(args, model, dataloader, checkpoint)

	results = {"loss": [],
				"lambdas": {"vocab": [], "question": [], "qa_answer": [], "passage": []}}

	# initialize summary writer
	writer = SummaryWriter(os.path.join("runs", args.run_name)) if args.saving else None

	# initilaize the loss function
	loss_fn = DecoderLoss(dataloader.dataset.pad_idx, dataloader.dataset.unk_idx)
	if n_gpu > 1:
		loss_fn = torch.nn.DataParallel(loss_fn, device_ids = device_list, output_device = main_device)
	loss_fn = loss_fn.to(main_device)

	# create folders for saving gradient flow and checkpoint if need
	if not bool(checkpoint) and args.saving:
		os.mkdir(checkpoint_path)

	gc.collect()

	for epoch in range(current_epoch, args.max_epochs):

		for train_step, batch in enumerate(dataloader, start = current_train_step):

			global_train_step += 1

			try:

				take_train_step(batch, model, optimizer, loss_fn, main_device, results)

			except RuntimeError as e:
				# to catch OOM errors
				print("[{}]".format(datetime.now().time().replace(microsecond = 0)), global_train_step, e)
				del batch
				gc.collect()
				for device_id in range(n_gpu):
					with torch.cuda.device(f"cuda:{device_id}"):
						torch.cuda.empty_cache()

			# empty cache after the first (optimizing) iteration
			if args.cudnn_backend and global_train_step == 1:
				gc.collect()
				for device_id in range(n_gpu):
					with torch.cuda.device(f"cuda:{device_id}"):
						torch.cuda.empty_cache()

			# print and log to the summary writer
			if (not global_train_step % args.print_and_log_every) and global_train_step:
				pprint_and_log(writer, results, global_train_step, optimizer.get_learning_rate())
				results = {"loss": [],
							"lambdas": {"vocab": [], "question": [], "qa_answer": [], "passage": []}}

			# save checkpoint
			if (not global_train_step % args.save_every) and global_train_step:
							ssave(model, optimizer, args, epoch, current_train_step, global_train_step,
				checkpoint_path, "ST_model")

		current_train_step = 0
		gc.collect()

		print("[{}] Finished epoch {}".format(datetime.now().time().replace(microsecond = 0), epoch))
		if bool(writer):
			ssave(model, optimizer, args, epoch + 1, current_train_step, global_train_step,
				checkpoint_path, "ST_model")

	if writer is not None:
		writer.close()


def take_train_step(batch, model, optimizer, loss_fn, device, results):

	# Representations of the sequences in the fixed vocabulary (indices)
	passage_fixed_vectors = batch[0].to(device)  # 2d long tensor [batch_size x seq_len_passage]
	query_fixed_vectors = batch[1].to(device)  # (2d long tensor [batch_size x seq_len_question]
	qa_answer_fixed_vectors = batch[2].to(device)  # 2d long tensor [batch_size x seq_len_qa]
	nlg_answer_src_vectors = batch[3].to(device)  # (2d long tensor) [batch_size x seq_len_nlg - 1]

	# Representation of the NLG answer in the extended vocabulary (shifted, ends with eos token)
	nlg_answer_trg_vectors = batch[4].to(device)  # (2d long tensor) [batch_size x seq_len_nlg - 1]

	# Representation of the concatination of passage, question and qa_answer in the extended vocabulary
	source_ext_vectors = batch[5].to(device)  # (2d long tensor) [batch_size x seq_len_passage + seq_len_question + seq_len_answer]

	d_ext_vocab = source_ext_vectors.max().item() + 1

	del batch

	# forward pass
	dec_scores, lambdas = model(passage_fixed_vectors, query_fixed_vectors, qa_answer_fixed_vectors,
							source_ext_vectors, d_ext_vocab, nlg_answer_src_vectors)

	# calculate loss per example
	loss = loss_fn(dec_scores, nlg_answer_trg_vectors)

	# add the average loss to the computational graph
	loss.mean().backward()
	
	# clip gradients
	torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

	# apply gradients
	optimizer.step()
	optimizer.zero_grad()

	# store losses and lambdas per example
	with torch.no_grad():
		lambda_vocab, lambda_question, lambda_qa_answer, lambda_passage = torch.split(lambdas.mean(dim = 1), 1, dim = -1)
	results["loss"].extend(loss.tolist())
	results["lambdas"]["vocab"].extend(lambda_vocab.tolist())
	results["lambdas"]["question"].extend(lambda_question.tolist())
	results["lambdas"]["qa_answer"].extend(lambda_qa_answer.tolist())
	results["lambdas"]["passage"].extend(lambda_passage.tolist())


if __name__ == '__main__':

	args = parse_arguments()
	train(args)
