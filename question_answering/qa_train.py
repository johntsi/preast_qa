import torch
from tensorboardX import SummaryWriter
import numpy as np
import os
from copy import deepcopy
from datetime import datetime
from multiprocessing import cpu_count
import gc
from time import time
import random
import sys

os.chdir("question_answering")

sys.path.append("./../models")
sys.path.append("./../general_modules")

from qa_argument_parser_helper import parse_arguments
from qa_train_helpers import init_dataloader, init_model, init_loss_fns, \
						plot_grad_flow, get_store_dicts, pprint_and_log
from my_train_helpers import init_optimizer, ssave


def train(args):

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
	if not args.deterministic and args.cudnn_backend and n_gpu > 0:
		torch.backends.cudnn.benchmark = True
		torch.backends.cudnn.enabled = True
		torch.backends.cudnn.deterministic = False

	# paths for saveing gradient flow and checkpoints
	if args.saving:
		gradient_save_path = os.path.join(args.gradient_save_path, args.run_name)
		checkpoint_path = os.path.join(args.checkpoint_path, args.run_name)
	else:
		gradient_save_path, checkpoint_path = None, None

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


	# initialize dataloader with dataset and optionaly custom batch sampler
	dataloader = init_dataloader(args, n_gpu)

	# initialize masque model, optionally load checkpoint and wrap in DataParallel
	model = init_model(args, dataloader, main_device, device_list, checkpoint)

	# intialize custom optimizer, optionally load state from checkpoint
	optimizer, current_epoch, current_train_step, global_train_step = \
			init_optimizer(args, model, dataloader, checkpoint)

	# initialize custom loss functions for the three tasks
	dec_loss_fn, rnk_loss_fn, cls_loss_fn = init_loss_fns(args, dataloader, main_device, device_list)

	# initialize summary writer
	writer = SummaryWriter(os.path.join("runs", args.run_name)) if args.saving else None

	# initilize dictionaries to track performance and feed them to the writer
	results_init, performance = get_store_dicts()
	results = deepcopy(results_init)

	# create folders for saving gradient flow and checkpoint if need
	if not bool(checkpoint) and args.saving:
		os.mkdir(gradient_save_path)
		os.mkdir(checkpoint_path)

	gc.collect()

	print("[{}] Starting training ...".format(datetime.now().time().replace(microsecond = 0)))

	start_time = time()
	for epoch in range(current_epoch, args.max_epochs):
		
		for train_step, batch in enumerate(dataloader, start = current_train_step):

			global_train_step += 1

			try:

				take_train_step(batch, model, optimizer, dec_loss_fn, rnk_loss_fn, cls_loss_fn,
								writer, main_device, results, global_train_step, gradient_save_path)

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

			if global_train_step > 1:

				# print and log to the summary writer
				if not global_train_step % args.print_and_log_every:
					pprint_and_log(writer, results, performance, global_train_step, optimizer.get_learning_rate(), args)
					results = deepcopy(results_init)
					
				# save checkpoint
				if not global_train_step % args.save_every and writer is not None:
					ssave(model, optimizer, args, epoch, train_step, global_train_step,
						checkpoint_path, "QA_model")

		current_train_step = 0
		gc.collect()

		print("[{}] Finished epoch {}".format(datetime.now().time().replace(microsecond = 0), epoch))
		if writer is not None:
			ssave(model, optimizer, args, epoch + 1, current_train_step, global_train_step,
				checkpoint_path, "QA_model")

	print(f"Total Run Time = {(time() - start_time) / 60} minutes")

	if writer is not None:
		writer.close()


def take_train_step(batch, model, optimizer, dec_loss_fn, rnk_loss_fn, cls_loss_fn,
					writer, device, results, global_train_step, gradient_save_path):


	# Representations according to the fixed vocabulary (indices)
	# the length of the sequences vary in each batch with max_seq_len = 100
	# each of lengths is set to the maximum length of the sequence in the example
	# num_passages is fixed to 10
	passage_fixed_vectors = batch[0].to(device)  # Passages (3d long tensor) [batch_size x num_passages x seq_len_passage]
	query_fixed_vectors = batch[1].to(device)  # Questions (2d long tensor) [batch_size x seq_len_question]

	relevancies = batch[3].to(device)  # Passage relevancies, (2d float tensor) [batch_size x num_passages]
	is_answerable = batch[4].to(device)  # Example answerabilites, (1d bool tensor) [batch_size]

	if args.include_dec:
		# Answer input, starts with style token
		# Non-answerable examples are padding vectors
		answer_src_vectors = batch[2].to(device)  # (2d long tensor) [batch_size x seq_lenum_answerablewer - 1]

		# Representation of the target according to the extended vocabulary (shifted, ends with eos token)
		# Non-answerable examples are padding vectors (1 at pad_idx in the d_ext_vocab dimension)
		answer_trg_vectors = batch[5].to(device)  # (3d long tensor) [batch_size x seq_lenum_answerablewer - 1 x d_ext_vocab]

		# Representation of the concatination of passages and question in the extended vocabulary
		# the examples that are not answerbale do not have a passage_query_ext_vectors (zero vectors)
		passage_query_ext_vectors = batch[6].to(device)  # (2d long tensor) [batch_size x (num_passages * seq_len_passage) + seq_lenum_answerablewer]

		maximum_vocab_id = passage_query_ext_vectors.max().item() + 1

		# list of size batch_size, each element is either "qa", "nlg" , or None if not answerabe
		styles = batch[7]

	else:
		answer_src_vectors, answer_trg_vectors, passage_query_ext_vectors, maximum_vocab_id, styles = None, None, None, None, None

	del batch

	dec_scores, rnk_scores, cls_scores, betas, lambdas = model(passage_fixed_vectors, query_fixed_vectors,
		passage_query_ext_vectors, is_answerable, answer_src_vectors, maximum_vocab_id)

	# calculate individual losses per the answerable example
	if args.include_dec and is_answerable.any().item():
		dec_loss = dec_loss_fn(dec_scores[is_answerable], answer_trg_vectors[is_answerable])  # 1d float tensor [num_answerable]
	else:
		# for convinience 0 loss if only encoder active or no answerable examples
		dec_loss = torch.tensor([.0], device = device)

	rnk_loss = rnk_loss_fn(rnk_scores, relevancies)  # 1d float tensor [batch_size]
	cls_loss = cls_loss_fn(cls_scores, is_answerable)  # 1d float tensor [batch_size]

	if args.include_dec:
		total_loss = dec_loss.mean() + args.gamma_rnk * rnk_loss.mean() + args.gamma_cls * cls_loss.mean()
	else:
		total_loss = args.gamma_rnk * rnk_loss.mean() + args.gamma_cls * cls_loss.mean()

	# add to the computational graph
	total_loss.backward()

	# clip gradients to max_grad_norm
	torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

	# save the gradient flow (useful for debugging)
	if (not global_train_step % args.plot_grad_every) and args.saving:
		plot_grad_flow(model.named_parameters(), global_train_step, writer, gradient_save_path)

	optimizer.step()
	optimizer.zero_grad()

	results["loss"]["rnk"].extend(rnk_loss.tolist())  # batch_size
	results["loss"]["cls"].extend(cls_loss.tolist())  # batch_size

	results["classification"]["trg"].extend(is_answerable.tolist())  # batch_size
	results["classification"]["prob"].extend(torch.sigmoid(cls_scores).tolist())  # batch_size

	if args.include_dec and is_answerable.any().item():

		results["loss"]["dec"].extend(dec_loss.tolist())  # num_answerable

		results["style"].extend([s for s in styles if bool(s)])  # num_answerable

		with torch.no_grad():
			lambda_v, lambda_q, lambda_p = torch.split(lambdas[is_answerable].mean(dim = 1), 1, dim = -1)

		results["lambdas"]["passage"].extend(lambda_p.tolist())
		results["lambdas"]["question"].extend(lambda_q.tolist())
		results["lambdas"]["vocabulary"].extend(lambda_v.tolist())

	if is_answerable.any().item():
		results["ranking"]["trg"].extend([r + [0] * (args.max_num_passages - passage_fixed_vectors.size(1))
										for r in relevancies[is_answerable].tolist()])  # num_answerable
		results["ranking"]["prob"].extend([b + [0] * (args.max_num_passages - passage_fixed_vectors.size(1))
										for b in betas[is_answerable].tolist()])  # num_answerable


if __name__ == '__main__':

	args = parse_arguments()
	train(args)
