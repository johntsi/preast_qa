import torch
import numpy as np
from collections import OrderedDict

from st_model import StModel
import my_constants


def decode_fn(seq_indices, tokenizer):
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

	seq_indices = torch.cat((seq_indices.cpu(), torch.ones(seq_indices.size(0), 1).long() * eos_idx), dim = -1).numpy()
	eos_pos = np.argmax(seq_indices == eos_idx, axis = 1)
	seq_decoded = [tokenizer.decode(s[:e]) for s, e in zip(seq_indices, eos_pos)]

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
	args.num_layers_qa_enc = checkpoint_args.num_layers_qa_enc
	args.num_layers_dec = checkpoint_args.num_layers_dec
	args.d_model = checkpoint_args.d_model
	args.d_inner = checkpoint_args.d_inner
	args.heads = checkpoint_args.heads
	args.coattention = checkpoint_args.coattention

	return args


def init_eval_model(args, fixed_token2id, main_device, device_list, model_checkpoint):
	"""
	Args
		args: the arguments of the run
		device: torch device object
		fixed_token2id: dict[str: int]
		model_checkpoint: torch checkpoint object
		only_encoder: bool, whether to initialize only the encoder part of the model
	Returns
		model: intialized model in eval mode
			torch.nn.module
	"""

	# initialize model
	model = StModel(args, fixed_token2id)

	# load checkpoint
	model_checkpoint = OrderedDict({key.replace("module.", ""): model_checkpoint[key] if "positional_encoder" not in key
										else model.state_dict()[key.replace("module.", "")].data for key in model_checkpoint.keys()})
	model.load_state_dict(model_checkpoint)

	model.to(main_device)

	# Optionally parallelize model
	if len(device_list) > 1:
		model = torch.nn.DataParallel(model, device_ids = device_list, output_device = main_device)

	model.eval()

	return model