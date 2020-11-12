import torch
from torch import nn
import os
from datetime import datetime

from my_optimizer import MyOptimizer


def init_optimizer(args, model, dataloader, checkpoint):
	"""
	Initializes optimizer and optionally loads state from another checkpoint

	Args:
		args: argument parser object
		model: torch.nn.module
		dataloader: torch.utils.data.DataLoader
		checkpoint: dict (see ssave function)
	Returns:
		optimizer: torch.optim.optimizer
		current_epoch, current_train_step, global_train_step: int
	"""

	max_steps = args.max_epochs * len(dataloader)

	# init custom optimizer with warm up and cosine anealing
	optimizer = MyOptimizer(model.named_parameters(),
							args.init_lr,
							args.max_lr,
							args.warm_up_steps,
							max_steps,
							args.weight_decay)
	print(f"Initialized Adam optimizer with cosine annealing")

	if checkpoint:
		optimizer.load_state_dict(checkpoint["optimizer"])
		current_epoch, current_train_step, global_train_step = checkpoint["steps"]
		print(f"Continuing training from epoch {current_epoch}, step {current_train_step} and global_train_step {global_train_step}")

	else:
		print(f"Warm-up Steps: {args.warm_up_steps} and Max Steps: {max_steps}")
		print(f"Initial learning rate {args.init_lr} and maximum learning rate {args.max_lr}")
		current_epoch, current_train_step, global_train_step = 0, 0, 0

	return optimizer, current_epoch, current_train_step, global_train_step


def ssave(model, optimizer, args, epoch, train_step, global_train_step, checkpoint_path, model_name):
	"""
	Saves the complete state of the training

	Args:
		model: torch module
		optimizer: torch optimizer
		args: argument parser object
		train_step: int
		global_train_step: int
		checkpoint_path: str
	"""

	dt = datetime.now().time().replace(microsecond = 0)
	torch.save({"model": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
				"optimizer": optimizer.state_dict(),
				"steps": [epoch, train_step, global_train_step],
				"args": args},
				os.path.join(checkpoint_path, model_name + "_" + str(global_train_step) + ".pt"))
	print("[{}] Saved state at {}".format(dt, checkpoint_path))