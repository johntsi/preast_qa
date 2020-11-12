import numpy as np
from torch import optim


class MyOptimizer():
	"""
	A wrapper for an Adam optimizer with linear warm-up and cosine annealing
	"""

	def __init__(self, named_parameters, init_lr, max_lr, warm_steps, max_steps, weight_decay):
		self.init_lr = init_lr
		self.max_lr = max_lr
		self.warm_steps = warm_steps
		self.max_steps = max_steps

		# initialize optimizer step
		self.n_steps = 0

		# Adam optimizer with modified L2 regularization for non-bias parameters
		self._optimizer = optim.AdamW([{"params": [p for n, p in named_parameters if (p.requires_grad and "weight" in n)],
										"lr": .0, "weight_decay": weight_decay},
									{"params": [p for n, p in named_parameters if (p.requires_grad and "bias" in n)],
										"lr": .0, "weight_decay": .0}])

		# array of warm-up and regular learning rate values
		warm_up_learning_rates = np.linspace(self.init_lr, self.max_lr, self.warm_steps + 1)[1:]
		regular_learning_rate = self.init_lr + 0.5 * (self.max_lr - self.init_lr) * \
								(1 + np.cos(np.arange(self.warm_steps, self.max_steps) / self.max_steps * np.pi))
								
		# putting them together
		self.learning_rates = np.append(warm_up_learning_rates, regular_learning_rate)

	def step(self):
		# updates learning rate and applies optimizer
		self._update_learning_rate()
		self._optimizer.step()
		
	def zero_grad(self):
		self._optimizer.zero_grad()

	def state_dict(self):
		return (self._optimizer.state_dict(), self.n_steps)

	def load_state_dict(self, state_dict):
		self._optimizer.load_state_dict(state_dict[0])
		self.n_steps = state_dict[1]

	def _update_learning_rate(self):
		for param_group in self._optimizer.param_groups:
			param_group['lr'] = self.learning_rates[self.n_steps]
		self.n_steps += 1

	def get_learning_rate(self):
		return self._optimizer.param_groups[0]["lr"]