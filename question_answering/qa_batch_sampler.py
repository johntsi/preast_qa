from torch.utils.data.sampler import Sampler
import random
import math
from itertools import cycle


class QaBatchSampler(Sampler):
	"""
	Batch sampler to balance the load (number of answerable examples) of each gpu
	for efficient training with DataParallel

	Args:
		dataset: MyDataset object
		batch_size: int
		n_gpu: int
		num_answerable_per_batch: int
	"""

	def __init__(self, dataset, batch_size, n_gpu, num_answerable_per_batch):

		self.bs = batch_size
		self.n_gpu = n_gpu
		self.bs_gpu = self.bs // self.n_gpu
		
		# indices of the examples in the dataset
		all_i = set(range(len(dataset)))
			
		self.answerable_i = sorted([i for i in all_i
			if (dataset.data["answers"][dataset.indices[i]] != ['No Answer Present.']) and
			(dataset.data["answers"][dataset.indices[i]] != [""])])
		self.unanswerable_i = sorted(all_i - set(self.answerable_i))

		# number of answerable, number of non-answerable and all
		self.n_ans = len(self.answerable_i)
		self.n_unans = len(self.unanswerable_i)
		self.n_all = len(all_i)

		# training steps in an epoch
		self.steps = self.n_all // self.bs

		assert self.n_all == self.n_ans + self.n_unans

		# ratio of answerable examples in the dataset
		self.ans_ratio = self.n_ans / self.n_all

		# number of answerable examples in a batch
		# if not given, infer it from the ratio
		self.n_ans_batch = num_answerable_per_batch if num_answerable_per_batch else math.floor(self.ans_ratio * self.bs)

		print(f"Ratio of answerables in the training set: {self.ans_ratio}")
		print(f"Answerables per batch {self.n_ans_batch}")

	def __iter__(self):

		# initialize indices
		epoch_indices = self._prepare()

		for i in epoch_indices:
			yield i

	def __len__(self):
		return self.n_all

	def _init_pool(self, pool_name):
		"""
		Initializes and shuffles the specificed pool
		"""

		if pool_name == "answerable":
			self.ans_pool = self.answerable_i.copy()
			random.shuffle(self.ans_pool)
		else:
			self.unans_pool = self.unanswerable_i.copy()
			random.shuffle(self.unans_pool)

	def _prepare(self):
		"""
		Loops through all the indices of the examples and order them in balanced batches
		re-initializes the answerable or non-answerable pool if empty
		"""

		self._init_pool("answerable")
		self._init_pool("unanswerable")

		epoch_indices = []

		for step in range(self.steps):

			n_ans_batch = self.n_ans_batch
			n_unans_batch = self.bs - n_ans_batch

			if n_ans_batch > len(self.ans_pool):
				print(f"Re-pool of answerables at {step} from {self.steps}")
				self._init_pool("answerable")

			if n_unans_batch > len(self.unans_pool):
				print(f"Re-pool of unanswerables at {step} from {self.steps}")
				self._init_pool("unanswerable")

			batch_idx = [self.ans_pool.pop() for _ in range(n_ans_batch)]
			batch_idx.extend([self.unans_pool.pop() for _ in range(n_unans_batch)])
			batch_ans = [1] * n_ans_batch + [0] * n_unans_batch

			# balance gpu allocation
			batch_idx = self._balance(batch_idx, batch_ans)

			epoch_indices.extend(batch_idx)

		return epoch_indices

	def _balance(self, batch_idx, batch_ans):
		"""
		Orders the examples in such away that when assigned to the gpus by DataParallel
		the load will be equal to each gpu

		Args:
			batch_idx: list[int]
			batch_ans: list[int]
		Returns:
			new_idx: list[int]
		"""

		gpu_allocation = [[] for _ in range(self.n_gpu)]
		gpu_idx_iter = cycle(range(self.n_gpu))

		# assign answerable indices in a balanced way
		# start from the second gpu (first one is already doing heavier work)
		current_gpu_idx = next(gpu_idx_iter)
		current_gpu_idx = next(gpu_idx_iter)
		for i, is_ans in enumerate(batch_ans):
			if is_ans:
				gpu_allocation[current_gpu_idx].append(i)
				current_gpu_idx = next(gpu_idx_iter)

		# fill remaining positions with un-answerable indices
		for i, is_ans in enumerate(batch_ans):
			if not is_ans:
				gpu_allocation[current_gpu_idx].append(i)
			if len(gpu_allocation[current_gpu_idx]) == self.bs_gpu:
				current_gpu_idx = next(gpu_idx_iter)

		# shuffle in-gpu allocations
		for current_gpu_idx in range(self.n_gpu):
			random.shuffle(gpu_allocation[current_gpu_idx])

		# flatten allocation
		new_idx = [batch_idx[i] for sub_allocation in gpu_allocation for i in sub_allocation]

		return new_idx