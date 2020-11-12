import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss


class DecoderLoss(nn.Module):
	"""
	Loss function for answer generation
	computes the negative log-likelihood of the ground truth tokens
	averaged per number of valid tokens
	"""

	def __init__(self, pad_idx, unk_idx):
		super(DecoderLoss, self).__init__()

		self.pad_idx = pad_idx
		self.unk_idx = unk_idx

	def forward(self, probs, a_trg):
		"""
		Args:
			probs: the probabilities of the model over the extended vocabulary for each token
				(3d float tensor) [n_ans x T x d_ext_vocab]
			a_trg: the indices of the ground truth tokens in the extended vocabulary
				(2d long tensor) [n_ans x T]
		Retuns:
			dec_loss: negative average log-likelihood for each example
				(1d float tensor) [n_ans]
		"""

		current_device = torch.device(f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_available() else torch.device("cpu")

		# take into account only non-padding and non-unkown tokens
		# [n_ans x T]
		mask = (a_trg != self.pad_idx) * (a_trg != self.unk_idx)
		mask = mask.to(current_device)

		# number of valid tokens per example
		n_steps = mask.sum(axis = 1)  # (1d long) [n_ans]

		# probabilities of the ground-truth (valid positions) in the sequences (2d float) [n_ans x T]
		target_probs = torch.gather(probs, 2, a_trg.unsqueeze(2)).squeeze(2).masked_fill(mask == 0, 1)

		# negative average log-probabilities
		dec_loss = - torch.log(target_probs).sum(dim = 1) / n_steps   # (1d float) [n_ans]

		return dec_loss


class RankerLoss(nn.Module):
	"""
	Loss functions for the pointwise and pairiwse rankers
	"""
	
	def __init__(self, args, label_smoothing = False):
		super(RankerLoss, self).__init__()
		
		self.method = args.rnk_method

		if self.method == "pointwise":
			self.loss_fn = BCEWithLogitsLoss(reduction = "none")
			self.label_smoothing = label_smoothing
			self.epsilon = args.epsilon_smoothing

		else:
			self.K = args.max_num_passages

	def forward(self, rnk_scores, rel):
		"""
		Args:
			rnk_scores: the scores of the ranker module
				Pointwise ranker: (2d float tensor) [bs x K]
				Pairwise ranker: (4f float tensor) [bs x K x K x 3]
			rel: ground-truth relevance labels for each passage
				(2d long tensor) [bs x K]
		Retuns:
			rnk_loss:
				Pointwise: negative average log-likelihood of the correct labels for example
					averaged per number of passages in the example
				Pairwise: negative average log-likelihood of the correct labels for each example
					averaged per number of comparisons K**2
				(1d float tensor) [n_ans]
		"""

		if self.method == "pointwise":

			rel = rel.float()

			if self.label_smoothing:
				rel = rel * (1 - self.epsilon)

			# ranking loss per example averaged over the number of available passages
			rnk_loss = self.loss_fn(rnk_scores, rel).mean(dim = 1)

		else:

			current_device = torch.device(f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_available() else torch.device("cpu")

			# batch size and number of passages
			bs, K = rel.size()

			# transform the pointwise relevance labels to pairwise relevance labels
			# (3d float tensor) [bs x K x K
			r = torch.ones([bs, K, K], dtype = torch.long, device = current_device)
			r = r.masked_fill(rel.unsqueeze(2) > rel.unsqueeze(1), 2)
			r = r.masked_fill(rel.unsqueeze(2) < rel.unsqueeze(1), 0)

			# negative average log likelihood of the correct pairwise labels per example
			# averaged per number of comparisons
			rnk_loss = - torch.log(torch.gather(F.softmax(rnk_scores, dim = -1),
									3, r.unsqueeze(3)).squeeze(3)).sum(dim = (1, 2)) / K**2

		return rnk_loss


class ClassifierLoss(nn.Module):
	def __init__(self, args, label_smoothing=False):
		super(ClassifierLoss, self).__init__()

		self.loss_fn = BCEWithLogitsLoss(reduction="none")

		# positive class label smoothing for regularization
		self.epsilon = args.epsilon_smoothing
		self.label_smoothing = label_smoothing

	def forward(self, scores, ans):
		"""
		Args:
			scores: non-probabilites scores from the classifier module
				(1d float tensor) [bs]
			ans: ground-truth labels for the answerability of each example
				(1d long tensor) [bs]
		Retuns:
			cls_loss: negative log-likelihood for each example
				(1d float tensor) [bs]
		"""

		ans = ans.float()
		if self.label_smoothing:
			ans = ans * (1 - self.epsilon)

		cls_loss = self.loss_fn(scores, ans)

		return cls_loss