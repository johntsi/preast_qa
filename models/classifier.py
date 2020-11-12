import torch
from torch import nn


class Classifier(nn.Module):
	"""
	Classification module for answer possibility of each example
	"linear" corresponds to the classification method of Nishida et al.
	"max" corresponds to the max pooling method of this implementation
	"""

	def __init__(self, args):
		super(Classifier, self).__init__()

		self.d_model = args.d_model
		self.K = args.max_num_passages
		self.method = args.cls_method

		if self.method == "linear":
			self.cls_layer = nn.Linear(self.K * self.d_model, 1, bias=False)

		elif self.method == "max":
			self.cls_layer = nn.Linear(self.d_model, 1, bias=False)

	def forward(self, Mp_cls):
		"""
		Args
			Mp_cls: the first token of every passage (cls token),
				(3d float tensor) [bs x K x d_model]
		Returns
			cls_scores: the score that the question is answerable given the set of passasges
				(1d float tensor) [bs]
		"""

		# input for the classifier
		if self.method == "linear":
			# concatinate the cls tokens along the model dimension
			x = Mp_cls.reshape(-1, self.K * self.d_model)  # (2d) [bs x d_model * K]

		elif self.method == "max":
			# max pooling along the number of passages
			x = torch.max(Mp_cls, dim=1)[0]  # (2d) [bs x d_model]

		# pass them through the cls layer to get the scores per example
		cls_scores = self.cls_layer(x).squeeze(-1)  # (1d) [bs]

		return cls_scores
