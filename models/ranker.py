import torch
from torch import nn

from transformer_blocks import EncoderLayer


class Ranker(nn.Module):
	"""
	Ranker module that uses the CLS representation of each passage

	Pointwise is the method of Nishida et al. that considers every passage individually
		and computes a relevance score for each passage with respect to the query

	Pairwise is the method introduced in this research that considers every pair of passages
		and computes a score which indicates which one is more relevant

	"""

	def __init__(self, args):
		super(Ranker, self).__init__()
		
		self.d_model = args.d_model
		self.method = args.rnk_method

		if self.method == "pointwise":
			self.rnk_layer = nn.Linear(self.d_model, 1, bias = False)

		elif self.method == "pairwise":
			self.rnk_layer = nn.Linear(4 * self.d_model, 3, bias = False)

		if args.include_rnk_transformer:
			self.rnk_transformer = EncoderLayer(self.d_model, args.d_inner, args.heads, args.dropout_rate)


	def forward(self, Mp_cls, mask_p):
		"""
		Args
			Mp_cls: the first token of every passage (cls token)
				(3d float tensor) [bs x K x d_model]
		Returns
			rnk_scores:
				(pointwise) the relevance score of each of each passage
					(2d float tensor) [bs x K]
				(pairwise) the relevance advantage of each passage wrt to the rest
					(4d float tensor) [bs x K x K x 3]
		"""

		bs, K, _ = Mp_cls.size()

		# pass the cls tokens through a transformer layer
		if hasattr(self, "rnk_transformer"):
			Mp_cls = self.rnk_transformer(Mp_cls, mask = mask_p[:, :, 1, :])

		if self.method == "pointwise":
			# pass the cls tokens through the ranking layer and put the result back into a 2d view
			rnk_scores = self.rnk_layer(Mp_cls.view(bs * K, self.d_model)).view(bs, K)  # (2d) [batch_size x num_passages]

		elif self.method == "pairwise":
			# pass the pairwise features through a linear layer to get for every pair of passages
			# a 3-dimensional vector which represents the scores that
			# (0) the second passage is more relevant
			# (1) the passages are of equal relevance
			# (2) the second passage is less relevant
			rnk_scores = self.rnk_layer(torch.cat([Mp_cls.unsqueeze(1).expand(bs, K, K, self.d_model),
													Mp_cls.unsqueeze(2).expand(bs, K, K, self.d_model),
													torch.abs(Mp_cls.unsqueeze(1) - Mp_cls.unsqueeze(2)),
													Mp_cls.unsqueeze(1) * Mp_cls.unsqueeze(2)],
										dim = -1))  # (4d) [bs x K x K x 3]

		return rnk_scores