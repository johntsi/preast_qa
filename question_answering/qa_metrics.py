from sklearn.metrics import label_ranking_average_precision_score, precision_score, recall_score, roc_curve, auc
import numpy as np


def calculate_ranking_metrics(rels, betas):
	"""
	Args
		rels: list[list[float]]
			the ground-truth relevance labels for each example for each passage
		betas: list[list[float]]
			the predicted relevance probabilities
	Returns
		map_score: float
			mean average precision
		mrr_score: float
			mean reciprocal rank
			
	"""

	if isinstance(rels, list): rels = np.array(rels)
	if isinstance(betas, list): betas = np.array(betas)

	map_score = label_ranking_average_precision_score(rels, betas)

	N, K = rels.shape
	sorted_passages = rels[np.arange(N)[:, None], np.argsort(betas, axis = 1)[:, ::-1]]
	mrr_score = np.mean(np.max(np.where(sorted_passages, 1. / (1 + np.arange(K)), 0), axis = 1))

	return map_score, mrr_score


def calculate_classification_metrics(y_true, y_prob, threshold):
	"""
	Args
		y_true: list[float]
		y_prob: list[float]
		threshold: float
	Returns
		recall: float
		precision: float
		f1: floats
	"""

	y_pred = np.array(y_prob) > threshold
	y_true = np.array(y_true)

	recall = recall_score(y_true, y_pred)
	precision = precision_score(y_true, y_pred)
	f1 = 2 * precision * recall / (precision + recall)

	return recall, precision, f1


def calculate_auc_score(y_true, y_prob):
	"""
	Args
		y_true: list[float]
		y_prob: list[float]
	Returns
		auc: float
	"""

	fpr, tpr, _ = roc_curve(np.array(y_true), np.array(y_prob), pos_label = 1)
	return auc(fpr, tpr)