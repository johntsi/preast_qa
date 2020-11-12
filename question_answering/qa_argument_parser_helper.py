import argparse


def parse_arguments():

	parser = argparse.ArgumentParser()

	# model architecture (relevant only for training)
	# in evaluation they are inferred from the loaded checkpoint
	parser.add_argument("--num_layers_shared_enc", type = int, default = 3,
						help = "Number of layers in the shared encoder transformer")
	parser.add_argument("--num_layers_passage_enc", type = int, default = 5,
						help = "Number of layers in the passage encoder transformer")
	parser.add_argument("--num_layers_question_enc", type = int, default = 2,
						help = "Number of layers in the question encoder transformer")
	parser.add_argument("--num_layers_dec", type = int, default = 8,
						help = "Number of layers in the decoder transformer")
	parser.add_argument("--d_model", type = int, default = 296,
						help = "Dimensionality of the model")
	parser.add_argument("--d_inner", type = int, default = 256,
						help = "Dimensionality of the inner layer in the feed-forward networks")
	parser.add_argument("--heads", type = int, default = 8,
						help = "Number of attention heads")
	parser.add_argument("--include_dec", action = "store_true",
						help = "Whether the question-answering task is active in training")
	parser.add_argument("--include_rnk", action = "store_true",
						help = "Whethr the relevance ranking task is active in training")
	parser.add_argument("--include_cls", action = "store_true",
						help = "Whether the answerability classification task is active in training")
	parser.add_argument("--rnk_method", type = str, choices = ["pointwise", "pairwise"], default = "pairwise",
						help = "The relevance ranking method")
	parser.add_argument("--cls_method", type = str, choices = ["linear", "max"], default = "max",
						help = "The answerability classifier method")
	parser.add_argument("--include_rnk_transformer", action = "store_true",
						help = "Whether to include a passage-to-passage transformer layer in the ranker")
	parser.add_argument("--tie_embeddings", action = "store_true",
						help = "whether to share the weights between the input and output embeddings")

	# lengths and sizes (relevant for both train and eval)
	parser.add_argument("--max_num_passages", type = int, default = 10,
						help = "(denoted by K) Maximum number of passages used for each example")
	parser.add_argument("--seq_len_passage", type = int, default = 100,
						help = "(denoted by L) Maximum number of tokens in the passages")
	parser.add_argument("--seq_len_question", type = int, default = 40,
						help = "(denoted by J) Maximum number of tokens in the questions")
	parser.add_argument("--seq_len_answer", type = int, default = 100,
						help = "(denoted by T) Maximum number of tokens in the answers")
	parser.add_argument("--batch_size", type = int, default = 44,
						help = "(denoted by bs)")
	parser.add_argument("--max_data_size", type = float, default = 100,
						help = "Percentage of the dataset examples to be used")

	# weights in the combined loss for mulit-task learning
	parser.add_argument("--gamma_rnk", type = float, default = 0.265,
						help = "Weighting factor of the ranker loss in the total loss")
	parser.add_argument("--gamma_cls", type = float, default = 0.1,
						help = "Weighting factor of the classification loss in the total loss")

	# training arguments (not relevant for eval)
	parser.add_argument("--load_checkpoint", action = "store_true",
						help = "Whether to load model and optimizer from a checkpoint (in training)")
	parser.add_argument("--dropout_rate", type = float, default = 0.3,
						help = "Dropout rate to be used in Highways, Residuals and Attentions")
	parser.add_argument("--emb_dropout_rate", type = float, default = 0.1,
						help = "Separate dropout rate to be used in embedding layer")
	parser.add_argument("--weight_decay", type = float, default = .01,
						help = "Weight decay for l2 regularization")
	parser.add_argument("--epsilon_smoothing", type = float, default = 0.1,
						help = "Smoothing factor to be used in the label-smoothing regularization of the rnk and cls loss")
	parser.add_argument("--init_lr", type = float, default = 0,
						help = "Learning rate at the start of the training")
	parser.add_argument("--max_lr", type = float, default = 2.5e-4,
						help = "Learning rate at the end of the warm up steps")
	parser.add_argument("--warm_up_steps", type = int, default = 4000,
						help = "Warm up steps")
	parser.add_argument("--max_epochs", type = int, default = 8,
						help = "Maximum number of epochs for the model to train")
	parser.add_argument("--max_grad_norm", type = float, default = 1,
						help = "Maximum norm to clip the gradients at")

	# the available styles for multi-style learning
	parser.add_argument("--available_styles", type = str, default = "both", choices = ["both", "qa", "nlg"],
						help = "The available answer styles")

	# monitoring during training
	parser.add_argument("--print_and_log_every", type = int, default = 50,
						help = "The amount of train steps between each report on the performance")
	parser.add_argument("--save_every", type = int, default = 999999,
						help = "The amount of train steps between each save of the model")
	parser.add_argument("--plot_grad_every", type = int, default = 999999,
						help = "The amount of train steps between each plotting of the gradient flow")
	parser.add_argument("--saving", action = "store_true",
						help = "Whether to save checkpoints during training")

	# memory, compute and random seeds
	parser.add_argument("--cudnn_backend", action = "store_true",
						help = "Whether to use cuDNN backend in training")
	parser.add_argument("--num_workers", type = int, default = 0,
						help = "Number of workers to be used in the dataloader (set to half the number of CPU cores works best)")
	parser.add_argument("--deterministic", action = "store_true",
						help = "Whether to set all random processes to deterministic")
	parser.add_argument("--seed", type = int, default = 42,
						help = "The random seed to be used in case of a deterministic run")
	parser.add_argument("--pin_memory", action = "store_true",
						help = "Whether to pin memory for loading data (only if gpu available)")

	# custom batch sampler (for training and multi-task learning)
	parser.add_argument("--custom_batch_sampler", action = "store_true",
						help = "whether to use a custom batch sampler that balances the load of gpus in training with nn.DataParallel")
	parser.add_argument("--num_answerable_per_batch", type = int, default = 27,
						help = "The number of answerable examples per batch used for the custom batch sampler.")

	# use of generated NLG answers from the style-transfer task
	parser.add_argument("--use_generated_nlg_answers", action = "store_true",
						help = "Whether the dataset should be augmented by the generated nlg answers \
						from the style transfer model")
	parser.add_argument("--generated_nlg_answers_path", type = str, default = "",
						help = "the path to the json file of the generated nlg answers")
	
	# evaluation arguments (not relevant for training)
	parser.add_argument("--max_seq_len_dec", type = int, default = 100,
						help = "Maximum number of decoding steps for inference")
	parser.add_argument("--subset_name", type = str, choices = ["ALL", "ANS", "NLG"], default = "ALL",
						help = "ALL includes all the examples, ANS the answerable ones and NLG the ones \
								that have an nlg style answer. ALL > ANS > NLG in terms of size.")

	# file names
	parser.add_argument("--dataset_name", type = str, default = "train", choices = ["train", "dev"],
						help = "The name of the dataset to be used.")
	parser.add_argument("--embeddings_name", type = str, default = "glove.840B.300d.txt.pt",
						help = "The name of the file containing the pre-trained GloVe vectors")
	parser.add_argument("--run_name", type = str, default = "",
						help = "The name of the run")
	parser.add_argument("--run_subname", type = str, default = "",
						help = "the specific checkpoint of the run (most recent if not specified")

	# paths
	parser.add_argument("--gradient_save_path", type = str, default = "gradients",
						help = "directory path to save gradient flow plots for this run")
	parser.add_argument("--checkpoint_path", type = str, default = "checkpoints",
						help = "Path to save and load checkpoints")
	parser.add_argument("--eval_path", type = str, default = "",
						help = "evaluation path, if not specificed is created using the run_name and \
								run_subname arguments")

	args = parser.parse_args()

	print("*" * 60)
	for arg in vars(args):
		print(arg, getattr(args, arg))
	print("*" * 60)

	return args