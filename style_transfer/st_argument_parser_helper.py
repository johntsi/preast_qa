import argparse


def parse_arguments():

	parser = argparse.ArgumentParser()

	# model architecture (relevant only for st_train.py)
	# in st_eval.py they are inferred from the laoded checkpoint
	parser.add_argument("--num_layers_shared_enc", type = int, default = 3,
						help = "Number of layers in the shared encoder transformer")
	parser.add_argument("--num_layers_passage_enc", type = int, default = 3,
						help = "Number of layers in the passage encoder transformer")
	parser.add_argument("--num_layers_question_enc", type = int, default = 1,
						help = "Number of layers in the question encoder transformer")
	parser.add_argument("--num_layers_qa_enc", type = int, default = 3,
						help = "Number of layers in the QA answer encoder transformer")
	parser.add_argument("--num_layers_passage_enc_2", type = int, default = 2,
						help = "Number of layers in the passage encoder transformer")
	parser.add_argument("--num_layers_question_enc_2", type = int, default = 1,
						help = "Number of layers in the question encoder transformer")
	parser.add_argument("--num_layers_qa_enc_2", type = int, default = 2,
						help = "Number of layers in the QA answer encoder transformer")
	parser.add_argument("--num_layers_dec", type = int, default = 8,
						help = "Number of layers in the decoder transformer")
	parser.add_argument("--d_model", type = int, default = 296,
						help = "Dimensionality of the model")
	parser.add_argument("--d_inner", type = int, default = 256,
						help = "Dimensionality of the inner layer in the feed-forward networks")
	parser.add_argument("--heads", type = int, default = 8,
						help = "Number of attention heads")
	parser.add_argument("--coattention", type = str, choices = ["dual", "triple"], default = "dual",
						help = "The coattention of the interaction between the source sequences \
								(passage, question, QA answer")

	# lengths and sizes (relevant for both train and eval)
	parser.add_argument("--max_seq_len_passage", type = int, default = 120,
						help = "(denoted by L) Maximum number of tokens in the passages")
	parser.add_argument("--max_seq_len_question", type = int, default = 40,
						help = "(denoted by J) Maximum number of tokens in the questions")
	parser.add_argument("--max_seq_len_qa_answer", type = int, default = 100,
						help = "(denoted by N) Maximum number of tokens in the QA answers")
	parser.add_argument("--max_seq_len_nlg_answer", type = int, default = 100,
						help = "(denoted by T) Maximum number of tokens in the NLG answers")
	parser.add_argument("--batch_size", type = int, default = 80,
						help = "(denoted by bs)")
	parser.add_argument("--max_data_size", type = float, default = 100,
						help = "Percentage of the dataset examples to be used")

	# training arguments (only relevant for st_train.py)
	parser.add_argument("--load_checkpoint", action = "store_true",
						help = "whether to continue training from a saved checkpoint")
	parser.add_argument("--dropout_rate", type = float, default = 0.3,
						help = "Dropout rate to be used in Highways, Residuals and Attentions")
	parser.add_argument("--emb_dropout_rate", type = float, default = 0.1,
						help = "Separate dropout rate to be used in embedding layer")
	parser.add_argument("--weight_decay", type = float, default = .01,
						help = "Weight decay for l2 regularization")
	parser.add_argument("--init_lr", type = float, default = 0,
						help = "Learning rate at the start of the training")
	parser.add_argument("--max_lr", type = float, default = 3e-4,
						help = "Learning rate at the end of the warm up steps")
	parser.add_argument("--warm_up_steps", type = int, default = 1000,
						help = "Warm up steps")
	parser.add_argument("--max_epochs", type = int, default = 8,
						help = "Maximum number of epochs for the model to train")
	parser.add_argument("--max_grad_norm", type = float, default = 1,
						help = "Maximum norm to clip the gradients at")
	parser.add_argument("--tie_embeddings", action = "store_true",
						help = "whether to share the weights between the input and output embeddings")

	# monitoring and saving (only relevant for st_train.py)
	parser.add_argument("--print_and_log_every", type = int, default = 50,
						help = "The amount of train steps between each report on the performance")
	parser.add_argument("--save_every", type = int, default = 999999,
						help = "The amount of train steps between each save of the model")
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

	parser.add_argument("--init_from_question_answering", type = str, default = "",
						help = "The path to a qa_model checkpoint")

	# only relevant for st_eval.py
	parser.add_argument("--mode", type = str, choices = ["train", "eval", "infer"],
						help = "st_train.py goes be default to train and st_eval.py is either in eval of infer")

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