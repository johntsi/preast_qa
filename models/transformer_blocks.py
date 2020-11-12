from torch import nn

from attention import MultiHeadAttention
from sublayers import FeedForwardNetwork


class TransformerEncoder(nn.Module):
	"""
	Transformer Encoder module as implemented in "Attention is all you need".
	The input is first mapped via a linear transformation to dimensionality d_model
	and then passed trhoguh num_layers transformer encoder layers.
	The module can have a variable number of transformer layers
	"""
	def __init__(self, num_layers, d_input, d_inner, d_model, heads, dropout):
		super(TransformerEncoder, self).__init__()
		
		self.num_layers = num_layers

		self.lin = nn.Linear(d_input, d_model)

		self.transformer_layers = nn.ModuleList([EncoderLayer(d_model, d_inner, heads, dropout)
										for _ in range(num_layers)])

	def forward(self, x, mask = None):
		"""
		Args
			x: A sequence representation
				(3d float tensor) [bs x sequence_length x d_input]
			mask: Mask of the sequence
				(3d bool tensor), [bs x sequence_length x 1]
		Returns
			y: The result of the transformer encoder
				(3d float tensor) [bs x sequence_length x d_model]
		"""
		
		# Pass through a linear layer to map the sequence from d_input to d_model
		y = self.lin(x)

		# iterativelly pass through the transformer layers
		for i in range(self.num_layers):
			y = self.transformer_layers[i](y, mask)

		return y
		

class EncoderLayer(nn.Module):
	"""
	Encoder layer module to be used in the Transformer Encoder.
	The input is first passed through a self-attention layer and
	then through a 2-layer feed-forward network with a GELu activation
	Residual and Normallization layers are applied right after the self-attention
	and feed-forward networks.
	"""
	def __init__(self, d_model, d_inner, heads, dropout_rate):
		super(EncoderLayer, self).__init__()

		self.MHA = MultiHeadAttention(heads, d_model, dropout_rate)
		self.DP1 = nn.Dropout(dropout_rate)
		self.LN1 = nn.LayerNorm(d_model)

		self.FFN = FeedForwardNetwork(d_model, d_inner)
		self.DP2 = nn.Dropout(dropout_rate)
		self.LN2 = nn.LayerNorm(d_model)

	def forward(self, x, mask = None):
		"""
		Args
			x: A sequence representation
				(3d float tensor) [bs x sequence_length x d_model]
			mask: Mask of the sequence
				(3d bool tensor), [bs x sequence_length x 1]
		Returns
			y: The result of the transformer encoder layer
				(3d float tensor) [bs x sequence_length x d_model]
		"""

		# multi head attention
		z = self.LN1(self.DP1(self.MHA(x, x, x, mask)) + x)

		# feed-forward network
		y = self.LN2(self.DP2(self.FFN(z)) + z)

		return y


class QaTransformerDecoder(nn.Module):
	"""
	Transformer Decoder module similar to the implementation of the decoder in
	"Attention is all you need". The input is first mapped to dimensionality d_model
	through a linear transformation and then passed through multiple decoder layers.
	"""
	def __init__(self, num_layers, d_input, d_model, d_inner, heads, dropout):
		super(QaTransformerDecoder, self).__init__()
		
		self.num_layers = num_layers

		self.lin = nn.Linear(d_input, d_model)

		self.transformer_layers = nn.ModuleList([QaDecoderLayer(d_model, d_inner, heads, dropout)
										for _ in range(self.num_layers)])
		
	def forward(self, Mp, Mq, Ea, mask_p, mask_q, mask_a = None):
		"""
		Inputs
			Mp: passage representation as output by the transformer encoder (concatinated)
				(4d float tensor) [bs x K x L x d_model]
			Mq: question representation as output by the transformer encoder
				(3d float tensor) [bs x J x d_model]
			Ea: answer embeddings from the embedder module
				(3d float tensor) [bs x T x d_emb]
			mask_p: passage mask
				(4d bool tensor) [bs x K x L x 1]
			mask_q: question mask
				(3d bool tensor) [bs x J x 1]
			mask_a: answer mask, combination of regular and not-seeing-the-future masks
				(3d bool tensor) [bs x T x T]
		Returns
			Ma: answer decoder representation
				(3d float tensor) [bs x T x d_model]
		"""

		# Map the embeddings of the answer to d_model dimensionality
		Ma = self.lin(Ea)  # (3d) [bs x T x d_model]

		# reshape passages
		bs, K, L, d_model = Mp.size()
		Mp = Mp.view(bs, K * L, d_model)
		mask_p = mask_p.view(bs, K * L, 1)

		# Pass the answer representation along with the rest of results from the encoder and ranker
		# through several transformer decoder layers
		for i in range(self.num_layers):
			Ma = self.transformer_layers[i](Mp, Mq, Ma, mask_p, mask_q, mask_a)

		return Ma


class QaDecoderLayer(nn.Module):
	"""
	Transformer decoder layer module to be used in the Transformer Decoder
	The implementation is almost the same as in "Attention is all you need" of the decoder
	but here there is another attention sublayer, making the number of sublayers 4.
	First there is a self-attention layer for the answers, then an attention layer
	between questions and answers, then another attention layer betweeen passages
	and answers, and lastly a 2-layer feed-forward network with GELu activation.
	After each sublayer, residual and normallization layers are applied.
	"""
	def __init__(self, d_model, d_inner, heads, dropout_rate):
		super(QaDecoderLayer, self).__init__()

		self.answer_attn = MultiHeadAttention(heads, d_model, dropout_rate)
		self.answer_DP = nn.Dropout(dropout_rate)
		self.answer_LN = nn.LayerNorm(d_model)

		self.question_attn = MultiHeadAttention(heads, d_model, dropout_rate)
		self.question_DP = nn.Dropout(dropout_rate)
		self.question_LN = nn.LayerNorm(d_model)

		self.passage_attn = MultiHeadAttention(heads, d_model, dropout_rate)
		self.passage_DP = nn.Dropout(dropout_rate)
		self.passage_LN = nn.LayerNorm(d_model)

		self.FFN = FeedForwardNetwork(d_model, d_inner)
		self.FFN_DP = nn.Dropout(dropout_rate)
		self.FFN_LN = nn.LayerNorm(d_model)
	
	def forward(self, Mp, Mq, Ea, mask_p, mask_q, mask_a):
		"""
		Inputs
			Mp: passage representation as output by the transformer encoder
				(3d float tensor) [batch_size x num_passages * seq_len_passages x d_model]
			Mq: question representation as output by the transformer encoder
				(3d float tensor) [batch_size x seq_len_questions x d_model]
			Ea: answer representation as output by the linear transformation in the beginining
				of the Transformer Decoder module or the previous Decoder Layer.
				(3d float tensor) [batch_size x seq_len_answers x d_model]
			mask_p: passage mask
				(3d bool tensor) [batch_size x num_passages * seq_len_passages x 1]
			mask_q: question mask
				(3d bool tensor) [batch_size x seq_len_questions x 1]
			mask_a: answer mask, combination of regular and not-seeing-the-future masks
				(3d bool tensor) [batch_size x seq_len_answers x seq_len_answers]
		Returns
			Ea4: answer representation
				(3d float tensor) [batch_size x seq_len_answers x d_model]
		"""

		# decoder self-attention
		Ea1 = self.answer_LN(self.answer_DP(self.answer_attn(Ea, Ea, Ea, mask_a)) + Ea)

		# encoder-decoder attention (question - answer)
		Ea2 = self.question_LN(self.question_DP(self.question_attn(Mq, Mq, Ea1, mask_q)) + Ea1)

		# encoder-decoder attention (passage - answer)
		Ea3 = self.passage_LN(self.passage_DP(self.passage_attn(Mp, Mp, Ea2, mask_p)) + Ea2)

		# feed forward layers
		Ea4 = self.FFN_LN(self.FFN_DP(self.FFN(Ea3)) + Ea3)

		return Ea4


class StTransformerDecoder(nn.Module):
	"""
	Transformer Decoder with multiple layers for three source sequences to be used in Style Transfer
	"""

	def __init__(self, args, d_emb):
		super(StTransformerDecoder, self).__init__()
		
		self.num_layers = args.num_layers_dec

		self.lin = nn.Linear(d_emb, args.d_model)

		self.transformer_layers = nn.ModuleList([StDecoderLayer(args.d_model, args.d_inner, args.heads, args.dropout_rate)
												for _ in range(self.num_layers)])
		
	def forward(self, Mp, Mq, Mqa, Enlg, mask_p, mask_q, mask_qa, mask_nlg = None):
		"""
		Inputs
			(See STDecoderLayer)
		Returns
			Ma: nlg_answer representation
				(3d float tensor) [batch_size x seq_len_nlg_answer x d_model]
		"""
		
		Mnlg = self.lin(Enlg)  # d_input --> d_model

		for i in range(self.num_layers):
			Mnlg = self.transformer_layers[i](Mp, Mq, Mqa, Mnlg, mask_p, mask_q, mask_qa, mask_nlg)

		return Mnlg


class StDecoderLayer(nn.Module):
	"""
	Transformer Decoder Layer with three source sequences to be used in Style Transfer
	"""

	def __init__(self, d_model, d_inner, heads, dropout_rate):
		super(StDecoderLayer, self).__init__()
		
		self.answer_attn = MultiHeadAttention(heads, d_model, dropout_rate)
		self.answer_DP = nn.Dropout(dropout_rate)
		self.answer_LN = nn.LayerNorm(d_model)

		self.qa_attn = MultiHeadAttention(heads, d_model, dropout_rate)
		self.qa_DP = nn.Dropout(dropout_rate)
		self.qa_LN = nn.LayerNorm(d_model)

		self.question_attn = MultiHeadAttention(heads, d_model, dropout_rate)
		self.question_DP = nn.Dropout(dropout_rate)
		self.question_LN = nn.LayerNorm(d_model)

		self.passage_attn = MultiHeadAttention(heads, d_model, dropout_rate)
		self.passage_DP = nn.Dropout(dropout_rate)
		self.passage_LN = nn.LayerNorm(d_model)

		self.FFN = FeedForwardNetwork(d_model, d_inner)
		self.FFN_DP = nn.Dropout(dropout_rate)
		self.FFN_LN = nn.LayerNorm(d_model)
	
	def forward(self, Mp, Mq, Mqa, Enlg, mask_p, mask_q, mask_qa, mask_nlg = None):
		"""
		Inputs
			Mp: passage representation as output by the transformer encoder
				(3d float tensor) [batch_size x  seq_len_passages x d_model]
			Mq: question representation as output by the transformer encoder
				(3d float tensor) [batch_size x seq_len_questions x d_model]
			Mqa: qa_answer representation as output by the transformer encoder
				(3d float tensor) [batch_size x seq_len_qa_answer x d_model]
			Enlg: nlg_answer representation as output by the linear transformation in the beginining
				of the Transformer Decoder module or the previous Decoder Layer.
				(3d float tensor) [batch_size x seq_len_answers x d_model]
			mask_p: passage mask
				(3d bool tensor) [batch_size x  seq_len_passages x 1]
			mask_q: question mask
				(3d bool tensor) [batch_size x seq_len_questions x 1]
			mask_qa: qa_answer mask
				(3d bool tensor) [batch_size x seq_len_qa_answerx 1]
			mask_nlg: answer mask, combination of regular and not-seeing-the-future masks
				(3d bool tensor) [batch_size x seq_len_nlg_answer x seq_len_nlg_answer]
		Returns
			Enlg4: answer representation
				(3d float tensor) [batch_size x seq_len_nlg_answer x d_model]
		"""

		Enlg0 = self.answer_LN(self.answer_DP(self.answer_attn(Enlg, Enlg, Enlg, mask_nlg)) + Enlg)
			
		Enlg1 = self.qa_LN(self.qa_DP(self.qa_attn(Mqa, Mqa, Enlg0, mask_qa)) + Enlg0)

		Enlg2 = self.question_LN(self.question_DP(self.question_attn(Mq, Mq, Enlg1, mask_q)) + Enlg1)

		Enlg3 = self.passage_LN(self.passage_DP(self.passage_attn(Mp, Mp, Enlg2, mask_p)) + Enlg2)

		Enlg4 = self.FFN_LN(self.FFN_DP(self.FFN(Enlg3)) + Enlg3)

		return Enlg4