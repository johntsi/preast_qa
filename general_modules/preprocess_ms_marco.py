import json
import argparse
from tqdm import tqdm

from my_tokenizer import construct_tokenizer

NOISE_THRESHOLD = 8


def whitespace_clean(text):
	text = text.strip()
	tokens = text.split()
	return " ".join(tokens)


def get_hashtag_spans(tokens):
	"""
	Finds the spans (start, end) of subtokes in a list of tokens

	Args:
		tokens: list[str]
	Returns:
		spans: list[tuple[int]]
	"""

	is_part = ["##" in t for t in tokens]
	
	spans = []
	pos_end = -1
	for pos_start, t in enumerate(is_part):
		if pos_start <= pos_end:
			continue
		if t:
			last_pos = len(is_part[pos_start:]) - 1
			for j, t_end in enumerate(is_part[pos_start:]):
				if not t_end:
					pos_end = pos_start + j
					break
				if j == last_pos:
					pos_end = pos_start + j + 1
			spans.append((pos_start, pos_end))
	
	return spans


def clean_noise(tokens, spans):
	"""
	Removes noisy subtokens

	Args:
		tokens: list[str]
		spans: list[tuple[int]]
	Returns:
		tokens: list[str]
	"""
	
	correction = 0
	for span in spans:
		start, end = span
		start -= correction
		end -= correction
		if end - start > NOISE_THRESHOLD:
			del tokens[start - 1: end]
			correction += end - start + 1
				
	return tokens


def clean_repeating_tokens(repeating_tokens, tokens):
	"""
	Reduces long sequences of repeating tokens (?????????????? --> ?)

	Args:
		repeating_tokens: str
		tokens: list[str]
	Returns:
		tokens: list[str]
	"""

	# find the span of the repeating tokens
	spans = []
	pos_end = -1
	for pos_start, t_start in enumerate(tokens):
		if pos_start <= pos_end:
			continue
		if t_start == repeating_tokens:
			last_pos = len(tokens[pos_start:]) - 1
			for j, t_end in enumerate(tokens[pos_start:]):
				if t_end != repeating_tokens:
					pos_end = pos_start + j
					break
				if j == last_pos:
					pos_end = pos_start + j + 1
			spans.append((pos_start, pos_end))
	

	if repeating_tokens == "_" or repeating_tokens == ".":
		num = 3
	else:
		num = 1

	correction = 0
	for span in spans:
		start, end = span
		start -= correction
		end -= correction
		span_length = end - start
		if span_length > num:
			tokens[start: end] = [repeating_tokens] * num
			correction += span_length - num

	return tokens


def process_query(query, tokenizer):
	"""
	cleans the query from repeating and noisy sequences

	Args:
		query: str
		tokenizer: transformer object
	Returns:
		cleaned_query: str
	"""

	query = whitespace_clean(query)

	# tokenize
	tokens, _ = tokenizer._unk_tokenize(query)

	# find repeating tokens
	repeating_tokens = []
	n_tokens = len(tokens)
	for token in set(tokens):
		seq = [token] * 3
		for j in range(n_tokens - 3):
			if seq == tokens[j: j + 3]:
				repeating_tokens.append(token)
				break

	# clean the repeating tokens
	has_repeats = repeating_tokens != []
	if has_repeats:
		for repeating_token in repeating_tokens:
			tokens = clean_repeating_tokens(repeating_token, tokens)

	# find the span of subtokens
	spans = get_hashtag_spans(tokens)

	# clean the potentially noisy subtokens
	is_noisy = any([span[1] - span[0] > NOISE_THRESHOLD for span in spans])
	if is_noisy:
		tokens = clean_noise(tokens, spans)

	# convert the tokens back to a string
	if has_repeats or is_noisy:
		cleaned_query = tokenizer.convert_tokens_to_string(tokens)
	else:
		cleaned_query = query

	return cleaned_query


def clean_dupl_passages(passages):
	"""
	Removes duplicate passages in an example

	Args:
		passages: list
			with size K
			with each element being a dict with keys ("passage_text", "is_selected")
		new_passages: list
			with size k <= K
			with each element being a dict with keys ("passage_text", "is_selected")
	"""

	new_passages = []
	new_passages_text = []
	
	for i, passage in enumerate(passages):
		passage["passage_text"] = whitespace_clean(passage["passage_text"])
		if passage["passage_text"] not in new_passages_text:
			new_passages_text.append(passage["passage_text"])
			new_passages.append(passage)
		else:
			if "is_selected" in passage.keys():
				if passage["is_selected"] == 1:
					duplicate_idx = new_passages_text.index(passage["passage_text"])
					if new_passages[duplicate_idx]["is_selected"] == 0:
						del new_passages[duplicate_idx]
						del new_passages_text[duplicate_idx]
						new_passages_text.append(passage["passage_text"])
						new_passages.append(passage)

	return new_passages


def clean_dupl_answers(answers):
	"""
	Removes duplicated answers from an answerable example

	Args:
		answers: list[str]
	Returns:
		answers: list[str]
	"""

	new_answers = []
	new_answers_lower = []

	answers = [whitespace_clean(a) for a in answers]

	if len(answers) > 1:
		for i, answer in enumerate(answers):
			if answer.lower() not in new_answers_lower:
				new_answers.append(answer)
				new_answers_lower.append(answer.lower())

		return new_answers
	else:
		return answers


def has_conflicting_info(passages, qa_answers, nlg_answers):
	"""
	Checks whether an example has conflicting information
	regarding its answerability

	Args:
		passages: list[{"is_selected": int, "passage_text": str}]
		qa_answers: list[str]
		nlg_answers: list[str]
	Returns:
		bool
	"""

	has_rel_passage = sum([p_info["is_selected"] for p_info in passages]) != 0
	qa_avail = (qa_answers != ['No Answer Present.']) and (qa_answers != [""])
	nlg_avail = nlg_answers != "[]"

	# there is at least one nlg anser but no qa answer
	if nlg_avail and not qa_avail:
		return True

	# there is at least one answer but no relevant passage
	elif qa_avail and not has_rel_passage:
		return True

	# there is at least one relevant passage but no answer is available
	elif has_rel_passage and not qa_avail:
		return True

	else:
		return False


def preprocess_ms_marco(source_file_path):

	print("Creating a clean version of the dataset ...")

	tokenizer = construct_tokenizer()

	# load data
	with open(source_file_path, "r") as read_file:
		data = json.load(read_file)

	num_cases_passages = 0
	num_cases_qa = 0
	num_cases_nlg = 0
	num_cases_query = 0
	num_cases_example = 0

	indices = list(data["passages"].keys())

	for idx in tqdm(indices):

		# remove noisy example
		if has_conflicting_info(data["passages"][idx], data["answers"][idx], data["wellFormedAnswers"][idx]):
			for k in data.keys():
				del data[k][idx]
			num_cases_example += 1
			continue

		# remove duplicate passages
		n_before = len(data["passages"][idx])
		data["passages"][idx] = clean_dupl_passages(data["passages"][idx])
		num_cases_passages += n_before - len(data["passages"][idx])

		# clean query from noisy and repeating symbols
		cleaned_query = process_query(data["query"][idx], tokenizer)
		num_cases_query += data["query"][idx] != cleaned_query
		data["query"][idx] = cleaned_query

		# remove duplicate qa answers
		qa_avail = (data["answers"][idx] != ['No Answer Present.']) and (data["answers"][idx] != [""])
		if qa_avail:
			n_before = len(data["answers"][idx])
			data["answers"][idx] = clean_dupl_answers(data["answers"][idx])
			num_cases_qa += n_before - len(data["answers"][idx])

		# remove duplicate nlg answers
		nlg_avail = data["wellFormedAnswers"][idx] != "[]"
		if nlg_avail:
			n_before = len(data["wellFormedAnswers"][idx])
			data["wellFormedAnswers"][idx] = clean_dupl_answers(data["wellFormedAnswers"][idx])
			num_cases_nlg += n_before - len(data["wellFormedAnswers"][idx])

	print(f"Removed {num_cases_example} noisy examples")
	print(f"Removed {num_cases_passages} duplicate passages from the examples")
	print(f"Removed {num_cases_qa} duplicate qa answers from the examples")
	print(f"Removed {num_cases_nlg} duplicate nlg answers from the examples")
	print(f"Processed and cleaned {num_cases_query} queries")

	output_file_path = source_file_path.replace(".json", "_cleaned.json")

	print(f"Saved cleaned data in {output_file_path}")
	with open(output_file_path, "w") as f:
		json.dump(data, f)


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("--source_file_path", required = True, type = str)
	args = parser.parse_args()

	preprocess_ms_marco(args.source_file_path)