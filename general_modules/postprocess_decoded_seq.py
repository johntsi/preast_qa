def postprocess_decoded_seq(answers):
	"""
	Corrects for some extra spaces that are created by the decode method
	of the tokenizer like in numerical strings
	example:  1, 000, 000 --> 1,000,000

	Args:
		answers: list[str]
	Returns:
		new_answers: list[str]
	"""

	new_answers = []

	for answer in answers:

		parts = answer.split(", ")
		if len(parts) > 1:

			try:
				new0 = parts[0]
				for i in range(1, len(parts)):
					if new0[-1].isnumeric() and parts[i][0].isnumeric():
						if len(parts[i]) > 3 and parts[i][3].isnumeric():
							new0 = ", ".join([new0, parts[i]])
						else:
							new0 = ",".join([new0, parts[i]])
					else:
						new0 = ", ".join([new0, parts[i]])
			except IndexError:
				print("--> IndexError:", answer)
				new0 = answer
		else:
			new0 = answer
			
		parts = new0.split(". ")
		if len(parts) > 1:
			new1 = parts[0]
			for i in range(1, len(parts)):
				try:
					if new1[-1].isnumeric() and parts[i][0].isnumeric():
						new1 = ".".join([new1, parts[i]])
					else:
						new1 = ". ".join([new1, parts[i]])
				except IndexError:
					new1 = parts[1]
		else:
			new1 = new0
				
		parts = new1.split(" : ")
		if len(parts) > 1:
			new2 = parts[0]
			for i in range(1, len(parts)):
				if new2[-1].isnumeric() and parts[i][0].isnumeric():
					new2 = ":".join([new2, parts[i]])
				else:
					new2 = " : ".join([new2, parts[i]])
		else:
			new2 = new1

		new_answers.append(new2)
						
	return new_answers