import ast
def formatted_question_MMLU(row):
    choices = ast.literal_eval(row['choices'])
    formatted_question = f"Question: {row['question']}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer:"
    return formatted_question

def formatted_question_MMLU_speech(row):
    choices = ast.literal_eval(row['choices'])
    formatted_question = f"Answer the following question only with the letter of the correct alternative. Your answer should be only A, B, C or D. Question: {row['question']}. Alternative A. {choices[0]}. Alternative B. {choices[1]}. Alternative C. {choices[2]}. Alternative D. {choices[3]}. Answer with the right alternative number:"
    return formatted_question