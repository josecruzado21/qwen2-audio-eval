import ast
def formatted_question_MMLU(row):
    choices = lst = ast.literal_eval(row['choices'])
    formatted_question = f"Question: {row['question']}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer:"
    return formatted_question