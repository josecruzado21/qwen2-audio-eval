def formatted_question_MMLU(row):
    formatted_question = f"Question: {row['question']}\n"
    f"A. {row['choices'][0]}\n"
    f"B. {row['choices'][1]}\n"
    f"C. {row['choices'][2]}\n"
    f"D. {row['choices'][3]}\n"
    f"Answer:"
    return formatted_question