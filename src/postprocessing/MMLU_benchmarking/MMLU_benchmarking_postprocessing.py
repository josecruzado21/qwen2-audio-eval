import pandas as pd
import os
import sys
current_file_path = os.path.abspath(__file__)
root_path = os.path.abspath(os.path.join(current_file_path, '../../../..'))
sys.path.append(root_path)
from src.benchmarking.qwen2_textonly import qwen2_textonly_chat_prompt
from openai import OpenAI
from tqdm import tqdm
from pathlib import Path
script_dir = Path(__file__).resolve()
from dotenv import load_dotenv
load_dotenv(dotenv_path = os.path.join(root_path, ".env"))

def clean_qwen2_MMLU_response_clean(MMLU_df):
    output_path = script_dir.parent / ".." / ".." / ".." / "data" / "MMLU.csv"
    col_name = "qwen2_textonly_response_clean"
    if col_name not in MMLU_df.columns:
        MMLU_df[col_name] = ""
    client = OpenAI()
    for idx, row in tqdm(MMLU_df.iterrows()):
        if ((row[col_name] == "") | pd.isna(row[col_name])) & (row["qwen2audio_textonly_response"]!="" | ~pd.isna(row["qwen2audio_textonly_response"])):
            original_prompt = qwen2_textonly_chat_prompt(row)
            start_index = original_prompt.find("\nA.")
            original_prompt = original_prompt[start_index+1:]
            original_response = row["qwen2_textonly_response"] 
            prompt = f"I will give the alternatives I gave to another LLM and its answer. Your task is to tell me which alternative the model is referring to, if it is A, B, C or D. Your answer should only be a letter or output NA if you cannot identify the alternative the other LLM was referring to\nalternatives: {original_prompt}\nmodel answer: {original_response}"
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            MMLU_df.at[idx, col_name] = response.choices[0].message.content
            MMLU_df.to_csv(output_path, index=False)

# def clean_qwen2audio_MMLU_response_clean(MMLU_df):
#     output_path = script_dir.parent / ".." / ".." / ".." / "data" / "MMLU.csv"
#     col_name = "qwen2audio_audioonly_response_clean"
#     if col_name not in MMLU_df.columns:
#         MMLU_df[col_name] = ""
#     client = OpenAI()
#     for idx, row in tqdm(MMLU_df.iterrows()):
#         if (row[col_name] == "") | pd.isna(row[col_name]):
#             original_prompt = qwen2_textonly_chat_prompt(row)
#             start_index = original_prompt.find("\nA.")
#             original_prompt = original_prompt[start_index+1:]
#             original_response = row["qwen2audio_audioonly_response"] 
#             prompt = f"I will give the alternatives I gave to another LLM and its answer. Your task is to tell me which alternative the model is referring to, if it is A, B, C or D. Your answer should only be a letter or output NA if you cannot identify the alternative the other LLM was referring to\nalternatives: {original_prompt}\nmodel answer: {original_response}"
#             response = client.chat.completions.create(
#                 model="gpt-4o-mini",
#                 messages=[
#                     {"role": "system", "content": "You are a helpful assistant."},
#                     {"role": "user", "content": prompt}
#                 ]
#             )
#             MMLU_df.at[idx, col_name] = response.choices[0].message.content
#             MMLU_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    MMLU_df = pd.read_csv(os.path.join(root_path, "data", "MMLU.csv"))
    clean_qwen2_MMLU_response_clean(MMLU_df)