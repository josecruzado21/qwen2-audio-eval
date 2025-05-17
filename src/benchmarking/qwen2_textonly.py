import os
import sys
current_file_path = os.path.abspath(__file__)
root_path = os.path.abspath(os.path.join(current_file_path, '../../..'))
sys.path.append(root_path)
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from utils.utils import formatted_question_MMLU
from pathlib import Path
import pandas as pd
script_dir = Path(__file__).resolve()

def qwen2_textonly_chat_prompt(row):
    formatted_question = formatted_question_MMLU(row)
    prompt = "Return ONLY the letter of the correct answer (A, B, C, or D). Your answer should be one character long: A, B, C, or D \n" + formatted_question
    return prompt

def qwen2_textonly_inference(MMLU_data):
    output_path = script_dir.parent / ".." / ".." / "data" / "MMLU.csv"
    col_name = "qwen2_textonly_response"
    if col_name not in MMLU_data.columns:
        MMLU_data[col_name] = ""
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B-Chat", 
                                            trust_remote_code=True,
                                            cache_dir = "/share/data/lang/users/ttic_31110/jcruzado/models/")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen-7B-Chat",
        trust_remote_code=True,
        cache_dir = "/share/data/lang/users/ttic_31110/jcruzado/models/"
    )
    for idx, row in tqdm(MMLU_data.iterrows()):
        if (row["qwen2audio_textonly_response"] == "") | pd.isna(row["qwen2audio_textonly_response"]):
            prompt = qwen2_textonly_chat_prompt(row)
            response, history = model.chat(tokenizer, prompt, history=None, max_new_tokens=8)
            MMLU_data.at[idx, col_name] = response
            MMLU_data.to_csv(output_path, index=False)      

if __name__ == "__main__":
    MMLU_data = pd.read_csv(os.path.join(root_path, "data", "MMLU.csv"))
    qwen2_textonly_inference(MMLU_data)