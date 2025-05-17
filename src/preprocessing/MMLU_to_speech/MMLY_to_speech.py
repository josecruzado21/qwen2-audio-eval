import os
import sys
current_file_path = os.path.abspath(__file__)
root_path = os.path.abspath(os.path.join(current_file_path, '../../../..'))
sys.path.append(root_path)
from dotenv import load_dotenv
load_dotenv(dotenv_path = root_path)
from utils.utils import formatted_question_MMLU_speech
from openai import OpenAI
from tqdm import tqdm
import pandas as pd

def openai_tts_MMLU(client, text, question_index):
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text,
        instructions="Speak like reading a question with 4 alternatives in an exam.",
    ) as response:
        response.stream_to_file(os.path.join(root_path, "data", "audios", f"question_{question_index}.wav"))

def convert_MMLU_to_audio(MMLU_data):
    client = OpenAI()
    col_name = "speech_version"
    if col_name not in tqdm(MMLU_data.columns):
        MMLU_data[col_name] = 0
    for idx, row in MMLU_data.iterrows():
        if MMLU_data[col_name] == 0:
            text_to_convert = formatted_question_MMLU_speech(row)
            openai_tts_MMLU(client, text_to_convert, idx)
            MMLU_data.at[idx, col_name] = 1
            MMLU_data.to_csv(os.path.join(root_path, "data", "MMLU.csv"), index=False)

if __name__ == "__main__":
    MMLU_data = pd.read_csv(os.path.join(root_path, "data", "MMLU.csv"))
    convert_MMLU_to_audio(MMLU_data)

