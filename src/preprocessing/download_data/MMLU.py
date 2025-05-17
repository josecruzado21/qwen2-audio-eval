from datasets import load_dataset
import pandas as pd
from pathlib import Path
script_dir = Path(__file__).resolve()

def download_and_sample_MMLU(n_questions = 500):
    output_path = script_dir.parent / ".." / ".." / ".." / "data" / "MMLU.csv"
    ds = load_dataset("cais/mmlu", "all")
    df_MMLU = pd.DataFrame(ds["test"])
    df_MMLU["prompt_length"] = df_MMLU.question.str.len() + df_MMLU.choices.str.len()
    df_MMLU = df_MMLU.sort_values(by="prompt_length", ignore_index=True)
    df_MMLU = df_MMLU.loc[:n_questions]
    df_MMLU.drop(columns=["prompt_length"], inplace=True)
    df_MMLU.to_csv(output_path, index=False)
    print("MMLU saved")

if __name__ == "__main__":
    download_and_sample_MMLU()