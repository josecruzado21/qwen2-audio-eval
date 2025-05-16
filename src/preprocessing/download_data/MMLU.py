from datasets import load_dataset
import pandas as pd
from pathlib import Path
script_dir = Path(__file__).resolve()

def download_and_sample_MMLU(n_questions = 500,
                             random_state = 42):
    output_path = script_dir.parent / ".." / ".." / ".." / "data" / "MMLU.csv"
    ds = load_dataset("cais/mmlu", "all")
    df_MMLU = pd.DataFrame(ds["test"])
    sampled_MMLU = df_MMLU.sample(n=n_questions, 
                                  random_state = random_state)
    sampled_MMLU.to_csv(output_path, index=False)

if __name__ == "__main__":
    download_and_sample_MMLU()