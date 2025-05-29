# Unpacking Degradation in Multimodal LLMs: The Case of Qwen2-Audio

This project investigates the performance and degradation of the Qwen2-Audio large language model (LLM) in multimodal (audio and text) tasks, focusing on benchmarks such as MMLU, Librispeech and timbre range classification. It includes data preprocessing, model inference, fine-tuning, and benchmarking scripts.

## Project Structure

```
.
├── data/                # Datasets, audio files, and CSVs for experiments
├── notebooks/           # Jupyter notebooks for analysis and experiments
├── src/
│   ├── 01_preprocessing/    # Scripts for downloading models and preparing datasets
│   ├── 02_benchmarking/     # Benchmarking scripts for ASR, MMLU, and timbre range
│   ├── 03_postprocessing/   # Postprocessing scripts for MMLU benchmarking
│   ├── 04_finetunning/      # Fine-tuning scripts and outputs
│   └── tests/               # Test scripts for model inference
├── utils/               # Utility functions (e.g., audio resampling, formatting)
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## Main Features

- **Data Preparation:** Download and preprocess MMLU and timbre range datasets ([src/01_preprocessing](src/01_preprocessing/)).
- **Model Inference:** Run Qwen2-Audio and Whisper models for ASR, MMLU, and timbre range tasks ([src/02_benchmarking](src/02_benchmarking/)).
- **Fine-tuning:** Scripts for LoRA-based fine-tuning of Qwen2-Audio on custom audio tasks ([src/04_finetunning](src/04_finetunning/)).
- **Benchmarking & Analysis:** Evaluate model performance using accuracy and WER metrics ([notebooks/](notebooks/), [src/02_benchmarking/analysis](src/02_benchmarking/analysis/)).
- **Postprocessing:** Clean and analyze model outputs for benchmarking ([src/03_postprocessing](src/03_postprocessing/)).
- **Utilities:** Audio resampling, question formatting, and other helpers ([utils/utils.py](utils/utils.py)).

## Notebooks

- `notebooks/` contains exploratory and analysis notebooks for ASR, MMLU, and timbre range tasks.

## Notes

- Some scripts require GPU or specific hardware/software environments.
- Model checkpoints and large datasets are not included in the repository.
- For OpenAI TTS and GPT-4o, ensure your API key is set in `.env`.