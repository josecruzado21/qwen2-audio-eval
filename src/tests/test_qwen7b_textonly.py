from io import BytesIO
from urllib.request import urlopen
import librosa
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B" ,trust_remote_code=True, cached_dir = "../../models")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B" ,trust_remote_code=True, cached_dir = "../../models")
prompt = "What is the capital of France?"
inputs = processor(text=prompt, return_tensors="pt")
generated_ids = model.generate(**inputs, max_length=256)
generated_ids2 = generated_ids[:, inputs.input_ids.size(1):]
response = processor.batch_decode(generated_ids2, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(response)