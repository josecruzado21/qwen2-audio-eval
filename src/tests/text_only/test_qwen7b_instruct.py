from io import BytesIO
from urllib.request import urlopen
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import time

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", 
                                          cache_dir = "/share/data/lang/users/ttic_31110/jcruzado/models/")
model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", device_map="auto", 
                                                           cache_dir = "/share/data/lang/users/ttic_31110/jcruzado/models/")

# conversation = [
#     {'role': 'system', 'content': 'You are a helpful assistant.'}, 
#     {"role": "user", "content": [
#         {"type": "text", "text": "What is the capital of france?"},
#     ]}]
conversation = [{'role': 'system', 'content': 'You are a helpful assistant. Answer the question with the correct letter'}, 
                {'role': 'user', 'content': 
                 [{'type': 'text', 'text': 'Question: If a psychologist acts as both a fact witness for the plaintiff and an expert witness for the court in a criminal trial, she has acted:\nA. unethically by accepting dual roles.\nB. ethically as long as she did not have a prior relationship with the plaintiff.\nC. ethically as long as she clarifies her roles with all parties.\nD. ethically as long as she obtains a waiver from the court.\nAnswer:'}
                  ]}]
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
inputs = processor(text=text, return_tensors="pt", padding=True)
# inputs.input_ids = inputs.input_ids.to("cuda")
inputs = inputs.to("cuda")
start = time.time()
generate_ids = model.generate(**inputs, max_new_tokens=16)
print("Time to generate answer:", time.time() - start, "seconds")
generate_ids = generate_ids[:, inputs.input_ids.size(1):]
response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print("Answer:", response)