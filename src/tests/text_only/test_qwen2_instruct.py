from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import time

# Model names: "Qwen/Qwen-7B-Chat", "Qwen/Qwen-14B-Chat"
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B-Chat", 
                                          trust_remote_code=True,
                                          cache_dir = "/share/data/lang/users/ttic_31110/jcruzado/models/")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-7B-Chat",
    trust_remote_code=True,
    cache_dir = "/share/data/lang/users/ttic_31110/jcruzado/models/"
)

start = time.time()
response, history = model.chat(tokenizer,"Return ONLY the letter of the correct answer (A, B, C, or D). Your answer should be one character long A, B, C, or D \n"
    "Question: If a psychologist acts as both a fact witness for the plaintiff and an expert witness for the court in a criminal trial, she has acted:\n"                         
    "A. unethically by accepting dual roles.\n"
    "B. ethically as long as she did not have a prior relationship with the plaintiff.\n"
    "C. ethically as long as she clarifies her roles with all parties.\n"
    "D. ethically as long as she obtains a waiver from the court.\n"
    "Answer:", 
    history=None, 
    max_new_tokens=8)
end = time.time()
print("Time to generate:", end - start)
print("History", history)
print()
print("Response:", response)