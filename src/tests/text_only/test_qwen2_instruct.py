from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

# Model names: "Qwen/Qwen-7B-Chat", "Qwen/Qwen-14B-Chat"
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B-Chat", 
                                          trust_remote_code=True,
                                          cache_dir = "/share/data/lang/users/ttic_31110/jcruzado/models/")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-7B-Chat",
    trust_remote_code=True,
    cache_dir = "/share/data/lang/users/ttic_31110/jcruzado/models/"
)
response, history = model.chat(tokenizer, "hello", history=None)
print(response)