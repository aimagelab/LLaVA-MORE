import torch
import transformers
from transformers import AutoModel, AutoTokenizer, LlamaForCausalLM

model_path = '/leonardo_scratch/large/userexternal/fcocchi0/rag_mlmm/hf_models/llama_3_1/fcocchi_srv/models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B/snapshots/24ae87a9c340aa4207dd46509414c019998e0161'

model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)
device = model.device

prompt = 'If you have 4 apple and a kid eat 2 of them. How many apple do you have? Please answer with a single word.'

input_ids = tokenizer(prompt, padding=True, 
                        truncation=True, max_length=1024,
                        return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(**input_ids, temperature=0.6, do_sample=True, max_new_tokens=20)
    decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

decoded_text
print('done!')
