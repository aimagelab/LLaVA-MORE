import os
from datasets import load_dataset

# print(f'Current HF cache directory: {os.environ["HF_HOME"]}')
print(f'Current HF cache directory: {os.environ["HF_DATASETS_CACHE"]}')

# TODO: next datasets to be downloaded "mmbench", "mmmu_test", "vqav2"
data_models= ["ScienceQA"]
pre_append= 'lmms-lab/'
ok = ["ScienceQA-FULL"]

for data_model, o in zip(data_models, ok):
    data= pre_append + data_model
    dataset = load_dataset(data, o)
    print(f"Dataset {data_model} loaded successfully")
print("All datasets loaded successfully")
