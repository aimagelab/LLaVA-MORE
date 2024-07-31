import sys
import os
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("."))
from llava.train.train import train

def is_debug():
    return int(os.environ.get('DEBUG', 0))

if __name__ == "__main__":
    if is_debug():
        train(attn_implementation=None)
    else:
        train(attn_implementation="flash_attention_2")
