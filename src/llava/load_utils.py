# put in this file the function to load the model and the tokenizer, 
# starting from the 'model_type'

import torch
import transformers
from typing import Dict
from src.llava.model import *


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def get_model(model_args, attn_implementation, training_args, bnb_model_from_pretrained_args):
    if model_args.model_architecture == "mpt":
        config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
        config.attn_config['attn_impl'] = training_args.mpt_attn_impl
        model = LlavaMptForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )
    elif model_args.model_architecture == "phi_4":
        model = LlavaPhiForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args
        )
    elif model_args.model_architecture == "gemma_2":
        model = LlavaGemmaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation='eager',
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args
        )
    else:
        model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
    
    model.config.model_architecture = model_args.model_architecture
    return model

def get_tokenizer(model_args, model, training_args):
    if model_args.model_architecture == 'mpt':
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
    
    # use the last option
    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token

    elif model_args.model_architecture == 'gemma_2':
        tokenizer.pad_token_id = 0
    elif model_args.model_architecture == "phi_4":
        tokenizer.pad_token = tokenizer.unk_token

    else:
        # for all the version of llama 3 not expand the dictionary with unk token
        # it can create problem in stage two when importing the configuration value of the vocab size
        if "llama_3" not in model_args.model_architecture:
            if tokenizer.unk_token is None:
                print("resize embedding dimesion")
                smart_tokenizer_and_embedding_resize(
                    special_tokens_dict=dict(unk_token="[UNK]"),
                    tokenizer=tokenizer,
                    model=model,
                )
        # select the correct PAD token for llama_3_1 and llama_3
        if model_args.model_architecture == "llama_3_1":
            print(f"pad token: {training_args.llm_pad_token}")
            if training_args.llm_pad_token == 'end_of_text':
                tokenizer.pad_token_id= 128001
            elif training_args.llm_pad_token == 'eot':
                tokenizer.pad_token_id= 128009
            elif training_args.llm_pad_token == 'pad':
                tokenizer.pad_token_id= 128004
            else:
                raise ValueError(f"Unknown llm_pad_token")
            
        elif model_args.model_architecture == "llama_3":
            if training_args.llm_pad_token == 'eos':
                tokenizer.pad_token = tokenizer.eos_token
            elif training_args.llm_pad_token == 'pad':
                tokenizer.pad_token_id= 128003
            else:
                tokenizer.pad_token = tokenizer.unk_token

        else:
            tokenizer.pad_token = tokenizer.unk_token


    return tokenizer
