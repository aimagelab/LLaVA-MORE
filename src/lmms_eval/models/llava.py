import os
import gc
import random
import numpy as np
import faiss
import ujson
import torch

from transformers import AutoModel, AutoImageProcessor, CLIPImageProcessor
torch.backends.cuda.matmul.allow_tf32 = True

import logging
import copy
from tqdm import tqdm
from datetime import timedelta

from src.lmms_eval import utils
from src.lmms_eval.api.instance import Instance
from src.lmms_eval.api.model import lmms
from src.lmms_eval.api.registry import register_model
from src.lmms_eval.utils import stop_sequences_criteria

from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from typing import List, Optional, Union, Tuple
import warnings
from src.llava.model import *
from transformers import GemmaTokenizer

warnings.filterwarnings("ignore")

eval_logger = logging.getLogger("lmms-eval")

# try:
eval_logger.info("Importing LLaVA ...")
from src.llava.model.builder import load_pretrained_model
from src.llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from src.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from src.llava.conversation import conv_templates, SeparatorStyle

IMAGE_TOKEN = f"{DEFAULT_IMAGE_TOKEN}\n\n"

# except ImportError:
#     eval_logger.error("LLaVA is not installed. Please install LLaVA to use this model.")

if torch.__version__ > "2.1.2":
    best_fit_attn_implementation = "sdpa"
else:
    best_fit_attn_implementation = "eager"


# TODO: REMOVE
class Retriever:
    def __init__(self, args):
        self.args = args
        print("Loading retrieval index - path hard coded for encyclopedic visual")
        if args.use_eva_retrieve:
            args.index_path = '/leonardo_scratch/large/userexternal/fcocchi0/rag_mlmm/index/visualRAG/encyclopedic_eva_clip/retrieval_index_image_noblack_l2'
            args.index_path_json = '/leonardo_scratch/large/userexternal/fcocchi0/rag_mlmm/index/visualRAG/encyclopedic_eva_clip/retrieval_index_json_image_l2'
        elif args.use_clip_retrieve:
            args.index_path = '/leonardo_scratch/large/userexternal/fcocchi0/rag_mlmm/index/visualRAG/encyclopedic_clip/retrieval_index_image_noblack_l2'
            args.index_path_json = '/leonardo_scratch/large/userexternal/fcocchi0/rag_mlmm/index/visualRAG/encyclopedic_clip/retrieval_index_json_image_l2'
        
        self.index = faiss.read_index(
            os.path.join(args.index_path, 'knn.index'))
        self.values = ujson.load(
            open(os.path.join(args.index_path_json, 'knn.json'), 'r'))
        print("Done loading retrieval index")

    def retrieve(self, query, k):
        if isinstance(query, torch.Tensor):
            query = query.detach().cpu().numpy()
        query = query.astype(np.float32)
        _, indexes = self.index.search(query, k=100)
        chosen_k = indexes[0, :k].tolist()
        return [self.values[k][0] for k in chosen_k]


@register_model("llava")
class Llava(lmms):
    """
    Llava Model
    """

    def __init__(
        self,
        pretrained: str = "liuhaotian/llava-v1.5-7b",
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        trust_remote_code: Optional[bool] = False,
        revision=None,
        model_name=None,
        attn_implementation=best_fit_attn_implementation,
        use_flash_attention_2=True,
        device_map="auto",
        conv_template="vicuna_v1",
        use_cache=True,
        truncate_context=False,  # whether to truncate the context in generation, set it False for LLaVA-1.6
        customized_config=None,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now - extract the information
        mlp_path= kwargs.pop("mlp_path", None)
        
        self.reflectiva_kb = kwargs.pop("reflectiva_kb", None)
        self.use_clip_retrieve = kwargs.pop("use_clip_retrieve", None)
        self.use_eva_retrieve = kwargs.pop("use_eva_retrieve", None)
        self.entity_k = kwargs.pop("entity_k", 4)
        self.model_architecture = kwargs.pop("model_architecture", None)
        self.conv_template = kwargs.pop("conv_mode", None) # define the conversation template
        
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1 and device_map == "":
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map
        
        # TODO: REMOVE all the reflectiva code block
        # if self.reflectiva_kb:
        #     print("Loading the retriever model ...")
        #     self.retriever = Retriever(self)

        #     if self.use_clip_retrieve:
        #         self.clip_name_or_path = '/leonardo_scratch/large/userexternal/fcocchi0/rag_mlmm/hf_models/openai/clip-vit-large-patch14-336/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1'
        #         self.clip = AutoModel.from_pretrained(self.clip_name_or_path).to(device)
        #         self.clip.to(torch.float16)
        #         self.clip.to(device)
        #         self.clip.eval()
        #         self.clip_preprocessor = AutoImageProcessor.from_pretrained(self.clip_name_or_path)
        #     elif self.use_eva_retrieve:
        #         self.clip_name_or_path = "/leonardo_scratch/large/userexternal/fcocchi0/rag_mlmm/hf_models/BAAI/EVA-CLIP-8B/models--BAAI--EVA-CLIP-8B/snapshots/0e4dca944e8ece27eb9dfe4a488c0ed0c4644fc9"
        #         self.clip_processor = "/leonardo_scratch/large/userexternal/fcocchi0/rag_mlmm/hf_models/openai/clip-vit-large-patch14/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41"
        #         self.clip = AutoModel.from_pretrained(self.clip_name_or_path, torch_dtype=torch.float16, trust_remote_code=True).to(device).eval()
        #         self.clip_preprocessor = CLIPImageProcessor.from_pretrained(self.clip_processor)
            
        #     wikipedia_path = '/leonardo_scratch/large/userexternal/fcocchi0/rag_mlmm/dataset/encyclopedic_vqa/encyclopedic_kb_wiki_dict.json'
        #     with open(wikipedia_path, 'r') as f:
        #         self.wikipedia = ujson.load(f)

        llava_model_args = {}
        llava_model_args["model_architecture"] = self.model_architecture
        llava_model_args["attn_implementation"] = attn_implementation
        if customized_config:
            llava_model_args["customized_config"] = customized_config
        if attn_implementation is not None:
            llava_model_args["attn_implementation"] = attn_implementation
        if "use_flash_attention_2" in kwargs:
            llava_model_args["use_flash_attention_2"] = kwargs["use_flash_attention_2"]

        model_name = model_name if model_name is not None else get_model_name_from_path(pretrained)
        model_name= pretrained # our fix: to use checkpoint saved on disk
        try:
            # Try to load the model with the multimodal argument
            self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(pretrained, None, model_name, device_map=self.device, mlp_path=mlp_path, **llava_model_args)
        except TypeError:
            # for older versions of LLaVA that don't have multimodal and attn_implementation arguments
            llava_model_args.pop("multimodal", None)
            llava_model_args.pop("attn_implementation", None)
            self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(pretrained, None, model_name, device_map=self.device_map, **llava_model_args)

        self._config = self._model.config
        self.model.eval()
        self.model.tie_weights()
        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        self.truncate_context = truncate_context

        # TODO: REMOVE change the conversation template accordily with backbone and select the correct one (self.conv_mode)
        # if 'llama_3_1' in pretrained or 'LLaMA_8B_31' in pretrained or 'LLaMA_3.1' in pretrained:
        #     self.conv_template = 'llama_3_1'
        # else:
        #     self.conv_template = conv_template

        # if 'llama3' in pretrained:
        #     self.conv_template = 'llama_3'
        # if 'gemma' in pretrained.lower():
        #     self.conv_template = 'gemma_2'
        #     # torch._inductor.config.triton.cudagraph_skip_dynamic_graphs=True
        # if 'phi' in pretrained.lower():
        #     self.conv_template = 'phi_4'
        # assert self.batch_size_per_gpu == 1, "Llava currently does not support batched generation. See https://github.com/haotian-liu/LLaVA/issues/754. HF Llava also has this issue."
        if accelerator.num_processes > 1 and device_map == "":
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")

            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
            self._rank = 0
            self._word_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        """ """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # TODO
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, doc_to_target, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            # encode, pad, and truncate contexts for this batch
            if type(doc_to_target) == str:
                continuation = doc_to_target
            else:
                continuation = doc_to_target(self.task_dict[task][split][doc_id])
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            if visuals:
                image = process_images(visuals, self._image_processor, self._config)
                if type(image) is list:
                    image = [_image.to(dtype=torch.float16, device=self.device) for _image in image]
                else:
                    image = image.to(dtype=torch.float16, device=self.device)
            else:
                image = None

            prompts_input = contexts[0] if isinstance(contexts, list) else contexts

            if image is not None and len(image) != 0 and DEFAULT_IMAGE_TOKEN not in prompts_input:
                """
                Three senarios:
                1. No image, and there for, no image token should be added.
                2. image token is already specified in the context, so we don't need to add it.
                3. image token is not specified in the context and there is image inputs, so we need to add it. In this case, we add the image token at the beginning of the context and add a new line.
                """
                image_tokens = [DEFAULT_IMAGE_TOKEN] * len(visuals)
                image_tokens = " ".join(image_tokens)
                prompts_input = image_tokens + "\n" + (contexts[0] if isinstance(contexts, list) else contexts)

            # This is much safer for llama3, as we now have some object type in it
            if "llama_3" in self.conv_template:
                conv = copy.deepcopy(conv_templates[self.conv_template])
            else:
                conv = conv_templates[self.conv_template].copy()
            conv.append_message(conv.roles[0], prompts_input)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            contxt_id = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
            # Add the answer of the second role
            conv.messages[1][1] = continuation

            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
            labels = input_ids.clone()
            # Context part no need to calculate for loss
            labels[0, : contxt_id.shape[1]] = -100
            with torch.inference_mode():
                outputs = self.model(input_ids=input_ids, labels=labels, images=image, use_cache=True)
            loss = outputs["loss"]
            # loss = torch.exp(loss)
            logits = outputs["logits"]
            greedy_tokens = logits.argmax(dim=-1)
            cont_toks = input_ids[:, contxt_id.shape[1] :]  # [1, seq]
            greedy_tokens = greedy_tokens[:, contxt_id.shape[1] : input_ids.shape[1]]  # [1, seq]
            max_equal = (greedy_tokens == cont_toks).all()
            res.append((float(loss.item()), bool(max_equal)))
            pbar.update(1)
        pbar.close()
        return res

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        
        def _concat_paragraph(paragraphs_list, question, sections, conv_mode, tokenizer):
            qs = IMAGE_TOKEN + question
            conv_relevant = conv_mode
            conv_relevant.append_message(conv_relevant.roles[0], qs)
            qs = '[Retrieval]'
            conv_relevant.append_message(conv_relevant.roles[1], qs)
            qs = 'Consider this paragraph: ' + '<paragraph>'
            for idx in paragraphs_list:
                qs += sections[idx]
            qs +='</paragraph>'
                
            conv_relevant.append_message(conv_relevant.roles[0], qs)
            conv_relevant.append_message(conv_relevant.roles[1], None)
            prompt = conv_relevant.get_prompt()
            return tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')[1:].unsqueeze(0).to(device='cuda', non_blocking=True)
        
        def _local_generate(self, gen_kwargs, input_ids, attention_masks, pad_token_ids, image_tensor):
            with torch.no_grad():
                cont = self.model.generate(
                    input_ids,
                    attention_mask=attention_masks,
                    pad_token_id=pad_token_ids,
                    images=image_tensor,
                    image_sizes=gen_kwargs["image_sizes"],
                    do_sample=True if gen_kwargs["temperature"] > 0 else False,
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                    use_cache=self.use_cache,
                )
            return cont
        
        def _generate_reflectiva_kb(self, requests):
            res = []
            
            count_ret_token = 0
            count_active_context = 0

            re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
            chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
            num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
            pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
            for chunk in chunks:
                contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
                task = task[0]
                split = split[0]
                visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
                visuals = self.flatten(visuals)
                gen_kwargs = all_gen_kwargs[0]

                until = [self.tok_decode(self.eot_token_id)]
                
                total_qs_relevant = []

                if "until" in gen_kwargs:
                    until = gen_kwargs.pop("until")
                    if isinstance(until, str):
                        until = [until]
                    elif not isinstance(until, list):
                        raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")

                if "image_aspect_ratio" in gen_kwargs.keys() and "image_aspect_ratio" not in self._config.__dict__:
                    # here we should pop it out of gen_kwargs so that it doesn't get passed to the model for next step of generation
                    self._config.image_aspect_ratio = gen_kwargs.pop("image_aspect_ratio")
                    eval_logger.info(f"Setting image aspect ratio: {self._config.image_aspect_ratio}")
                # encode, pad, and truncate contexts for this batch
                
                # apply here image retrieval
                with torch.no_grad():           
                    if self.use_clip_retrieve:
                        pixel_values = torch.from_numpy(self.clip_preprocessor(visuals[0]).pixel_values[0])[None].to(dtype=torch.float16, device=self.clip.device)
                        image_features = self.clip.get_image_features(pixel_values)
                    elif self.use_eva_retrieve:
                        pixel_values = torch.from_numpy(self.clip_preprocessor(visuals[0]).pixel_values[0])[None].to(dtype=torch.float16, device=self.clip.device)
                        image_features = self.clip.encode_image(pixel_values)
                        
                image_features /= image_features.norm(dim=-1, p=2)
                entity_urls = self.retriever.retrieve(image_features, k=self.entity_k) # TODO va: add args.entity_k
                sections = []
                for entity_url in entity_urls:
                    for section in self.wikipedia[entity_url]['section_texts']:
                        if len(section.strip().lower()) < 5:
                            continue
                        sections.append(section)
                        
                assert len(sections) != 0
                question = contexts[0]
                
                # create the different input_ids for different inference stage
                for section in sections: # relevant
                    section = section.strip()
                    qs_retrieval = IMAGE_TOKEN + question
                    conv_relevant = copy.deepcopy(conv_templates[self.conv_template])
                    # conv_relevant = conv_templates[self.conv_template].copy()
                    qs_relevant = IMAGE_TOKEN + question
                    conv_relevant.append_message(conv_relevant.roles[0], qs_relevant)
                    qs_relevant = '[Retrieval]'
                    conv_relevant.append_message(conv_relevant.roles[1], qs_relevant)
                    
                    qs_relevant = 'Consider this paragraph: ' + '<paragraph>'
                    qs_relevant += section + '</paragraph>'
                    
                    conv_relevant.append_message(conv_relevant.roles[0], qs_relevant)
                    conv_relevant.append_message(conv_relevant.roles[1], None)
                    prompt_relevant = conv_relevant.get_prompt()
                    
                    input_ids_relevant = tokenizer_image_token(prompt_relevant, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
                    total_qs_relevant.append(input_ids_relevant[1:])
                
                # retrieval
                conv_retrieval = copy.deepcopy(conv_templates[self.conv_template])
                # conv_retrieval = conv_templates[self.conv_template].copy()
                conv_retrieval.append_message(conv_retrieval.roles[0], qs_retrieval)
                conv_retrieval.append_message(conv_retrieval.roles[1], None)
                prompt_retrieval = conv_retrieval.get_prompt()
                question_input = []

                if visuals:
                    image_tensor = process_images(visuals, self._image_processor, self._config)
                    if type(image_tensor) is list:
                        image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                    else:
                        image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)
                else:
                    image_tensor = None
                # prompts_input = contexts[0]

                # for visual, context in zip(visuals, contexts):
                #     if image_tensor is not None and len(image_tensor) != 0 and DEFAULT_IMAGE_TOKEN not in context:
                #         """
                #         Three senarios:
                #         1. No image, and there for, no image token should be added.
                #         2. image token is already specified in the context, so we don't need to add it.
                #         3. image token is not specified in the context and there is image inputs, so we need to add it. In this case, we add the image token at the beginning of the context and add a new line.
                #         """
                #         image_tokens = [DEFAULT_IMAGE_TOKEN] * len(visual) if isinstance(visual, list) else [DEFAULT_IMAGE_TOKEN]
                #         image_tokens = " ".join(image_tokens)
                #         question = image_tokens + "\n" + context
                #     else:
                #         question = context

                #     # This is much safer for llama3, as we now have some object type in it
                #     if "llama_3" in self.conv_template:
                #         conv = copy.deepcopy(conv_templates[self.conv_template])
                #         conv.tokenizer = self.tokenizer
                #         # \n must be removed in the template
                #         def replace_nth(sub,repl,txt,nth):
                #             arr=txt.split(sub)
                #             part1=sub.join(arr[:nth])
                #             part2=sub.join(arr[nth:])
                            
                #             return part1+repl+part2
                #         # question = replace_nth("\n", " ", question, 2) # TODO: consider to remove it

                #     else:
                #         conv = conv_templates[self.conv_template].copy()
                #     conv.append_message(conv.roles[0], question)
                #     conv.append_message(conv.roles[1], None)
                #     prompt_question = conv.get_prompt()
                #     if "llama_3" in self.conv_template and self.conv_template != 'llama_3_1':
                #         sep= '<|start_header_id|>' + conv.roles[1] + '<|end_header_id|>' + '\n\n'
                #         prompt_question = prompt_question + sep
                #     question_input.append(prompt_question)

                # # The above for loop has bugs. When there is no visuals, e.g. pure text, there will be no for loop execute resulting in an empty question_input (because no visuals)
                # # ideally in multimodal dataset, it is not necessary this if condition
                # if len(visuals) == 0:
                #     for context in contexts:
                #         question = context
                #         conv = conv_templates[self.conv_template].copy()
                #         conv.append_message(conv.roles[0], question)
                #         conv.append_message(conv.roles[1], None)
                #         prompt_question = conv.get_prompt()
                #         question_input.append(prompt_question)

                gen_kwargs["image_sizes"] = [visuals[idx].size for idx in range(len(visuals))]
                if "max_new_tokens" not in gen_kwargs:
                    gen_kwargs["max_new_tokens"] = 1024
                if "temperature" not in gen_kwargs:
                    gen_kwargs["temperature"] = 0
                if "top_p" not in gen_kwargs:
                    gen_kwargs["top_p"] = None
                if "num_beams" not in gen_kwargs:
                    gen_kwargs["num_beams"] = 1

                # input_ids_list = [tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for prompt in prompt_retrieval]
                input_ids_list = tokenizer_image_token(prompt_retrieval, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                if 'llama_3' in conv_retrieval.version:
                        if input_ids_list[0] == input_ids_list[1] == conv_retrieval.tokenizer.bos_token_id:
                            input_ids_list = input_ids_list[1:]
                            input_ids_list = input_ids_list.unsqueeze(dim=0)
                            # input_ids_list = [input_ids.squeeze(0) for input_ids in input_ids_list]
                            # for idx, el in enumerate(input_ids_list):
                            #     input_ids_list[idx] = input_ids_list[idx][1:]
                else:
                    input_ids_list = [input_ids.squeeze(0) for input_ids in input_ids_list]
                
                pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
                input_ids = self.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_ids).to(self.device)
                attention_masks = input_ids.ne(pad_token_ids).to(self.device)
                # TODO: pay attention to this major generation step - RETRIEVAL GENERATION
                try:
                    cont = _local_generate(self, gen_kwargs, input_ids, attention_masks, pad_token_ids, image_tensor)          
                    
                    text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=False)
                    text_outputs_cleaned = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
                    noise_flag = False
                except Exception as e:
                    eval_logger.error(f"Error {e} in generating")
                    cont = ""
                    text_outputs = [""]
                    
                # RELEVANT
                # torch.cuda.empty_cache()
                # gc.collect()
                
                relevant_paragraphs_detected = []
                if 128251 == cont[0][0].item():
                    count_ret_token += 1
                    for id_context, element in enumerate(total_qs_relevant):
                        relevant_ids = element.to(device='cuda', non_blocking=True)
                    
                        # Relevant Forward
                        with torch.inference_mode():
                            attention_masks = input_ids.ne(pad_token_ids).to(self.device)
                            output_ids= _local_generate(self, gen_kwargs, relevant_ids, attention_masks, pad_token_ids, image_tensor)
                            # output_ids, output_score = inference(args, model, relevant_ids, images, image_sizes, relevant_forward=True)
                    
                        if 128253 == output_ids[0][0].item():
                            relevant_paragraphs_detected.append(id_context)
                    
                    if len(relevant_paragraphs_detected) == 0:
                        relevant_paragraphs_detected = [ random.randint(0,len(sections)-1) ]
                        noise_flag = True
                    else:
                        count_active_context += 1

                    if not noise_flag:
                        with torch.inference_mode():
                            conv = copy.deepcopy(conv_templates[self.conv_template])
                            concat_relevant_paraghraphs = _concat_paragraph(relevant_paragraphs_detected, question, sections, conv, self.tokenizer)
                            attention_masks = concat_relevant_paraghraphs.ne(pad_token_ids).to(self.device)

                            output_ids= _local_generate(self, gen_kwargs, concat_relevant_paraghraphs, attention_masks, pad_token_ids, image_tensor)
                            
                            text_outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                            del output_ids
                        
                    else:
                        text_outputs = text_outputs_cleaned        
                else:
                    text_outputs = text_outputs_cleaned
                    
                torch.cuda.empty_cache()
                gc.collect()
                
                res.extend(text_outputs)
                self.cache_hook.add_partial("generate_until", (question, gen_kwargs), text_outputs)
                pbar.update(1)
            res = re_ords.get_original(res)

            pbar.close()
            
            # print stats
            print(f"Number of times that the model predict retrieval: {count_ret_token} over {len(res)}")
            print(f"Number of times when the context is different to none: {count_active_context} over {len(res)}")

            return res
        
        
        #  separate flow for reflectiva_kb experiments
        if self.reflectiva_kb:
            res = _generate_reflectiva_kb(self, requests)
            return res
        else:
            res = []
   
            re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
            chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
            num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
            pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
            for chunk in chunks:
                contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
                task = task[0]
                split = split[0]
                visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
                visuals = self.flatten(visuals)
                # we assume all gen kwargs in the batch are the same
                # this is safe to assume because the `grouper` object ensures it.
                gen_kwargs = all_gen_kwargs[0]

                # Set default values for until and max_new_tokens
                if isinstance(self.tokenizer, GemmaTokenizer) and not isinstance(self.eot_token_id, list):
                    until = [self.tok_decode([self.eot_token_id])]    
                else:
                    until = [self.tok_decode(self.eot_token_id)]

                # Update values from gen_kwargs if present
                if "until" in gen_kwargs:
                    until = gen_kwargs.pop("until")
                    if isinstance(until, str):
                        until = [until]
                    elif not isinstance(until, list):
                        raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")

                if "image_aspect_ratio" in gen_kwargs.keys() and "image_aspect_ratio" not in self._config.__dict__:
                    # here we should pop it out of gen_kwargs so that it doesn't get passed to the model for next step of generation
                    self._config.image_aspect_ratio = gen_kwargs.pop("image_aspect_ratio")
                    eval_logger.info(f"Setting image aspect ratio: {self._config.image_aspect_ratio}")
                # encode, pad, and truncate contexts for this batch

                # Here apply S2 with self._image_processor
                if visuals:
                    image_tensor = process_images(visuals, self._image_processor, self._config)
                    if type(image_tensor) is list:
                        image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                    else:
                        image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)
                else:
                    image_tensor = None

                # prompts_input = contexts[0]

                question_input = []

                for visual, context in zip(visuals, contexts):
                    if image_tensor is not None and len(image_tensor) != 0 and DEFAULT_IMAGE_TOKEN not in context:
                        """
                        Three senarios:
                        1. No image, and there for, no image token should be added.
                        2. image token is already specified in the context, so we don't need to add it.
                        3. image token is not specified in the context and there is image inputs, so we need to add it. In this case, we add the image token at the beginning of the context and add a new line.
                        """
                        image_tokens = [DEFAULT_IMAGE_TOKEN] * len(visual) if isinstance(visual, list) else [DEFAULT_IMAGE_TOKEN]
                        image_tokens = " ".join(image_tokens)
                        question = image_tokens + "\n" + context
                    else:
                        question = context

                    # This is much safer for llama3, as we now have some object type in it
                    if "llama_3" in self.conv_template:
                        conv = copy.deepcopy(conv_templates[self.conv_template])
                        conv.tokenizer = self.tokenizer
                        # \n must be removed in the template
                        def replace_nth(sub,repl,txt,nth):
                            arr=txt.split(sub)
                            part1=sub.join(arr[:nth])
                            part2=sub.join(arr[nth:])
                            
                            return part1+repl+part2
                        # question = replace_nth("\n", " ", question, 2) # TODO: consider to remove it

                    else:
                        conv = conv_templates[self.conv_template].copy()
                    conv.append_message(conv.roles[0], question)
                    conv.append_message(conv.roles[1], None)
                    prompt_question = conv.get_prompt()
                    if "llama_3" in self.conv_template and self.conv_template != 'llama_3_1':
                        sep= '<|start_header_id|>' + conv.roles[1] + '<|end_header_id|>' + '\n\n'
                        prompt_question = prompt_question + sep
                    if self.conv_template == 'llama_3_1_reasoning':
                        sep = '<｜Assistant｜>'
                        prompt_question = prompt_question + sep
                    question_input.append(prompt_question)

                # The above for loop has bugs. When there is no visuals, e.g. pure text,
                # there will be no for loop execute resulting in an empty question_input (because no visuals)
                # Scenario 1 won't even be execute
                if len(visuals) == 0:
                    for context in contexts:
                        question = context
                        conv = conv_templates[self.conv_template].copy()
                        conv.append_message(conv.roles[0], question)
                        conv.append_message(conv.roles[1], None)
                        prompt_question = conv.get_prompt()
                        question_input.append(prompt_question)

                # input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
                # preconfigure gen_kwargs with defaults
                gen_kwargs["image_sizes"] = [visuals[idx].size for idx in range(len(visuals))]
                if "max_new_tokens" not in gen_kwargs:
                    gen_kwargs["max_new_tokens"] = 1024
                if "temperature" not in gen_kwargs:
                    gen_kwargs["temperature"] = 0
                if "top_p" not in gen_kwargs:
                    gen_kwargs["top_p"] = None
                if "num_beams" not in gen_kwargs:
                    gen_kwargs["num_beams"] = 1

                input_ids_list = [tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for prompt in question_input]
                if 'llama_3' in conv.version or 'gemma_2' in conv.version:
                        if input_ids_list[0][0] == input_ids_list[0][1] == conv.tokenizer.bos_token_id:
                            input_ids_list = [input_ids.squeeze(0) for input_ids in input_ids_list]
                            for idx, el in enumerate(input_ids_list):
                                input_ids_list[idx] = input_ids_list[idx][1:]
                else:
                    input_ids_list = [input_ids.squeeze(0) for input_ids in input_ids_list]
                
                pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
                input_ids = self.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_ids).to(self.device)
                attention_masks = input_ids.ne(pad_token_ids).to(self.device)
                # These steps are not in LLaVA's original code, but are necessary for generation to work
                # TODO: pay attention to this major generation step...
                try:                    
                    cont = self.model.generate(
                        input_ids,
                        attention_mask=attention_masks,
                        pad_token_id=pad_token_ids,
                        images=image_tensor,
                        image_sizes=gen_kwargs["image_sizes"],
                        do_sample=True if gen_kwargs["temperature"] > 0 else False,
                        temperature=gen_kwargs["temperature"],
                        top_p=gen_kwargs["top_p"],
                        num_beams=gen_kwargs["num_beams"],
                        max_new_tokens=gen_kwargs["max_new_tokens"],
                        use_cache=self.use_cache,
                    )
                    text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
                except Exception as e:
                    eval_logger.error(f"Error {e} in generating")
                    cont = ""
                    text_outputs = [""]

                if 'phi' in self.conv_template and '<|end|>' in text_outputs[0]:
                    text_outputs[0] = text_outputs[0].replace('<|end|>', '')
                res.extend(text_outputs)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_outputs)
                pbar.update(1)
                # reorder this group of results back to original unsorted form
            res = re_ords.get_original(res)

            pbar.close()
            return res
