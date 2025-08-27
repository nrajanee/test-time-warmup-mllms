# Here you'd have the training loop.
# The infer function.
import os
import torch
import copy
import random
import argparse
import os
import json
import shortuuid
from PIL import Image
import torch.nn as nn
import requests
from torch.utils.data import DataLoader, RandomSampler
from torch.optim import AdamW
from transformers import get_scheduler
from peft import get_peft_model, LoraConfig, TaskType

from tqdm.auto import tqdm
import wandb
import math
from utils import get_rel_paths

from accelerate.hooks import remove_hook_from_submodules
from accelerate import dispatch_model

wandb.util.logger.setLevel("DEBUG")
import gc
from qa_eval import get_accuracy
from torchviz import make_dot
import numpy as np
import time
from model_functions.llama_model import LlamaVisionModel
from model_functions.gemma_model import GemmaModel
from model_functions.llava_model import LlavaModel
from model_functions.qwen_model import QwenModel
from utils import NUM_TRAIN_IMAGES,NUM_VAL_IMAGES, NUM_AUX_LABELS_PER_IMAGE


def initialize_mm_model(model_path, quantized=False):
    if "Llama" in model_path or "SSP_gqa_save" in model_path:
        return LlamaVisionModel("llama", model_path, quantized)
    if "llava" in model_path:
        return LlavaModel("llava", model_path)
    if "qwen" in model_path:
        return QwenModel("qwen", model_path)
    if "gemma" in model_path: 
        return GemmaModel("gemma", model_path)
    return None


class BaseAdaptInfer:
    def __init__(
        self,
        model_type,
        adaptation_type,
        training_parameters,
        train_dataset,
        train_images,
        val_images,
        dataset_type,
        img_question_mapping,
        image_folder,
        len_set_of_prompts,
        lora
    ):

        self.base = "base adapt infer"
        self.model_type = model_type
        self.adaptation_type = adaptation_type
        self.training_parameters = training_parameters
        self.img_question_mapping = img_question_mapping
        self.image_folder = image_folder
        self.train_dataset = train_dataset
        self.train_images = train_images
        self.val_images = val_images
        self.dataset_type = dataset_type
        print("training_params", self.training_parameters)
        # these will get set in the finetune_infer method before calling train so that each sweep has a different model.
        self.mm_model = None
        self.mm_tokenizer = None
        self.mm_model_connector_name = None
        self.mm_model_llm_name = None
        self.model_path = None
        self.len_set_of_prompts = len_set_of_prompts
        self.lora = lora

    def custom_collate_fn(self, batch):
        return self.mm_model.custom_collate_fn(batch)
    
    def unfreeze_connector_parameters(self): 
        nested = self.mm_model_connector_name.split(".")
        obj = self.mm_model.model
        for (
            attr
        ) in nested:  # this goes upto the connector name getting doen.
            obj = getattr(obj, attr)
        param = obj
        # Unfreeze all parameters in this module
        for p in param.parameters():
            p.requires_grad = True


    def unfreeze_parameters(self, training_parameters):
         # freeze everything first. 
        for param in self.mm_model.model.parameters():
            param.requires_grad = False
        # print([i for i in self.mm_model.model.named_parameters()])
        if "qwen" in self.model_type:  # just freeze the vision component for now.
            for param in self.mm_model.model.parameters():  # unfreeze all first
                param.requires_grad = True

            nested = self.mm_model.get_vision_name().split(".")
            obj = self.mm_model.model
            for attr in nested:  # this goes upto the connector name getting doen.
                obj = getattr(obj, attr)
            param = obj
            # freeze all parameters in this module
            for p in param.parameters():
                p.requires_grad = False  # freeze visual stuff.

        else:  # llama and llava and gemma.
            for train_param in training_parameters:
                if train_param == "connector":
                    nested = self.mm_model_connector_name.split(".")
                    obj = self.mm_model.model
                    for (
                        attr
                    ) in nested:  # this goes upto the connector name getting doen.
                        obj = getattr(obj, attr)
                    param = obj
                    # Unfreeze all parameters in this module
                    for p in param.parameters():
                        p.requires_grad = True

                if train_param == "llm":
                    nested = self.mm_model_llm_name.split(".")
                    obj = self.mm_model.model
                    for attr in nested:
                        obj = getattr(obj, attr)
                    param = obj
                    # Unfreeze all parameters in this module. Or maybe just a few layers?
                    for p in param.parameters():
                        p.requires_grad = True

    def get_answer_for_question_img(
        self, qs, img_path
    ):  # ensure correct prompting here
        # ensure training is not on.
        return self.mm_model.get_answer_for_question_img(qs, img_path)

    def infer_per_image(self, img_path=None, image_file=None):
        if image_file == None:
            image_file = img_path.split("/")[-1]

        if img_path is None:
            img_path = os.path.join(self.image_folder, image_file)

        qs_id, qs = self.img_question_mapping[image_file]
        print("question", qs)
        img_path = os.path.join(self.image_folder, image_file)
        answer = self.get_answer_for_question_img(qs, img_path)
        print("answer", answer)
        return qs_id, answer

    def infer_on_dataset(self, infer_train=True):
        self.mm_model.model.eval()
        results = {}
        if infer_train:
            image_dataset = self.train_images
            assert len(image_dataset) == NUM_TRAIN_IMAGES
        else:  # val
            image_dataset = self.val_images
            assert len(image_dataset) == NUM_VAL_IMAGES

        answered_imgs = set()  # cause of mul captions per image.
        num_imgs = 0
        for i, image_file in image_dataset.items():  # akk t
            if image_file in answered_imgs:
                continue
            answered_imgs.add(image_file)
            qs_id, answer = self.infer_per_image(image_file=image_file)
            
            print("answer", answer)
            results[qs_id] = answer
            num_imgs += 1

        return results

    def train_infer(self, train_results=None):
        rel_paths = get_rel_paths(self.dataset_type)
        annotations_file = rel_paths["annotation_file"]

        if self.adaptation_type == "tta" or self.adaptation_type == "online_cont":
            assert train_results is not None

        if train_results is None:
            train_results = self.infer_on_dataset(infer_train=True)

        train_accuracy = get_accuracy(
            annotations_file, train_results, self.dataset_type
        )

        return train_accuracy

    def val_infer(self):
        print("val infer")
        rel_paths = get_rel_paths(self.dataset_type)
        annotations_file = rel_paths["annotation_file"]
        val_results = self.infer_on_dataset(infer_train=False)  # val results
        val_accuracy = get_accuracy(annotations_file, val_results, self.dataset_type)

        return val_accuracy
    
    def unfreeze_lora_parameters(self): 
        for param in self.mm_model.model.parameters():
                param.requires_grad = False
            
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ], 
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
    
        self.mm_model.model.language_model = get_peft_model(self.mm_model.model.language_model, lora_config)
        self.unfreeze_connector_parameters()

    def initialize_model_and_freeze_parameters(self, quantized=False):
        self.mm_model = initialize_mm_model(self.model_path, quantized)
        self.mm_tokenizer = self.mm_model.get_tokenizer()
        self.mm_model_connector_name = self.mm_model.get_connector_name()
        self.mm_model_llm_name = self.mm_model.get_llm_name()

        if self.lora == True: 
             print("using LoRA")
             self.unfreeze_lora_parameters()


        else: 
         self.unfreeze_parameters(self.training_parameters)
        
        print(
            "unfrozen relevant parameters",
            [
                name
                for name, param in self.mm_model.model.named_parameters()
                if param.requires_grad
            ],
        )
        

    
    def create_vllm_model_get_opt_dict_dev_map(self, optimizer): 
        print(f"Allocated memory before offloading to CPU: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        hf_device_map = self.mm_model.model.hf_device_map

        remove_hook_from_submodules(self.mm_model.model)

        self.mm_model.model.to('cpu')
        optimizer_state_dict = optimizer.state_dict()
        optimizer = None  
        gc.collect()           
        torch.cuda.empty_cache() 

        print(f"Allocated memory after offloacing to CPU: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        # save the model to load to a path 
        
        save_path = "/local/zemel/nikita/all-multi-modals/online_learning/save_vllm_model/"

        self.mm_model.model.save_pretrained(save_path)
        self.mm_model.mm_processor.save_pretrained(save_path)


        from vllm import LLM
        # Load vllm model.        )
        #vllm_model = LLM(model=save_path)
        vllm_model = LLM(model=save_path, gpu_memory_utilization=0.85, max_model_len=4096, enforce_eager=True, max_num_seqs=20, seed=0) # use 2 devices. 

        return vllm_model, optimizer_state_dict, hf_device_map
    
    def dispatch_model_optimizer_after_use_vllm(self,hf_device_map, optimizer_state_dict, start_learning_rate): 
        # move model back to GPU and create optimizer
        #self.mm_model.model = AutoModelForPreTraining.from_pretrained(save_path, device_map='auto')
        #self.mm_model.mm_processor = AutoProcessor.from_pretrained(save_path)
        self.mm_model.model = dispatch_model(self.mm_model.model, device_map=hf_device_map)

        optimizer = AdamW( filter(lambda p: p.requires_grad, self.mm_model.model.parameters()),lr=start_learning_rate)
        optimizer.load_state_dict(optimizer_state_dict)

        # unfreeze params again. 
        self.unfreeze_parameters(self.training_parameters)

        return optimizer
    
    def uda_log_first_order_metric(self, vllm_model, epoch): 
        self.train_dataset.set_vllm_model(vllm_model)

        with self.wandb_lock:
            self.train_dataset.calculate_and_log_current_first_order_metric(epoch)

    def unsup_domain_adapt(self):
        pass

    def tta_adapt(self):
        pass
