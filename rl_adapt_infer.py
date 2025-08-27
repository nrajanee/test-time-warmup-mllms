# borrowed some implementation details for GRPO from here: https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py
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
from tqdm.auto import tqdm
from auxiliary_data.rl_caption_dataset import RLCaptionDataset
import wandb
import math
from accelerate import dispatch_model
import torch.nn.functional as F

wandb.util.logger.setLevel("DEBUG")
import gc
from qa_eval import get_accuracy
from torchviz import make_dot
import numpy as np
import time
from model_functions.llama_model import LlamaVisionModel
from model_functions.llava_model import LlavaModel
from model_functions.qwen_model import QwenModel
from utils import NUM_TRAIN_IMAGES, NUM_PROMPTS_PER_IMAGE, BASELINE_ACCS, NUM_AUX_LABELS_PER_IMAGE
from utils import intialize_clip_model
from utils import get_prompts
from auxiliary_data.rl_caption_dataset import RLCaptionDataset
from base_adapt_infer import BaseAdaptInfer
from auxiliary_data.caption_generation import train_dataset_unfiltered_for_RL_online_caption_generation

# deterministic behaviour
np.random.seed(30)
random.seed(30)
torch.manual_seed(30)
torch.cuda.manual_seed(30)
torch.cuda.manual_seed_all(30)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#torch.use_deterministic_algorithms(True)


class RLAdaptInfer(BaseAdaptInfer):
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
    ):
        super().__init__(
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
        )

        self.clip_model, self.clip_processor, self.clip_model_max_length = (
            intialize_clip_model()
        )

        self.ref_model = None # copy after initialization. 
        self.beta = 0.01 # for KL divergence. 

    
    def _compute_advantages(self, rewards):
        # num_generations = NUM_AUX_LABELS
        num_generations = NUM_AUX_LABELS_PER_IMAGE
        mean_grouped_rewards = rewards.view(-1, num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, num_generations).std(dim=1)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4) # so that you don't divide by 0. 
        return advantages

    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        # Get logits from model.forward it only returns number of input_ids logits. 
        print(f"Allocated memory before forward {torch.cuda.memory_reserved() / 1e9:.2f} GB", )
        forward_out = model.forward(input_ids=input_ids, attention_mask=attention_mask)
        logits = forward_out.logits[:, -logits_to_keep:, :]
        return F.log_softmax(logits, dim=-1)

    # This should be able to handle all completion_ids (=NUM_AUX_LABELS). 
    def _compute_grpo_loss(self, prompt_capt_scores):
        prompt_completion_ids = prompt_capt_scores["prompt_completion_ids"]
        prompt_c_ids_attn_mask =  prompt_capt_scores["prompt_cids_attn_mask"]
        completion_ids =  prompt_capt_scores["completion_ids"] 
        completion_mask = prompt_capt_scores["completion_ids_attn_mask"]
        logits_to_keep = completion_ids.size(1)
        with torch.no_grad(): # don't need gradients for ref_model
            ref_per_token_logps = self._get_per_token_logps(self.ref_model, prompt_completion_ids, prompt_c_ids_attn_mask,logits_to_keep) # have to do this every batch cause doing before in dataset won't work without padding. And padding can only be done in train_datalaoder custom_collate cause diff batches. 

        curr_per_token_logps = self._get_per_token_logps(self.mm_model.model, prompt_completion_ids, prompt_c_ids_attn_mask, logits_to_keep)
        rewards = prompt_capt_scores["clip_scores"]
        advantages = self._compute_advantages(rewards)[0] # make it 1-D
        
        # Compute the KL divergence between the model and the reference model
        per_token_kl = torch.exp(ref_per_token_logps - curr_per_token_logps) - (ref_per_token_logps - curr_per_token_logps) - 1
        #print("per_token_kl", per_token_kl)
        # x - x.detach() allows for preserving gradients from x -> need to check this detach thing. Does it give you old weights? before training on epoch. 
        per_token_loss = torch.exp(curr_per_token_logps - curr_per_token_logps.detach()) * advantages.unsqueeze(1)
        #print("old model - new model",  torch.exp(curr_per_token_logps - curr_per_token_logps.detach()))
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

        del ref_per_token_logps, curr_per_token_logps, per_token_kl, per_token_loss, advantages

        return loss
        

    def rl_custom_collate_fn(self, batch):
        return self.mm_model.rl_custom_collate_fn(batch)

    def create_uda_train_dataloader(self, vllm_model, batch_size): 
        prompts = get_prompts()
        c_ablation = {
            "id": 2,
            "gen_num_captions": 10,
            "choose_cap_per_prompt": True,
        }  # always this right now. 

        train_regen_unfiltered_captions = train_dataset_unfiltered_for_RL_online_caption_generation(
        self.dataset_type,
        self.mm_model,
        self.clip_model, 
        self.clip_processor,
        self.clip_model_max_length,
        self.train_images,
        self.image_folder,
        c_ablation,
        prompts,
        use_vllm=True,
        vllm_model=vllm_model
        )

        # Create dataset again.
        train_dataset = RLCaptionDataset(
            c_ablation["id"],
            self.dataset_type,
            self.image_folder,
            self.train_images,
            self.val_images,
            train_regen_unfiltered_captions,
            random_augment=True,  # it's always true for now anyway.
            len_set_of_prompts = self.len_set_of_prompts 
        )
        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=batch_size,
            collate_fn=self.rl_custom_collate_fn,
            num_workers=0,
        )

        return train_dataset, train_dataloader


    def _batch_to_gpu(self, batch): 
        batch_to_gpu = []
        for input in batch: 
            input_to_gpu = {}
            for k, v in input.items(): 
                if k == "img_path": 
                    continue
                input_to_gpu[k] = v.to(self.mm_model.model.device)
            
            batch_to_gpu.append(input_to_gpu)
        
        return batch_to_gpu

    
    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        # Get logits from model.forward it only returns number of input_ids logits. 
        completion_ids = input_ids[:, -logits_to_keep:]

        forward_out = model.forward(input_ids=input_ids, attention_mask=attention_mask)
        # add 1 because the last lpgits of the seq is excluded. 
        logits = forward_out.logits[:, -(logits_to_keep+1):, :]
        log_softmax_probs = F.log_softmax(logits, dim=-1) # on the last dim. the first dimension will be number of completion_ids
        selected_logps = log_softmax_probs.gather(dim=2, index=completion_ids.unsqueeze(2)).squeeze(2).to(self.mm_model.model.device)
        return selected_logps


    def unsup_domain_adapt(self):
        config = wandb.config
        start_learning_rate = config.learning_rate
        batch_size = config.batch_size
        epochs = config.epochs
        
        print("Dataloader created")
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.mm_model.model.parameters()),
            lr=start_learning_rate,
        )
        print("Train")

        start = time.time()
        global_train_step = 0  # ends at num_training_steps

        # initialization. 
        #vllm_model, optimizer_state_dict, hf_device_map = self.create_vllm_model_get_opt_dict_dev_map(optimizer)
        
        for epoch in range(1, epochs + 1): 
            # use VLLM otherwise it'll be too slow. 
            # offload optimizer
            #  model to CPU   

            vllm_model, optimizer_state_dict, hf_device_map = self.create_vllm_model_get_opt_dict_dev_map(optimizer)
            train_dataset, train_dataloader = self.create_uda_train_dataloader(vllm_model, batch_size)
            del vllm_model
            gc.collect()
            torch.cuda.empty_cache()  
            optimizer = self.dispatch_model_optimizer_after_use_vllm(hf_device_map, optimizer_state_dict, start_learning_rate)
            #print("unfrozen parameters")
            #for name, param in self.mm_model.model.named_parameters():
                #if param.requires_grad:
                    #print(name, param.shape)

            if epoch == 1: # just do this once before any training. 
                num_training_steps = epochs * len(train_dataloader)
                dataset_info = train_dataset.get_dataset_info_for_config()
                with self.wandb_lock:
                    train_dataset.log_initialize_first_order_metric(self.dataset_type)


                update_config = {
                    "model": self.mm_model.model_type,
                    "num_training_steps": num_training_steps,
                    "training_parameters": self.training_parameters,
                    "dataset_info": dataset_info,
                    "caption_generation": "online",
                    "adaptation_type": self.adaptation_type,
                }
                progress_bar = tqdm(range(num_training_steps))

                wandb.config.update(update_config)

            with self.wandb_lock:
                train_dataset.log_initialize_first_order_metric(self.dataset_type)
                wandb.log(
                    {
                        "epoch": 0,
                        "train_accuracy": BASELINE_ACCS[
                            f"{self.model_type}_{self.dataset_type}"
                        ][0],
                    }
                )
                wandb.log(
                    {
                        "epoch": 0,
                        "val_accuracy": BASELINE_ACCS[
                            f"{self.model_type}_{self.dataset_type}"
                        ][1],
                    }
                )

            self.train_dataset = train_dataset
         
            self.train_dataset.set_mm_model(self.mm_model)
            assert (
                self.train_dataset.__len__() == NUM_TRAIN_IMAGES * NUM_PROMPTS_PER_IMAGE # TODO: replace 3 with NUM_IMAGES. 
            ), "train dataset is not the correct length here."

            self.mm_model.model.train()
            assert self.mm_model.model.training == True # cause you switch for inference. 
            # Create train_dataloader
            for batch in train_dataloader:
                global_train_step += 1
                batch = self._batch_to_gpu(batch)
                loss = 0
                for prompt_scores_capt in batch: 
                    loss += self._compute_grpo_loss(prompt_scores_capt)
                loss = loss/len(batch)
                #outputs = self.mm_model.model(**batch)
                # print('outputs',outputs)
                #print('loss', loss)
                loss.backward()
                #torch.nn.utils.clip_grad_norm(self.mm_model.model.parameters(), max_norm=0.1)
                optimizer.step()
                # lr_scheduler.step()
                optimizer.zero_grad()

                with self.wandb_lock:
                    wandb.log(
                        {
                            # "learning_rate": lr_scheduler.get_last_lr()[
                            #   0
                            # ],  # Get current LR from scheduler
                            "learning_rate": start_learning_rate,
                            "loss": loss.item(),
                            "epoch": epoch,
                        }
                    )

                progress_bar.update(1)

            end = time.time()
            print("Train time: ", end - start)

            # after every epoch make sure it updates:
            self.train_dataset.set_mm_model(self.mm_model)
            
            ## INFER

            start = time.time()
            train_accuracy = self.train_infer()
            val_accuracy = self.val_infer()
            end = time.time()
            print("Q/A time", end - start)

            with self.wandb_lock:
                wandb.log({"epoch": epoch, "train_accuracy": train_accuracy})
                wandb.log({"epoch": epoch, "val_accuracy": val_accuracy})

            print(f"DONE train,val accuracy", train_accuracy, val_accuracy)

            ## FIRST ORDER METRIC
            #start = time.time()
            # initialize again and subsuquent loops will take care of it. 
            #vllm_model, optimizer_state_dict, hf_device_map = self.create_vllm_model_get_opt_dict_dev_map(optimizer)
            #self.uda_log_first_order_metric(vllm_model, epoch)
            #end = time.time()
            #print("Weakly supervised label metric time", end - start)
            
            del train_dataloader

        assert global_train_step == num_training_steps
        # clear things at the end.
        del (optimizer, self.mm_model.model, self.mm_model, vllm_model)
        gc.collect()
        torch.cuda.empty_cache()

    def tta_adapt(self):
        pass

    def initialize_wandb(self):
        group = f"SSP_RLGRPO_{self.adaptation_type}_{self.model_type}_dataset_{self.dataset_type}_TP_{self.training_parameters}"  # outside of training opt ablations.
        print("WandB initialized.")
        wandb.init(group=group)
        from threading import Lock

        wandb_lock = Lock()
        self.wandb_lock = wandb_lock
    

    def rl_adapt_infer(
        self,
    ):  # save the results to "/local/zemel/nikita/all-multi-modals/online_learning/dataset/infer_pos/answers/"
        # freeze everything first.
        # Create new model per sweep.
        try:
            model_path = None
            if self.model_type == "llama":
                model_path = "/local/zemel/weights/Llama-3.2-11B-Vision-Instruct"

            if self.model_type == "gqa_best_epoch_llama": 
                model_path = "/local/zemel/nikita/all-multi-modals/online_learning/SSP_gqa_save_best_model"
            
            if self.model_type == "gqa_epoch_3_llama":
                model_path = "/local/zemel/nikita/all-multi-modals/online_learning/SSP_gqa_save_epoch3_model"

            if self.model_type == "llava":
                model_path = "/local/zemel/weights/llava-v1.6-mistral-7b-hf/"

            if self.model_type == "qwen":
                model_path = "/local/zemel/weights/Qwen-VL-Chat-qwen-7b"

            self.model_path = model_path
            self.initialize_wandb()

            if self.adaptation_type == "uda":
                self.initialize_model_and_freeze_parameters(quantized=False)
                self.ref_model = copy.deepcopy(self.mm_model.model)
                self.unsup_domain_adapt()

            if self.adaptation_type == "tta":
                self.tta_adapt()

        except Exception as e:
            print(f"Run failed with error: {e}")
            import traceback

            traceback.print_exc()
            wandb.log({"error": str(e)})
            raise
        finally:
            wandb.finish()  # finish the run
            print("Finished the run and clearing memory")
            gc.collect()
            torch.cuda.empty_cache()

            
