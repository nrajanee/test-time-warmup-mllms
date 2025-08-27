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
import wandb
import math
from base_adapt_infer import BaseAdaptInfer
from transformers import AutoModelForPreTraining, AutoProcessor, AutoModelForCausalLM, AutoConfig
from accelerate import dispatch_model
from utils import BASELINE_ACCS

wandb.util.logger.setLevel("DEBUG")
import gc
from qa_eval import get_accuracy
from torchviz import make_dot
import numpy as np
import time
from model_functions.llama_model import LlamaVisionModel
from model_functions.llava_model import LlavaModel
from model_functions.qwen_model import QwenModel
from utils import NUM_TRAIN_IMAGES,NUM_VAL_IMAGES, NUM_AUX_LABELS_PER_IMAGE
from utils import intialize_clip_model
from utils import get_prompts
from auxiliary_data.caption_generation import (
    train_dataset_online_caption_generation,
    per_image_caption_generation,
)

# deterministic behaviour
np.random.seed(30)
random.seed(30)
torch.manual_seed(30)
torch.cuda.manual_seed(30)
torch.cuda.manual_seed_all(30)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# torch.use_deterministic_algorithms(True)
from auxiliary_data.caption_dataset import CaptionDataset

# from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter(log_dir="./logs")


class OnlineAuxAdaptInfer(BaseAdaptInfer):
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
        len_set_of_prompts = 10,
    ):
        super().__init__(
            model_type,
            adaptation_type,
            training_parameters,
            None, # don't need to set train_dataset because you don't use it
            train_images,
            val_images,
            dataset_type,
            img_question_mapping,
            image_folder,
            len_set_of_prompts
        )
        self.clip_model, self.clip_processor, self.clip_model_max_length = (
            intialize_clip_model()
        )

        self.base_train_dataset = train_dataset

    
    def create_uda_train_dataloader_dataset(self, epoch, vllm_model, batch_size):
        # Produce captions
        prompts = get_prompts()
        c_ablation = {
            "id": 2,
            "gen_num_captions": 10,
            "choose_cap_per_prompt": True,
        }  # always this right now. 

        train_regenerate_captions = train_dataset_online_caption_generation(
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

        with open(f"/local/zemel/nikita/all-multi-modals/online_learning/{epoch}_caption_data.json", "w") as json_file:
            json.dump(train_regenerate_captions, json_file, indent=4) 

        # Create dataset again.
        train_dataset = CaptionDataset(
            self.adaptation_type,
            c_ablation["id"],
            self.dataset_type,
            self.image_folder,
            self.train_images,
            self.val_images,
            train_regenerate_captions,
            random_augment=True,
            len_set_of_prompts=self.len_set_of_prompts  # it's always true for now anyway.
        )
        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=batch_size,
            collate_fn=self.custom_collate_fn,
            num_workers=0,
        )

        return train_dataset, train_dataloader
    
    def create_tta_train_dataloader_dataset(self, vllm_model, batch_size): 
        pass

    ## TODO: fix the vllm bit here. 
    def unsup_domain_adapt(self):
        # Access parameters for this run
        config = wandb.config
        start_learning_rate = config.learning_rate
        batch_size = config.batch_size
        epochs = config.epochs

        initial_train_dataloader = DataLoader(
            self.base_train_dataset,
            shuffle=True,
            batch_size=batch_size,
            collate_fn=self.custom_collate_fn,
            num_workers=0,
        )

        num_training_steps = epochs * len(initial_train_dataloader)
        dataset_info = self.base_train_dataset.get_dataset_info_for_config()

        update_config = {
            "model": self.mm_model.model_type,
            "num_training_steps": num_training_steps,
            "training_parameters": self.training_parameters,
            "dataset_info": dataset_info,
            "caption_generation": "online",
            "adaptation_type": self.adaptation_type,
        }

        wandb.config.update(update_config)
        print("Dataloader created")
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.mm_model.model.parameters()),
            lr=start_learning_rate,
        )

        progress_bar = tqdm(range(num_training_steps))
        print("Train")

        start = time.time()
        global_train_step = 0  # ends at num_training_steps
        with self.wandb_lock:
            self.base_train_dataset.log_initialize_first_order_metric(self.dataset_type)

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

        for epoch in range(1, epochs + 1):  
            self.mm_model.model.train() #  # cause you switch for inference. 
            assert self.mm_model.model.training == True
            if epoch == 1:
                train_dataloader = initial_train_dataloader
                train_dataset = self.base_train_dataset

            else:
                # use VLLM otherwise it'll be too slow. 
                # offload optimizer
                #  model to CPU
                vllm_model, optimizer_state_dict, hf_device_map = self.create_vllm_model_get_opt_dict_dev_map(optimizer)
                train_dataset, train_dataloader = self.create_uda_train_dataloader_dataset(epoch, vllm_model, batch_size)
                del vllm_model
                gc.collect()
                torch.cuda.empty_cache()  
                optimizer = self.dispatch_model_optimizer_after_use_vllm(hf_device_map, optimizer_state_dict, start_learning_rate)
            
            self.train_dataset = train_dataset

            self.train_dataset.set_mm_model(self.mm_model)
            assert (
                self.train_dataset.__len__() == NUM_TRAIN_IMAGES * NUM_AUX_LABELS_PER_IMAGE
            ), "train dataset is not the correct length here."

            # Create train_dataloader
            for batch, img_paths in train_dataloader:
                global_train_step += 1
                
                if "qwen" in self.model_type:
                    batch_input_ids, batch_attention_mask, batch_labels = batch
                    batch_input_ids = batch_input_ids.to(self.mm_model.model.device)
                    batch_attention_mask = batch_attention_mask.to(
                        self.mm_model.model.device
                    )
                    batch_labels = batch_labels.to(self.mm_model.model.device)
                    outputs = self.mm_model.model(
                        input_ids=batch_input_ids,
                        attention_mask=batch_attention_mask,
                        labels=batch_labels,
                    )
                    # print("outputs", outputs)
                else:
                    batch = {
                        k: (
                            v.to(self.mm_model.model.device)
                            if k != "image_sizes"
                            else v
                        )
                        for k, v in batch.items()
                    }
                    outputs = self.mm_model.model(**batch)
                # print('outputs',outputs)
                loss = outputs.loss
                # print('loss', loss)
                loss.backward()

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
           
            ## FIRST ORDER METRIC
            #start = time.time()
            #self.uda_log_first_order_metric(epoch, start_learning_rate, optimizer)
            #print("Weakly supervised label metric time", end - start)

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
            
            del train_dataloader
            gc.collect()
            torch.cuda.empty_cache()

        assert global_train_step == num_training_steps
        # clear things at the end.
        del (initial_train_dataloader, optimizer, self.mm_model.model, self.mm_model)
        gc.collect()
        torch.cuda.empty_cache()

    def tta_adapt(self):
        config = wandb.config
        start_learning_rate = config.learning_rate
        batch_size = config.bs_epochs["batch_size"]
        epochs = config.bs_epochs["epochs"]

        dataset_info = {"caption id": 2, "random_augment": False}  # for now.

        update_config = {
            "batch_size": batch_size,
            "epochs": epochs,
            "model": self.mm_model.model_type,
            "training_parameters": self.training_parameters,
            "dataset_info": dataset_info,
            "caption_generation": "online",
            "adaptation_type": self.adaptation_type,
        }

        wandb.config.update(update_config)
        per_epoch_results = []
        per_epoch_first_order_metrics = []
        for i in range(0, epochs):
            per_epoch_results.append({})  # each of them per_img_results
            per_epoch_first_order_metrics.append({})  # per img first order metric.

        for img_idx in range(0, len(self.base_train_dataset)):
            self.initialize_model_and_freeze_parameters(quantized=False)  # initialize the model for each image.
            start = time.time()
            img_caption_gens = self.base_train_dataset[img_idx]
            assert (
                len(img_caption_gens) == NUM_AUX_LABELS_PER_IMAGE
            ), "number of image captions not equal to 10"
            base_train_per_img_dataset = CaptionDataset(
                self.adaptation_type,
                2,
                self.dataset_type,
                self.image_folder,
                self.train_images,
                None,
                img_caption_gens,
                random_augment=False,
                len_set_of_prompts=self.len_set_of_prompts
            )  # c_abl_id is 2 for no. don't need val_images, random_augmnet is True.

            with self.wandb_lock:
                base_train_per_img_dataset.log_initial_train_first_order_metric(
                    self.dataset_type
                )

                wandb.log(
                    {
                        "epoch": 0,
                        "train_accuracy": BASELINE_ACCS[
                            f"{self.model_type}_{self.dataset_type}"
                        ][0],
                    }
                )

            initial_train_dataloader = DataLoader(
                base_train_per_img_dataset,
                shuffle=True,
                batch_size=batch_size,
                collate_fn=self.custom_collate_fn,
                num_workers=0,
            )

            optimizer = AdamW(
                filter(lambda p: p.requires_grad, self.mm_model.model.parameters()),
                lr=start_learning_rate,
            )

            num_training_steps = epochs * len(train_dataloader)
            print("num_training_steps", num_training_steps)
            calc_nts = epochs * math.ceil(
                base_train_per_img_dataset.__len__() / batch_size
            )
            assert (
                num_training_steps == calc_nts
            ), f"not equal calc: {calc_nts}"  # epochs
            progress_bar = tqdm(range(num_training_steps))
            global_train_step = 0

            for epoch in range(1, epochs + 1):  # you only have
                # here create new set of captions for an image, datalaoder if not 1st epoch.
                if epoch == 1:
                    train_dataloader = initial_train_dataloader
                    train_per_img_dataset = base_train_per_img_dataset
                else:
                    prompts = get_prompts()
                    c_ablation = {
                        "id": 2,
                        "gen_num_captions": 10,
                        "choose_cap_per_prompt": True,
                    }  # always this right now.
                    train_per_img_dataset = per_image_caption_generation( ## TODO: create the caption dataset. use create_dataset_loader call here. 
                        self.dataset_type,
                        self.mm_model,
                        self.clip_model,
                        self.clip_processor,
                        self.clip_model_max_length,
                        img_idx,
                        img_path,
                        self.image_folder,
                        c_ablation,
                        prompts,
                        use_vllm=True
                    )
                    train_dataloader = DataLoader(
                        train_per_img_dataset,
                        shuffle=True,
                        batch_size=batch_size,
                        collate_fn=self.custom_collate_fn,
                        num_workers=0,
                    )

                train_per_img_dataset.set_mm_model(self.mm_model)

                per_img_train_results = per_epoch_results[epoch - 1]  # cause 1-index
                per_img_train_first_order_metrics = per_epoch_first_order_metrics[
                    epoch - 1
                ]
                self.mm_model.model.train() # cause you switch for inference. 
                assert self.mm_model.model.training == True

                for batch, img_paths in train_dataloader:
                    first_img_path = img_paths[0]
                    for (
                        img_path
                    ) in img_paths:  # all the img_paths here should be the same.
                        assert (
                            img_path == first_img_path
                        ), f"img_path: {img_path}, first image path: {first_img_path}"

                    global_train_step += 1

                    if "qwen" in self.model_type:
                        batch_input_ids, batch_attention_mask, batch_labels = batch
                        batch_input_ids = batch_input_ids.to(self.mm_model.model.device)
                        batch_attention_mask = batch_attention_mask.to(
                            self.mm_model.model.device
                        )
                        batch_labels = batch_labels.to(self.mm_model.model.device)
                        outputs = self.mm_model.model(
                            input_ids=batch_input_ids,
                            attention_mask=batch_attention_mask,
                            labels=batch_labels,
                        )
                        # print("outputs", outputs)
                    else:
                        batch = {
                            k: (
                                v.to(self.mm_model.model.device)
                                if k != "image_sizes"
                                else v
                            )
                            for k, v in batch.items()
                        }
                        outputs = self.mm_model.model(**batch)
                    # print('outputs',outputs)
                    loss = outputs.loss
                    # print('loss', loss)
                    loss.backward()

                    optimizer.step()
                    # lr_scheduler.step()
                    optimizer.zero_grad()

                    with self.wandb_lock:
                        wandb.log(
                            {
                                "learning_rate": start_learning_rate,
                                "loss": loss.item(),
                                "epoch": epoch,
                            }
                        )
                    progress_bar.update(1)

                train_per_img_dataset.set_mm_model(
                    self.mm_model
                )  # set after every epoch.

                # infer per epoch and # first order metric per epoch

                img_path = first_img_path
                qs_id, answer = self.infer_per_image(img_path=img_path)
                per_img_train_results[qs_id] = answer
                per_img_train_first_order_metrics[img_path] = (
                    train_per_img_dataset.first_order_metric_per_img(
                        img_path, metric_train=True
                    )
                )

                del train_dataloader  # delete every epoch.
                gc.collect()
                torch.cuda.empty_cache()

            end = time.time()
            print("Per image time: ", end - start)

            assert global_train_step == num_training_steps
            # clear things per image.
            del optimizer, initial_train_dataloader, self.mm_model.model, self.mm_model
            gc.collect()
            torch.cuda.empty_cache()

        # Calculate accuracy and first order metric for all the epochs.
        for i in range(0, epochs):
            calc_per_img_train_results = per_epoch_results[i]
            calc_per_img_train_first_order_metrics = per_epoch_first_order_metrics[i]
            mean_diff_fot, max_diff_fot = (
                train_per_img_dataset.train_first_order_metric(
                    calc_per_img_train_first_order_metrics
                )
            )
            train_accuracy = self.train_infer(train_results=calc_per_img_train_results)
            with self.wandb_lock:
                print("logging acc, first order metric for epoch", i + 1)
                wandb.log(
                    {"epoch": i + 1, "train_caption_first_order_metric": mean_diff_fot}
                )
                wandb.log(
                    {
                        "epoch": i + 1,
                        "train_max_caption_first_order_metric": max_diff_fot,
                    }
                )
                wandb.log({"epoch": i + 1, "train_accuracy": train_accuracy})

    def initialize_wandb(self):
        group = f"SSP_online_{self.adaptation_type}_{self.model_type}_dataset_{self.dataset_type}_TP_{self.training_parameters}"  # outside of training opt ablations.
        print("WandB initialized.")
        wandb.init(group=group)
        from threading import Lock

        wandb_lock = Lock()
        self.wandb_lock = wandb_lock

    def online_adapt_infer(
        self,
    ):  # save the results to "/local/zemel/nikita/all-multi-modals/online_learning/dataset/infer_pos/answers/"
        # freeze everything first.
        # Create new model per sweep.
        try:
            model_path = None
            if self.model_type == "llama":
                model_path = "/local/zemel/weights/Llama-3.2-11B-Vision-Instruct"

            if self.model_type == "llava":
                model_path = "/local/zemel/weights/llava-v1.6-mistral-7b-hf/"

            if self.model_type == "qwen":
                model_path = "/local/zemel/weights/Qwen-VL-Chat-qwen-7b"

            self.model_path = model_path
            self.initialize_wandb()

            if self.adaptation_type == "uda":
                self.initialize_model_and_freeze_parameters(quantized=False)
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
