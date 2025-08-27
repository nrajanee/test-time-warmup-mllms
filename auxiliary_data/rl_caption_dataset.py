from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import v2
import json
from utils import get_prompts
from auxiliary_data.caption_generation import CaptionGenerator
import wandb
import os
import numpy as np
from utils import get_prompts
from utils import NUM_AUX_LABELS_PER_IMAGE
from auxiliary_data.base_caption_dataset import BaseCaptionDataset
import torch


class RLCaptionDataset(BaseCaptionDataset):
    def __init__(
        self,
        caption_id,
        dataset_type,
        image_folder,
        train_images,
        val_images,
        train_unfiltered_captions_clip_scores,
        random_augment=False,
        len_set_of_prompts = 10):
        super().__init__(
            caption_id,
            dataset_type,
            image_folder,
            train_images,
            val_images,
            random_augment,
            len_set_of_prompts
        )

        self.train_unfiltered_captions_clip_scores = train_unfiltered_captions_clip_scores
        self.gen_num_caps_per_prompt = NUM_AUX_LABELS_PER_IMAGE  # gen n

    def set_mm_model(self, mm_model):
        self.mm_model = mm_model

    def get_prompt_caption_ids(self, prompt, captions, img_path):
        image = Image.open(img_path).convert("RGB")
        # generate captions for prompt using mm_model.
        # and get their rewards.
        #if self.rand_augment:
            # print("rand_aug")
            #augmenter = v2.RandAugment()
            #image = augmenter(image)
            
        prompt_caption_input_ids = []
        for c in captions:
            if self.mm_model.model_type == "qwen":
                inputs = self.get_inputs(prompt, img_path, c)
            else:
                inputs = self.get_inputs(prompt, image, c)

            prompt_caption_input_ids.append(self.mm_model.get_input_ids(inputs))

        return prompt_caption_input_ids

    def __getitem__(self, index):  # for a specific image and prompt?
        data = self.train_unfiltered_captions_clip_scores[index]
        prompt = data["prompt"]
        img_path = data["img_path"]
        captions= data["captions"]
        assert len(captions) ==  self.gen_num_caps_per_prompt # 10 captions per prompt. 
        clip_scores = data["clip_scores"] 
        prompt_caption_ids = self.get_prompt_caption_ids(prompt, captions, img_path)
        assert len(clip_scores[0].tolist()) == len(captions)
        return prompt_caption_ids, clip_scores, img_path

    def __len__(self):
        return len(self.train_unfiltered_captions_clip_scores)

    def get_dataset_info_for_config(self): 
        return {"caption_filtered_by_clip": "False", "caption id": self.id, "random_augment": self.rand_augment, "len_set_of_prompts": self.len_set_of_prompts}
