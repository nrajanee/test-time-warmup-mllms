from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import v2
import json
import wandb
import os
import numpy as np
from auxiliary_data.base_caption_dataset import BaseCaptionDataset


class CaptionDataset(BaseCaptionDataset):
    def __init__(
        self,
        adaptation_type,
        caption_id,
        dataset_type,
        image_folder,
        train_images,
        val_images,
        train_caption_gens,
        random_augment=False,
        len_set_of_prompts=10, 
    ):
        super().__init__(
            caption_id,
            dataset_type,
            image_folder,
            train_images,
            val_images,
            random_augment,
            len_set_of_prompts
        )

        self.adaptation_type = adaptation_type
        self.train_caption_gens = train_caption_gens
        self.mm_model = None

    def get_per_img_wsls_prompt(self, index):
        per_img_data = self.train_caption_gens[index]
        # print("per_img_data", per_img_data)
        first_img_path = per_img_data[0]["img_path"]
        concat_captions = ""  # ignore prompts for now.
        for i in range(0, len(per_img_data)):
            data = per_img_data[i]
            img_path = data["img_path"]
            assert (
                img_path == first_img_path
            ), f"img_path: {img_path}, first image path: {first_img_path}"  # all info for a single image.
            prompt = data["prompt"]
            caption = data["caption"].strip()
            concat_captions += f"{caption}"

        additional_info_prompt = f"Here are a detailed list of captions of the image: {concat_captions}. Answer the following question using these captions: "
        return first_img_path, additional_info_prompt

    def _retrieve_input_ids_labels(self, img_path, data):
        prompt = data["prompt"]
        caption = data["caption"].strip()
        image = Image.open(img_path).convert("RGB")
        if self.rand_augment:
            print("rand_aug", self.rand_augment)
            augmenter = v2.RandAugment()
            image = augmenter(image)

        if self.mm_model.model_type == "qwen":
            inputs = self.get_inputs(prompt, img_path, caption)
        else:
            inputs = self.get_inputs(prompt, image, caption)

        labels = self.mm_model.get_input_ids(inputs)

        return inputs, labels

    def __getitem__(self, index):
        data = self.train_caption_gens[index]
        img_path = data["img_path"]
        inputs, labels = self._retrieve_input_ids_labels(img_path, data)
        return inputs, labels, img_path

    def __len__(self):  # should always return total number of samples.
        return len(self.train_caption_gens)

    def get_dataset_info_for_config(self):
        return {"caption_filtered_by_clip": "True", "caption id": self.id, "random_augment": self.rand_augment, "len_set_of_prompts": self.len_set_of_prompts}

    def get_descr_for_group_name(self):
        return f"caption_CD_{self.id}"
