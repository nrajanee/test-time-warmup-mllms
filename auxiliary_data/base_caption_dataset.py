from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import v2
import json
from auxiliary_data.caption_generation import CaptionGenerator
from utils import intialize_clip_model
from utils import get_baseline_prompts
import wandb
import os
import numpy as np
from utils import NUM_TRAIN_IMAGES, NUM_VAL_IMAGES, NUM_AUX_LABELS_PER_IMAGE


class BaseCaptionDataset(Dataset):
    def __init__(
        self,
        caption_id,
        dataset_type,
        image_folder,
        train_images,
        val_images,
        random_augment,
        len_set_of_prompts
    ):
        self.id = caption_id
        self.dataset_type = dataset_type
        self.image_folder = image_folder
        self.train_images = train_images 
        self.val_images = val_images
        self.mm_model = None
        self.rand_augment = random_augment
        print("self.rand_augment", random_augment)
        #assert self.rand_augment == True # for now. 
        self.len_set_of_prompts = len_set_of_prompts

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

    def set_mm_model(self, mm_model):
        self.mm_model = mm_model
    
    def set_vllm_model(self, vllm_model): 
        self.vllm_model = vllm_model

    def get_inputs(self, prompt, image, caption):
        return self.mm_model.get_inputs(prompt, image, caption)

    def _add_to_baseline_img_mappings(
        self, prompts_to_check, baseline_img_captions_mapping, caption_gens
    ):
        for data in caption_gens:
            img_path = data["img_path"]  # choose one prompt here.
            if img_path not in baseline_img_captions_mapping:
                baseline_img_captions_mapping[img_path] = []
                baseline_img_captions_mapping[img_path].append(
                    []
                )  # 1 list for prompts (0)
                baseline_img_captions_mapping[img_path].append([])  # 1 for captions (1)
            caption = data["caption"]
            prompt = data["prompt"]
            if prompt in prompts_to_check:
                baseline_img_captions_mapping[img_path][0].append(prompt)
                baseline_img_captions_mapping[img_path][1].append(caption)

    def first_order_metric_per_img(self, img_path, metric_train, prompt_caption_mapping = None):
        baseline_img_captions_mapping, c_g = (
            self._get_baseline_img_mappings_and_caption_gen(metric_train)
        )
        c_g.prompts = baseline_img_captions_mapping[img_path][0]
        # print("c_g.prompts", c_g.prompts)
        assert len(c_g.prompts) == NUM_AUX_LABELS_PER_IMAGE
        if prompt_caption_mapping is None: 
            prompt_caption_mapping = c_g.get_clip_filtered_n_captions_based_on_img_prompts(img_path)
        captions = []
        for (
            prompt,
            caption,
        ) in prompt_caption_mapping:  # number of captions = choose_num_captions.
            captions.append(caption)

        baseline_chosen_captions = baseline_img_captions_mapping[img_path][1]
        assert len(baseline_chosen_captions) == NUM_AUX_LABELS_PER_IMAGE
        baseline_clip_scores_img = c_g.clip_score_of_captions(
            img_path, baseline_chosen_captions
        )[0].tolist()
        adapted_clip_scores_for_img = c_g.clip_score_of_captions(img_path, captions)[
            0
        ].tolist()

        # difference between the avg of the two.
        diff_avg_adapted_vs_baseline = np.mean(adapted_clip_scores_for_img) - np.mean(
            baseline_clip_scores_img
        )
        max_diff_adapted_vs_baseline = np.max(
            np.array(adapted_clip_scores_for_img) - np.array(baseline_clip_scores_img)
        )

        return diff_avg_adapted_vs_baseline, max_diff_adapted_vs_baseline

    
    """
    For a single image, this metric is calculated as the difference of the following: # For eg. if we have 50 captions for 1,2 and 3. 
    1. Sum of average CLIP score per image baseline: Compute the clip score of the randomly chosen 3 captions (from 50) per image and then take the average. 
    2. Sum of average CLIP per image TTT:  Compute the clip score of the randomly chosen 3 captions (from 50) captions per image and then take the average. 
    
    The caption first order metric then returns the sum of these values (TTT avg - baseline avg) so that negative differences are penalized and positive differences are rewarded.   
    """

    def _caption_first_order_metric_on_whole_dataset(self, metric_train=True):
        if metric_train:
            image_dataset = self.train_images
            assert len(image_dataset) == NUM_TRAIN_IMAGES
        else:
            image_dataset = self.val_images
            assert len(image_dataset) == NUM_VAL_IMAGES

        num_imgs = 0
        per_img_first_order_metrics = {}

        # TODO: vllm 25k do this for all images. 
        if self.vllm_model is not None: 
            gen_num_captions = 1  # only generate 1 caption per prompt
            # generate caption from finetuned model for the first caption dataset type (want to only assess multimodal ability and not involve CLIP).
            clip_model, clip_processor, clip_model_max_length = intialize_clip_model()
            choose_num_captions = NUM_AUX_LABELS_PER_IMAGE  # because you choose t
            #prompts = get_prompts()
            prompts = get_baseline_prompts()
            c_g = CaptionGenerator(
                self.mm_model,
                clip_model,
                clip_processor,
                clip_model_max_length,
                prompts,  
                gen_num_captions,
                choose_num_captions=choose_num_captions,
                choose_cap_per_prompt=None,  # not req since gen 1 cap per prompt.
                vllm_model=self.vllm_model
            )

            multi_model_fmt_imgs_prompts, img_path_prompt_list = c_g.get_multimodal_prompts_for_all_images(image_dataset=image_dataset, image_folder=self.image_folder)
            mapped_img_prompt_vllm_response = c_g.get_mapped_img_prompt_vllm_responses(multi_model_fmt_imgs_prompts, img_path_prompt_list)

            for img_path in mapped_img_prompt_vllm_response: 
                prompt_caption_mapping = []
                for prompt, caption in mapped_img_prompt_vllm_response[img_path].items(): 
                    # it's a list of len 1 
                    assert len(caption) == 1
                    caption = caption[0]
                    prompt_caption_mapping.append((prompt, caption))
                
                diff_avg_adapted_vs_baseline, max_diff_adapted_vs_baseline = (
                    self.first_order_metric_per_img(img_path, metric_train, prompt_caption_mapping = prompt_caption_mapping)
                )

                print("diff_avg", diff_avg_adapted_vs_baseline)
                per_img_first_order_metrics[img_path] = (
                    diff_avg_adapted_vs_baseline,
                    max_diff_adapted_vs_baseline,
                )
                num_imgs += 1
            
        else: 
            for i, image_file in image_dataset.items():
                img_path = os.path.join(self.image_folder, image_file)
                # difference between the avg of the two.
                diff_avg_adapted_vs_baseline, max_diff_adapted_vs_baseline = (
                    self.first_order_metric_per_img(img_path, metric_train)
                )
                print("diff_avg", diff_avg_adapted_vs_baseline)
                per_img_first_order_metrics[img_path] = (
                    diff_avg_adapted_vs_baseline,
                    max_diff_adapted_vs_baseline,
                )
                num_imgs += 1

        assert num_imgs == NUM_TRAIN_IMAGES or num_imgs == NUM_VAL_IMAGES, num_imgs  # train or val

        return per_img_first_order_metrics

    def _get_baseline_img_mappings_and_caption_gen(self, metric_train=True):
        # use all prompts in get_prompts
        prompts_to_check = get_baseline_prompts()

        assert len(prompts_to_check) == NUM_AUX_LABELS_PER_IMAGE
        baseline_img_captions_mapping = {}

        if metric_train:
            baseline_train_captions_gens_file = f"/local/zemel/nikita/all-multi-modals/online_learning/baseline/{self.dataset_type}/{self.mm_model.model_type}/baseline_train_1_c_abl.jsonl"
            baseline_train_captions_gens = [
                json.loads(data)
                for data in open(baseline_train_captions_gens_file, "r")
            ]
            self._add_to_baseline_img_mappings(
                prompts_to_check,
                baseline_img_captions_mapping,
                baseline_train_captions_gens,
            )

        else:
            baseline_val_captions_gens_file = f"/local/zemel/nikita/all-multi-modals/online_learning/baseline/{self.dataset_type}/{self.mm_model.model_type}/baseline_val_1_c_abl.jsonl"
            baseline_val_captions_gens = [
                json.loads(data) for data in open(baseline_val_captions_gens_file, "r")
            ]
            self._add_to_baseline_img_mappings(
                prompts_to_check,
                baseline_img_captions_mapping,
                baseline_val_captions_gens,
            )
        
        baseline_caption_len = len(baseline_img_captions_mapping.items())
        assert baseline_caption_len == 1000 #NUM_TRAIN_IMAGES or baseline_caption_len == NUM_VAL_IMAGES  # val or train # this will be 1000

        gen_num_captions = 1  # only generate 1 caption per prompt
        # generate caption from finetuned model for the first caption dataset type (want to only assess multimodal ability and not involve CLIP).
        clip_model, clip_processor, clip_model_max_length = intialize_clip_model()
        choose_num_captions = NUM_AUX_LABELS_PER_IMAGE  # because you choose t
        c_g = CaptionGenerator(
            self.mm_model,
            clip_model,
            clip_processor,
            clip_model_max_length,
            None,  # set prompts dynamically
            gen_num_captions,
            choose_num_captions=choose_num_captions,
            choose_cap_per_prompt=None,  # not req since gen 1 cap per prompt.
            vllm_model=self.vllm_model
        )

        return baseline_img_captions_mapping, c_g

    def train_first_order_metric(self, per_img_first_order_metrics=None):
        image_dataset = self.train_images
        assert len(image_dataset) == NUM_TRAIN_IMAGES
        if per_img_first_order_metrics is None:
            per_img_first_order_metrics = (
                self._caption_first_order_metric_on_whole_dataset(metric_train=True)
            )  # dictionary that saves path img_path: diff, max

        first_order_metric = 0
        max_first_order_metric = 0
        num_imgs = 0
        for img_path, metrics in per_img_first_order_metrics.items():
            diff_avg_adapted_vs_baseline, max_diff_adapted_vs_baseline = metrics
            first_order_metric += diff_avg_adapted_vs_baseline
            max_first_order_metric += max_diff_adapted_vs_baseline
            num_imgs += 1

        assert num_imgs == NUM_TRAIN_IMAGES, num_imgs  # train or val

        return first_order_metric / len(image_dataset), max_first_order_metric / len(
            image_dataset
        )  # average of the difference.

    def _val_first_order_metric(self):
        image_dataset = self.val_images
        per_img_first_order_metrics = self._caption_first_order_metric_on_whole_dataset(
            metric_train=False
        )  # dictionary that saves path img_path: diff, max

        first_order_metric = 0
        max_first_order_metric = 0
        num_imgs = 0
        for img_path, metrics in per_img_first_order_metrics.items():
            diff_avg_adapted_vs_baseline, max_diff_adapted_vs_baseline = metrics
            first_order_metric += diff_avg_adapted_vs_baseline
            max_first_order_metric += max_diff_adapted_vs_baseline
            num_imgs += 1

        assert num_imgs == NUM_VAL_IMAGES, num_imgs  

        return first_order_metric / len(image_dataset), max_first_order_metric / len(
            image_dataset
        )  # average of the difference.

    def log_initialize_first_order_metric(self, dataset_type=None):
        wandb.log({"epoch": 0, "train_caption_first_order_metric": 0})
        wandb.log({"epoch": 0, "train_max_caption_first_order_metric": 0})
        wandb.log({"epoch": 0, "val_caption_first_order_metric": 0})
        wandb.log({"epoch": 0, "val_max_caption_first_order_metric": 0})

    def log_initial_train_first_order_metric(self, dataset_type=None):
        wandb.log({"epoch": 0, "train_caption_first_order_metric": 0})
        wandb.log({"epoch": 0, "train_max_caption_first_order_metric": 0})

    def calculate_and_log_current_first_order_metric(
        self, epoch, per_img_train_first_order_metrics=None
    ):
        train_caption_first_order_metric, train_max_caption_first_order_metric = (
            self.train_first_order_metric(
                per_img_first_order_metrics=per_img_train_first_order_metrics
            )
        )
        val_caption_first_order_metric, val_max_caption_first_order_metric = (
            self._val_first_order_metric()
        )
        wandb.log(
            {
                "epoch": epoch,
                "train_caption_first_order_metric": train_caption_first_order_metric,
            }
        )
        wandb.log(
            {
                "epoch": epoch,
                "train_max_caption_first_order_metric": train_max_caption_first_order_metric,
            }
        )
        wandb.log(
            {
                "epoch": epoch,
                "val_caption_first_order_metric": val_caption_first_order_metric,
            }
        )
        wandb.log(
            {
                "epoch": epoch,
                "val_max_caption_first_order_metric": val_max_caption_first_order_metric,
            }
        )
