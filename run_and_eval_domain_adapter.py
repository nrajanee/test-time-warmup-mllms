import random
import numpy as np
import copy
import torch
from transformers import AutoModelForPreTraining, AutoProcessor
import os
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
import pickle
from utils import get_prompts, get_rel_paths, get_image_to_questions_mapping
import json
from pathlib import Path
from PIL import Image
from auxiliary_data.caption_generation import offline_caption_generation
from torchvision.transforms import v2
from adapt_infer import AdaptInfer
from model_functions.llama_model import LlamaVisionModel
from model_functions.llava_model import LlavaModel
from utils import check_val_and_train_no_overlap
from auxiliary_data.caption_dataset import CaptionDataset
from object_detection_dataset import ObjectDetectionDataset
from utils import NUM_VAL_IMAGES, NUM_TRAIN_IMAGES, NUM_AUX_LABELS_PER_IMAGE
from online_adapt_infer import OnlineAuxAdaptInfer
from rl_adapt_infer import RLAdaptInfer

# deterministic behaviour
np.random.seed(30)
random.seed(30)
torch.manual_seed(30)
torch.cuda.manual_seed(30)
torch.cuda.manual_seed_all(30)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import wandb

wandb.util.logger.setLevel("DEBUG")

device = "cuda" if torch.cuda.is_available() else "cpu"
import sys

print("NUM_TRAIN_IMAGES", NUM_TRAIN_IMAGES)
print("NUM_VAL_IMAGES", NUM_VAL_IMAGES)

offline_online_aux_label_loss_fn = sys.argv[1]
adaptation_type = sys.argv[2]
model_type = sys.argv[3]
dataset_type = sys.argv[4]
weak_sup_label = sys.argv[5]
print("adaption_type", adaptation_type)
print("model_type", model_type)
print("dataset_type", dataset_type)
if weak_sup_label == "caption":  # or it can be object_det
    given_c_id = sys.argv[6]
    print("given caption id", given_c_id)

lora = sys.argv[7]
if lora == "use_lora": 
    lora = True

else: 
    lora = False

rel_paths = get_rel_paths(dataset_type)
question_file = rel_paths["question_file"]
image_folder = rel_paths["image_folder"]

len_set_of_prompts = len(get_prompts()) # can make this 7th arg. 

wandb.login()

questions = [json.loads(q) for q in open(os.path.expanduser(question_file), "r")]

rand_aug = True

check_val_and_train_no_overlap(dataset_type)

train_rand_chosen_images_file = f"/local/zemel/nikita/all-multi-modals/online_learning/train_{dataset_type}_images.json"
with open(train_rand_chosen_images_file, "r") as f:
    train_images = json.load(f)  # {'0': img_file_name, ..}

train_images =  dict(list(train_images.items())[:NUM_TRAIN_IMAGES])
assert len(train_images.items()) == NUM_TRAIN_IMAGES
for i in range(0, NUM_TRAIN_IMAGES): 
    assert train_images[str(i)] is not None

val_images_file = f"/local/zemel/nikita/all-multi-modals/online_learning/val_{dataset_type}_images.json"
with open(val_images_file, "r") as f:
    val_images = json.load(f)  # {'0': img_file_name, ..}

val_images =  dict(list(val_images.items())[:NUM_VAL_IMAGES])
assert len(val_images.items()) == NUM_VAL_IMAGES
for i in range(0, NUM_VAL_IMAGES): 
    assert val_images[str(i)] is not None

img_q_mapping = get_image_to_questions_mapping(dataset_type)


def get_train_dataset(adaptation_type, dataset_type, train_images, val_images):
    if weak_sup_label == "caption":
        c_abl_id = given_c_id
        print("c_abl_id", c_abl_id)
        offline_captions_gens_file = f"/local/zemel/nikita/all-multi-modals/online_learning/offline_train_caption_gens/{dataset_type}/{model_type}/{len_set_of_prompts}_setofprompts_{c_abl_id}_c_abl.jsonl"
        print("offline_captions_gens_file", offline_captions_gens_file)

        offline_caption_gens = [
            json.loads(data) for data in open(offline_captions_gens_file, "r")
        ][0:NUM_TRAIN_IMAGES*NUM_AUX_LABELS_PER_IMAGE]

        if adaptation_type == "tta":
            per_img_offline_caption_gens = []
            for i in range(0, NUM_TRAIN_IMAGES):
                per_img_offline_caption_gens.append([])
            prev_img_id = "0"
            img_idx = 0
            for c_gen in offline_caption_gens:
                img_id = c_gen["img_id"]
                if img_idx >= NUM_TRAIN_IMAGES: 
                    break 
                if img_id == prev_img_id:
                    per_img_offline_caption_gens[img_idx].append(c_gen)
                else:
                    img_idx += 1
                    per_img_offline_caption_gens[img_idx].append(
                        c_gen
                    )  # first w label for image.

                prev_img_id = img_id

            assert len(per_img_offline_caption_gens) == NUM_TRAIN_IMAGES
            train_caption_dataset = per_img_offline_caption_gens
            # print("train_caption_dataset", train_caption_dataset)

        elif adaptation_type == "uda":
            train_caption_gens = offline_caption_gens
            train_caption_dataset = CaptionDataset(
                adaptation_type,
                c_abl_id,
                dataset_type,
                image_folder,
                train_images,
                val_images,
                train_caption_gens,
                random_augment=rand_aug,
                len_set_of_prompts=len_set_of_prompts
            )

        else:
            raise ValueError("Adaption type is invalid", adaptation_type)

        if adaptation_type == "tta" or adaptation_type == "online_cont":
            assert len(train_caption_dataset) == NUM_TRAIN_IMAGES
        else:
            assert len(train_caption_dataset) / NUM_AUX_LABELS_PER_IMAGE == NUM_TRAIN_IMAGES

        return train_caption_dataset

    if weak_sup_label == "object_det":
        offline_obj_detection_file = f"/local/zemel/nikita/all-multi-modals/online_learning/offline_obj_detection/{dataset_type}/train.jsonl"
        offline_obj_detection_pairs = [
            json.loads(data) for data in open(offline_obj_detection_file, "r")
        ]

        qa_pairs_by_img_id = {}

        for det_pair in offline_obj_detection_pairs:
            img_id = int(det_pair["img_id"])
            if img_id not in qa_pairs_by_img_id:
                qa_pairs_by_img_id[img_id] = []
            qa_pairs_by_img_id[img_id].append(det_pair)

        chosen_offline_obj_detection_pairs = []
        chosen_qa_pairs_by_img_id = []
        objs_detected_imgs = 0
        for img_id in range(0, NUM_TRAIN_IMAGES):  # all images
            if img_id not in qa_pairs_by_img_id:
                print(f"no obj detection labels for image {img_id}")
                continue
            objs_detected_imgs += 1
            qa_pairs = qa_pairs_by_img_id[img_id]
            print("img_id", img_id)
            sample_size = min(
                NUM_AUX_LABELS_PER_IMAGE, len(qa_pairs)
            )  # max sample size is NUM_AUX_LABELS_PER_IMAGE
            assert (
                sample_size == NUM_AUX_LABELS_PER_IMAGE
            )  # otherwise tti/online won't work cause each batch only has to contain info about one image.
            random_qa_pairs = random.sample(qa_pairs, sample_size)
            for pair in random_qa_pairs:
                chosen_offline_obj_detection_pairs.append(pair)

            chosen_qa_pairs_by_img_id.append(random_qa_pairs)

        assert len(chosen_qa_pairs_by_img_id) == objs_detected_imgs, len(
            chosen_qa_pairs_by_img_id
        )
        # print(chosen_offline_obj_detection_pairs)
        # print("chosen_offline_obj_detection_pairs", len(chosen_offline_obj_detection_pairs))
        assert (
            len(chosen_offline_obj_detection_pairs)
            == objs_detected_imgs * NUM_AUX_LABELS_PER_IMAGE
        )
        train_obj_det_data = chosen_offline_obj_detection_pairs

        if adaptation_type == "tta":
            train_obj_detection_dataset = chosen_qa_pairs_by_img_id
            # print("chosen_qa_pairs_by_img_id", chosen_qa_pairs_by_img_id)
        elif adaptation_type == "uda":
            train_obj_detection_dataset = ObjectDetectionDataset(
                adaptation_type,
                train_obj_det_data,
                dataset_type,
                train_images,
                val_images,
            )
        else:
            raise ValueError("Invalid adaptation type", adaptation_type)

        return train_obj_detection_dataset


def test_time_adaptation():
    train_dataset = get_train_dataset(
        adaptation_type, dataset_type, train_images, None
    )  # we don't care about val_images here. just one dataset that's test.

    print("Test time adapt infer")
    # save per adaptation_type and captiion ablation and del the models. save answers to "/local/zemel/nikita/all-multi-modals/online_learning/dataset/infer_pos/answers/". and move on to next.
    training_parameters = ["connector", "llm"]
    a_i = AdaptInfer(
        model_type,
        adaptation_type,
        training_parameters,
        train_dataset,
        train_images,
        val_images,
        dataset_type,
        img_q_mapping,
        image_folder,
        len_set_of_prompts=len_set_of_prompts,
        lora=lora
    )

    learning_rate = 1e-6 # this lr did best in finetuning, llama vqav2 and gqa.
    epochs = 2 
    if lora == True: 
        learning_rate = 1e-4
        epochs = 5 # tried gemma and 2 before. 
    
    sweep_config = {
        "method": "grid",
        "parameters": {
            "learning_rate": {
                "values": [learning_rate]
            },  
            "bs_epochs": {
                "values": [
                    #{"batch_size": 2, "epochs": 4},
                    {"batch_size": 5, "epochs": epochs}, # 2 for now. 
                ]
            },
        },
    }

    sweep_id = wandb.sweep(sweep_config, project="multimodal-test-time-adaptation")
    wandb.agent(sweep_id, a_i.adapt_infer, count=9)


def unsupervised_domain_adaptation():
    train_dataset = None
    if offline_online_aux_label_loss_fn != 'rl_grpo': # because this is being done separately on adapt infer and isn't saved. 
        train_dataset = get_train_dataset(
            adaptation_type, dataset_type, train_images, val_images
        )

    print("UDA adapt infer")
    # save per adaptation_type and captiion ablation and del the models. save answers to "/local/zemel/nikita/all-multi-modals/online_learning/dataset/infer_pos/answers/". and move on to next.
    training_parameters = ["connector", "llm"]
    # method defined in offline caption. This is basically things that are not specific to the optimization in training.
    num_epochs = 5
    batch_size = 10
    learning_rate = 1e-6
    if offline_online_aux_label_loss_fn == "rl_grpo":
        learning_rate = 5e-6
        batch_size = 5

    sweep_config = {
        "method": "grid",
        "parameters": {
            "learning_rate": {
                "values": [learning_rate]
            },  # we know 1e-6 works well now so just use that . # 5e-6, 1e-6, 1e-5, 5e-7, 1e-7
            "batch_size": {"values": [batch_size]},
            "epochs": {"values": [num_epochs]},
        },
        # "metric": {'name': 'caption_first_order_max_metric', 'goal': 'maximize'}, # maximize this
    }

    sweep_id = wandb.sweep(sweep_config, project="multimodal-test-time-adaptation")
    if offline_online_aux_label_loss_fn == "offline":
        a_i = AdaptInfer(
            model_type,
            adaptation_type,
            training_parameters,
            train_dataset,
            train_images,
            val_images,
            dataset_type,
            img_q_mapping,
            image_folder,
            len_set_of_prompts=len_set_of_prompts
        )
        wandb.agent(sweep_id, a_i.adapt_infer, count=9)

    if offline_online_aux_label_loss_fn == "online":
        online_a_i = OnlineAuxAdaptInfer(
            model_type,
            adaptation_type,
            training_parameters,
            train_dataset,
            train_images,
            val_images,
            dataset_type,
            img_q_mapping,
            image_folder,
            len_set_of_prompts=len_set_of_prompts
        )
        wandb.agent(sweep_id, online_a_i.online_adapt_infer, count=9)
    

    if offline_online_aux_label_loss_fn == "rl_grpo": 
        rl_a_i = RLAdaptInfer(model_type, adaptation_type, training_parameters, None,  train_images, val_images, dataset_type, img_q_mapping, image_folder, len_set_of_prompts=len_set_of_prompts)
        wandb.agent(sweep_id, rl_a_i.rl_adapt_infer, count=9)


if adaptation_type == "tta":
    test_time_adaptation()

elif adaptation_type == "uda":
    unsupervised_domain_adaptation()

else:
    raise ValueError("Invalid adaptation type", adaptation_type)
