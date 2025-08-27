import sys
import random
import numpy as np
import copy
import torch
from transformers import AutoModelForPreTraining, AutoProcessor
import os
import json
from auxiliary_data.object_detection import offline_object_detection_qa_pairs
from utils import initialize_obj_det_model

dataset_type = sys.argv[1]
print(dataset_type, dataset_type)


# deterministic behaviour
np.random.seed(30)
random.seed(30)
torch.manual_seed(30)
torch.cuda.manual_seed(30)
torch.cuda.manual_seed_all(30)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def if_no_object_detection_generate(
    obj_det_model, obj_det_processor, dataset_type, image_folder
):
    train_images_file = f"/local/zemel/nikita/all-multi-modals/online_learning/train_{dataset_type}_images.json"  # 100 for now.
    train_images_file = f"/local/zemel/nikita/all-multi-modals/online_learning/train_{dataset_type}_images.json"  # 100 for now.
    with open(train_images_file, "r") as f:
        train_images = json.load(f)  # {'0': img_file_name, ..}

    val_images_file = f"/local/zemel/nikita/all-multi-modals/online_learning/val_{dataset_type}_images.json"
    with open(val_images_file, "r") as f:
        val_images = json.load(f)  # {'0': img_file_name, ..}

    offline_object_detection_qa_pairs(
        dataset_type,
        obj_det_model,
        obj_det_processor,
        train_images,
        val_images,
        image_folder,
    )


if dataset_type == "vqav2":
    question_file = (
        f"/local/zemel/nikita/datasets/vqav2/formatted_val2014_questions.jsonl"
    )
    image_folder = "/local/zemel/nikita/datasets/vqav2/val2014"

if dataset_type == "gqa":
    question_file = f"/local/zemel/nikita/datasets/gqa/val_balanced_questions.json"
    image_folder = "/local/zemel/nikita/datasets/gqa/images/"


obj_det_model, obj_det_processor = initialize_obj_det_model()

if_no_object_detection_generate(
    obj_det_model, obj_det_processor, dataset_type, image_folder
)
