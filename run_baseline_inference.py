# Run inference for train_images and val_images
# get_question_mapping
# initialize model
#  self.mm_model.get_answer_for_question_img(qs, img_path)
import json
import torch
import os
from pycocotools.coco import COCO

from utils import get_image_to_questions_mapping, get_rel_paths
from qa_eval import get_accuracy

from model_functions.llama_model import LlamaVisionModel
from model_functions.llava_model import LlavaModel
from model_functions.qwen_model import QwenModel
from model_functions.gemma_model import GemmaModel
import numpy as np
import random
from utils import eval_obj_det, NUM_TRAIN_IMAGES, NUM_VAL_IMAGES

print("NUM_TRAIN_IMAGES", NUM_TRAIN_IMAGES)
print("NUM_VAL_IMAGES", NUM_VAL_IMAGES)

# deterministic behaviour
np.random.seed(30)
random.seed(30)
torch.manual_seed(30)
torch.cuda.manual_seed(30)
torch.cuda.manual_seed_all(30)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def initialize_mm_model(model_path, quantized=False):
    if "Llama" in model_path:
        return LlamaVisionModel("llama", model_path, quantized=quantized)
    if "llava" in model_path:
        return LlavaModel("llava", model_path, quantized=quantized)
    if "qwen" in model_path:
        return QwenModel("qwen", model_path, quantized=quantized)
    if "gemma" in model_path: 
        return GemmaModel("gemma", model_path, quantized=quantized)
    
    return None


import sys

model_type = sys.argv[1]
dataset_type = sys.argv[2]
print("model_type", model_type)
print("dataset_type", dataset_type)

question_or_obj = sys.argv[3]
print("question_or_obj", question_or_obj)

if model_type == "llama":
    model_path = "/local/zemel/weights/Llama-3.2-11B-Vision-Instruct"

if model_type == "qwen":
    model_path = "/local/zemel/weights/Qwen-VL-Chat-qwen-7b"

if model_type == "llava":
    model_path = "/local/zemel/weights/llava-v1.6-mistral-7b-hf/"

if model_type == "gemma": 
    model_path = "/local/zemel/weights/gemma-3-12b-it/"

print(model_path)

mm_model = initialize_mm_model(model_path, quantized=False)

image_questions_mapping = get_image_to_questions_mapping(dataset_type)

rel_paths = get_rel_paths(dataset_type)

train_rand_chosen_images_file = f"/local/zemel/nikita/all-multi-modals/online_learning/train_{dataset_type}_images.json"  # 100 for now.
with open(train_rand_chosen_images_file, "r") as f:
    train_images = json.load(f)  # {'0': img_file_name, ..}

train_images =  dict(list(train_images.items())[:NUM_TRAIN_IMAGES])
assert len(train_images.items()) == NUM_TRAIN_IMAGES # it's either one. 

val_images_file = f"/local/zemel/nikita/all-multi-modals/online_learning/val_{dataset_type}_images.json"
with open(val_images_file, "r") as f:
    val_images = json.load(f)  # {'0': img_file_name, ..}

val_images =  dict(list(val_images.items())[:NUM_VAL_IMAGES])
assert len(val_images.items()) == NUM_VAL_IMAGES

coco = None
if dataset_type == "vqav2" and question_or_obj == "obj":
    # Path to the COCO validation annotations JSON file
    coco_obj_label_file = rel_paths["obj_annotations_file"]
    # Load the COCO annotations
    coco = COCO(coco_obj_label_file)

scene_graph_data = None
if dataset_type == "gqa" and question_or_obj == "obj":
    scene_graph_file = rel_paths["obj_annotations_file"]
    with open(scene_graph_file, "r") as f:
        scene_graph_data = json.load(f)
        print(scene_graph_data[list(scene_graph_data.keys())[0]])


def get_base_model_answers_to_dataset_questions():
    train_results = {}
    num_imgs = 0
    print("TRAIN")
    
    for i, image_file in train_images.items():
        qs_id, qs = image_questions_mapping[image_file]
        print("question", qs)
        img_path = os.path.join(rel_paths["image_folder"], image_file)
        answer = mm_model.get_answer_for_question_img(qs, img_path)
        print("answer", answer)
        train_results[qs_id] = answer
        num_imgs += 1
        if num_imgs == NUM_TRAIN_IMAGES: # get rid of this later. 
            break

    all_train_imgs_train_accuracy = get_accuracy(
        rel_paths["annotation_file"], train_results, dataset_type
    )

    print(f"All train images - Train accuracy: {num_imgs}", all_train_imgs_train_accuracy)

    """
    first_250_train_results = dict(list(train_results.items())[:250])
    first_250_accuracy = get_accuracy(
        rel_paths["annotation_file"], first_250_train_results, dataset_type
    )

    print(f"First 250 train images - Train accuracy: {len(first_250_train_results.keys())}", first_250_accuracy)


    first_500_train_results = dict(list(train_results.items())[:500])

    first_500_accuracy = get_accuracy(
        rel_paths["annotation_file"], first_500_train_results, dataset_type
    )

    print(f"First 500 train images - Train accuracy: {len(first_500_train_results.keys())}", first_500_accuracy)



    train_results_file = f"/local/zemel/nikita/all-multi-modals/online_learning/baseline/{dataset_type}/{mm_model.model_type}/train_results.json"
    with open(train_results_file, "w") as f:
        json.dump(train_results, f)

    
    print("VAL")

    val_results = {}
    num_imgs = 0
    for i, image_file in val_images.items():
        qs_id, qs = image_questions_mapping[image_file]
        img_path = os.path.join(rel_paths["image_folder"], image_file)
        answer = mm_model.get_answer_for_question_img(qs, img_path)
        print("answer", answer)
        val_results[qs_id] = answer
        num_imgs += 1
        if num_imgs == NUM_VAL_IMAGES: 
            break

    val_accuracy = get_accuracy(rel_paths["annotation_file"], val_results, dataset_type)

    print(f"All val images - Val accuracy {num_imgs}", val_accuracy)

    val_results_file = f"/local/zemel/nikita/all-multi-modals/online_learning/baseline/{dataset_type}/{mm_model.model_type}/val_results.json"
    with open(val_results_file, "w") as f:
        json.dump(val_results, f)
    
    """


def ask_base_model_obj_det_questions():
    num_imgs = 0.0
    recall_score_objs_recalled = 0.0
    for i, image_file in train_images.items():
        img_path = os.path.join(rel_paths["image_folder"], image_file)
        model_objects = mm_model.get_answer_for_question_img(
            "What objects are visible in the image?", img_path, short_response=False
        )
        print("model objects", model_objects)
        img_objs_recalled = eval_obj_det(
            coco, scene_graph_data, dataset_type, img_path, model_objects
        )
        print("img_objs_recalled", img_objs_recalled)
        recall_score_objs_recalled += img_objs_recalled
        num_imgs += 1

    print("Train object detection accuracy", recall_score_objs_recalled / num_imgs)

    num_imgs = 0.0
    recall_score_objs_recalled = 0.0
    for i, image_file in val_images.items():
        img_path = os.path.join(rel_paths["image_folder"], image_file)
        model_objects = mm_model.get_answer_for_question_img(
            "What objects are visible in the image?", img_path, short_response=False
        )
        print("model objects", model_objects)
        img_objs_recalled = eval_obj_det(
            coco, scene_graph_data, dataset_type, img_path, model_objects
        )
        print("img_objs_recalled", img_objs_recalled)
        recall_score_objs_recalled += img_objs_recalled
        num_imgs += 1

    print("Val object detection accuracy", recall_score_objs_recalled / num_imgs)


if question_or_obj == "question":
    get_base_model_answers_to_dataset_questions()

if question_or_obj == "obj":
    ask_base_model_obj_det_questions()
