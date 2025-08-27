import random
import numpy as np
import torch
import json
from pathlib import Path
from PIL import Image
from utils import intialize_clip_model
from auxiliary_data.caption_generation import offline_caption_generation
from utils import get_prompts, get_rel_paths
from model_functions.llama_model import LlamaVisionModel
from model_functions.llava_model import LlavaModel
from model_functions.qwen_model import QwenModel
from model_functions.gemma_model import GemmaModel
import sys


model_type = sys.argv[1]
dataset_type = sys.argv[2]
use_siglip = sys.argv[3]

if use_siglip == "siglip": 
    use_siglip = True
else: 
    use_siglip = False

print("model_type", model_type)
print("dataset_type", dataset_type)
print("use siglip", use_siglip)

# deterministic behaviour
np.random.seed(30)
random.seed(30)
torch.manual_seed(30)
torch.cuda.manual_seed(30)
torch.cuda.manual_seed_all(30)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def if_no_caption_gens_generate(
    mm_model,
    clip_model,
    clip_processor,
    clip_model_max_length,
    dataset_type,
    image_folder,
):
    prompts = get_prompts()
    #prompts = larger_subset_get_prompts()
    assert len(prompts) == 10 
    caption_generator_ablations = [
       # {"id": 1, "gen_num_captions": 1, "choose_cap_per_prompt": True},
        {"id": 2, "gen_num_captions": 10, "choose_cap_per_prompt": True},
        # {"id": 3, "gen_num_captions": 10, "choose_cap_per_prompt": False},
        # {"id": 4, "gen_num_captions": 10, "choose_cap_per_prompt": False} # to check
    ]  # caption ablations

    train_images_file = f"/local/zemel/nikita/all-multi-modals/online_learning/train_{dataset_type}_images.json"  # 100 for now.
    with open(train_images_file, "r") as f:
        train_images = json.load(f)  # {'0': img_file_name, ..}

    val_images_file = f"/local/zemel/nikita/all-multi-modals/online_learning/val_{dataset_type}_images.json"
    with open(val_images_file, "r") as f:
        val_images = json.load(f)  # {'0': img_file_name, ..}

    offline_caption_generation(
        dataset_type,
        mm_model,
        clip_model,
        clip_processor,
        clip_model_max_length,
        train_images,
        val_images,
        image_folder,
        caption_generator_ablations,
        prompts)  # doesn't generate if it already exists.


def initialize_mm_model(model_path):
    if "Llama" in model_path:
        return LlamaVisionModel("llama", model_path)
    if "llava" in model_path:
        return LlavaModel("llava", model_path)
    if "qwen" in model_path:
        return QwenModel("qwen", model_path)
    
    if "gemma" in model_path: 
        return GemmaModel("gemma", model_path)

    return None


rel_paths = get_rel_paths(dataset_type)
question_file = rel_paths["question_file"]
image_folder = rel_paths["image_folder"]

clip_model, clip_processor, clip_model_max_length = intialize_clip_model(dataset=dataset_type, use_siglip=use_siglip)

if model_type == "llama":
    model_path = "/local/zemel/weights/Llama-3.2-11B-Vision-Instruct"

if model_type == "qwen":
    model_path = "/local/zemel/weights/Qwen-VL-Chat-qwen-7b"

if model_type == "llava":
    model_path = "/local/zemel/weights/llava-v1.6-mistral-7b-hf/"

if model_type == "gemma": 
    model_path =  "/local/zemel/weights/gemma-3-12b-it/"

print(model_path)


mm_model = initialize_mm_model(model_path)
if_no_caption_gens_generate(
    mm_model,
    clip_model,
    clip_processor,
    clip_model_max_length,
    dataset_type,
    image_folder
)
