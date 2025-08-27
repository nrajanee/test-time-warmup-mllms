import torch
import os
from transformers import AutoModelForPreTraining, AutoProcessor
from auxiliary_data.caption_dataset import CaptionDataset
from object_detection_dataset import ObjectDetectionDataset
from utils import NUM_TRAIN_IMAGES, NUM_AUX_LABELS_PER_IMAGE, get_prompts
import numpy as np
import random
import json
from utils import get_image_to_questions_mapping
from model_functions.llama_model import LlamaVisionModel
from model_functions.llava_model import LlavaModel
from model_functions.qwen_model import QwenModel
from qa_eval import get_accuracy
from utils import BASELINE_ACCS

# deterministic behaviour
np.random.seed(30)
random.seed(30)
torch.manual_seed(30)
torch.cuda.manual_seed(30)
torch.cuda.manual_seed_all(30)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = "cuda" if torch.cuda.is_available() else "cpu"
import sys

## here you include all info for a given image.
adaptation_type = sys.argv[1]
model_type = sys.argv[2]
dataset_type = sys.argv[3]
weak_sup_label = sys.argv[4]
print("model_type", model_type)
print("dataset_type", dataset_type)
if weak_sup_label == "caption":  # or it can be object_det
    given_c_id = sys.argv[5]
    print("given caption id", given_c_id)

rand_aug = True
prompts = get_prompts()

def initialize_mm_model(model_path):
    print("model_path", model_path)
    if "Llama" in model_path:
        return LlamaVisionModel("llama", model_path)
    if "llava" in model_path:
        return LlavaModel("llava", model_path)
    if "qwen" in model_path:
        return QwenModel("qwen", model_path)
    return None


def get_train_dataset(mm_model, dataset_type, train_images, val_images):
    if weak_sup_label == "caption":
        c_abl_id = given_c_id
        print("c_abl_id", c_abl_id)
        offline_captions_gens_file = f"/local/zemel/nikita/all-multi-modals/online_learning/offline_train_caption_gens/{dataset_type}/{mm_model.model_type}/{len(prompts)}_setofprompts_{c_abl_id}_c_abl.jsonl"

        offline_caption_gens = [
            json.loads(data) for data in open(offline_captions_gens_file, "r")
        ]

        train_caption_gens = offline_caption_gens

        per_img_offline_caption_gens = []
        for i in range(0, NUM_TRAIN_IMAGES):
            per_img_offline_caption_gens.append([])
        prev_img_id = "0"
        img_idx = 0
        for c_gen in offline_caption_gens:
            img_id = c_gen["img_id"]
            print("img_id", img_id)
            if img_id == prev_img_id:
                per_img_offline_caption_gens[img_idx].append(c_gen)
                print("img_idx", img_idx)
            else:
                print("in here")
                img_idx += 1
                per_img_offline_caption_gens[img_idx].append(
                    c_gen
                )  # first w label for image.
                print("img_idx", img_idx)

            prev_img_id = img_id

            if int(img_id) == NUM_TRAIN_IMAGES - 1: # last one break
                break

        print(per_img_offline_caption_gens[1])
        assert len(per_img_offline_caption_gens) == NUM_TRAIN_IMAGES, len(
            per_img_offline_caption_gens
        )
        train_caption_gens = per_img_offline_caption_gens

        train_caption_dataset = CaptionDataset(
            adaptation_type,
            c_abl_id,
            dataset_type,
            image_folder,
            train_images,
            val_images,
            train_caption_gens,
            random_augment=rand_aug,
            len_set_of_prompts=len(prompts)
        )

        assert len(train_caption_dataset) == NUM_TRAIN_IMAGES
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
            )  # you don't need this anymore because you're doing dynamic batch size now by first putting all img info in batch of 1 and then changing it in get_item of dataset.
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
        train_obj_det_data = chosen_qa_pairs_by_img_id
        # print("chosen_qa_pairs_by_img_id", chosen_qa_pairs_by_img_id)
        train_obj_detection_dataset = ObjectDetectionDataset(
            None, train_obj_det_data, dataset_type, train_images, val_images
        )

        return train_obj_detection_dataset


from utils import get_rel_paths
rel_paths = get_rel_paths(dataset_type)
question_file = rel_paths["question_file"]
image_folder = rel_paths["image_folder"]

img_q_mapping = get_image_to_questions_mapping(dataset_type)

train_rand_chosen_images_file = f"/local/zemel/nikita/all-multi-modals/online_learning/train_{dataset_type}_images.json"
with open(train_rand_chosen_images_file, "r") as f:
    train_images = json.load(f)  # {'0': img_file_name, ..}



model_path = None
if model_type == "llama":
    model_path = "/local/zemel/weights/Llama-3.2-11B-Vision-Instruct"

if model_type == "llava":
    model_path = "/local/zemel/weights/llava-v1.6-mistral-7b-hf/"

if model_type == "qwen":
    model_path = "/local/zemel/weights/Qwen-VL-Chat-qwen-7b"


mm_model = initialize_mm_model(model_path)
train_dataset = get_train_dataset(mm_model, dataset_type, train_images, None)


def get_answer_for_question_img(
    mm_model,
    qs,
    img_path,
    per_img_wsl_add_info,
):  # ensure correct prompting here
    # ensure training is not on.
    return mm_model.get_answer_for_question_img(
        qs, img_path, per_img_wsl_add_info=per_img_wsl_add_info
    )


# just do it for the train dataset here for now cause you haven't generate weakly supervised labels for validation set which makes sense.
# train dataset


def infer_per_image(mm_model, img_path, per_img_wsl_add_info):
    image_file = img_path.split("/")[-1]
    qs_id, qs = img_q_mapping[image_file]
    img_path = os.path.join(image_folder, image_file)
    answer = get_answer_for_question_img(
        mm_model, qs, img_path=img_path, per_img_wsl_add_info=per_img_wsl_add_info
    )

    return qs_id, answer


def train_infer(train_results):
    annotations_file = rel_paths["annotation_file"] 

    assert train_results is not None

    train_accuracy = get_accuracy(annotations_file, train_results, dataset_type)

    return train_accuracy

per_img_train_results = {}
for index in range(0, train_dataset.__len__()):
    img_path, per_img_wsl_add_info = train_dataset.get_per_img_wsls_prompt(
        index
    )  # img_path, captions,
    print("per_img_wsl_add_info", per_img_wsl_add_info)
    qs_id, answer = infer_per_image(
        mm_model, img_path=img_path, per_img_wsl_add_info=per_img_wsl_add_info
    )
    print("answer", answer)
    per_img_train_results[qs_id] = answer


train_accuracy = train_infer(train_results=per_img_train_results)

print(
    "base accuracy for train dataset", BASELINE_ACCS[f"{model_type}_{dataset_type}"][0]
)

print("after in context for train dataset accuracy", train_accuracy)
