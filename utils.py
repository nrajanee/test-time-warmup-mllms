# map image to questions.
import os
import json
import ast
import random
from pathlib import Path
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from transformers import DetrImageProcessor, DetrForObjectDetection
from datasets import load_dataset, concatenate_datasets
from open_clip import create_model_from_pretrained, get_tokenizer
from transformers import AutoProcessor, AutoModel


np.random.seed(30)
random.seed(30)
torch.manual_seed(30)
torch.cuda.manual_seed(30)
torch.cuda.manual_seed_all(30)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
NUM_VAL_IMAGES =  350 #350 for mmmu (change)
NUM_TRAIN_IMAGES = 500 # 1000 
NUM_AUX_LABELS_PER_IMAGE = 10
NUM_PROMPTS_PER_IMAGE = 10

# 250 train images, 250 val images - prelim results. 

'''
BASELINE_ACCS = {
            "llama_vqav2": (0.7788, 0.73600),
            "llama_gqa": (0.612, 0.64),
            "llama_vqa-rad": (0.472, 0.46), 
            "gqa_best_epoch_llama_gqa": (0.62, 0.676),
            "gqa_epoch_3_llama_gqa": (0.608, 0.656), 
            "llama_textvqa": (0.7172, 0.7364)
        }  # train, val
'''
#1000 train images, 1000 val images

if NUM_TRAIN_IMAGES == 1000 and NUM_VAL_IMAGES==1000: 
    BASELINE_ACCS = {
        "llama_gqa": (0.623, 0.63),
        "llama_vqa-rad": (0.482, 0.452), 
        "llama_vqav2": (0.74689, 0.74989),
        "gemma_gqa": (-1, -1),
    }

# 500 train, 1000 val images. 

if NUM_TRAIN_IMAGES == 500 and NUM_VAL_IMAGES==1000: 
    BASELINE_ACCS = {
        "llama_gqa": (0.612, 0.63),
        "llama_vqa-rad": (0.492, 0.452), 
        "llama_vqav2": (0.7364, 0.74989), 
        "gemma_gqa": (0.498, -1), 
    }


# 250 train, 1000 val images. 
if NUM_TRAIN_IMAGES == 250 and NUM_VAL_IMAGES==1000: 
    BASELINE_ACCS = {
        "llama_vqa-rad": (0.492, 0.452),
        "llama_gqa": (0.612, 0.63), 
        "llama_vqav2": (0.7788, 0.74989)
    }


if NUM_TRAIN_IMAGES == 500 and NUM_VAL_IMAGES==350: # just for the mmmu dataset. 
    BASELINE_ACCS = {
        "llama_MMMU": (0.448, -1),
        "gemma_MMMU": (0.51, 0.49428)
    }

if NUM_TRAIN_IMAGES == 500: 
    BASELINE_ACCS = {
        "llama_MMMU": (0.448, -1),
        "llama_gqa": (0.612, -1),
        "llama_vqa-rad": (0.492, -1), 
        "llama_vqav2": (0.7364, -1), 
         "gemma_MMMU": (0.51, -1), 
        "gemma_gqa": (0.498, -1),
    }

def get_rel_paths(dataset_type):
    if dataset_type == "vqav2":
        rel_paths = {
            "question_file": f"/local/zemel/nikita/datasets/vqav2/formatted_val2014_questions.jsonl",
            "image_folder": "/local/zemel/nikita/datasets/vqav2/val2014",
            "annotation_file": "/local/zemel/nikita/datasets/vqav2/v2_mscoco_val2014_annotations.json",
            "obj_annotations_file": "/local/zemel/nikita/datasets/coco_annotations/instances_val2014.json",
        }

    if dataset_type == "gqa":
        rel_paths = {
            "question_file": f"/local/zemel/nikita/datasets/gqa/val_balanced_questions.json",
            "image_folder": "/local/zemel/nikita/datasets/gqa/images/",
            "annotation_file": "/local/zemel/nikita/datasets/gqa/questions1.2/val_balanced_questions.json",
            "obj_annotations_file": "/local/zemel/nikita/datasets/gqa/eval/val_sceneGraphs.json",
        }
    
    if dataset_type == "vqa-rad": 
        rel_paths = {
            "question_file": "/local/zemel/nikita/datasets/vqa-rad/train_test_dataset_questions.jsonl",
            "image_folder":"/local/zemel/nikita/datasets/vqa-rad/train_test_images",
            "annotation_file": None, # you get it from the dataset itself. 
        }

    if dataset_type == "MMMU": 
        rel_paths = {
            "question_file" : '/local/zemel/nikita/datasets/MMMU/validation_dataset_questions.jsonl', 
            "image_folder": "/local/zemel/nikita/datasets/MMMU/validation_images",
             "annotation_file": None
        }
    
    if dataset_type == "textvqa": 
        rel_paths = {
            "question_file" : '/local/zemel/nikita/datasets/textvqa/textvqa_val_v051_ocr.jsonl', 
            "image_folder": "/local/zemel/nikita/datasets/textvqa/train_and_val_images",
             "annotation_file": '/local/zemel/nikita/datasets/textvqa/TextVQA_0.5.1_val.json'
        }

    return rel_paths


# save this to /local/zemel/nikita/all-multi-modals/dataset/
def get_image_to_questions_mapping(dataset):
    rel_paths = get_rel_paths(dataset)
    questions = [
        json.loads(q) for q in open(os.path.expanduser(rel_paths["question_file"]), "r")
    ]
    img_questions_mapping = {}
    for line in questions:
        qs = line["text"]
        image = line["image"]
        q_id = line["question_id"]
        img_questions_mapping[image] = (q_id, qs)

    return img_questions_mapping


def get_questions_and_image_mapping(dataset): 
    rel_paths = get_rel_paths(dataset)

    questions = [
        json.loads(q) for q in open(os.path.expanduser(rel_paths["question_file"]), "r")
    ]
    question_and_images_mapping = {}
    for line in questions:
        qs = line["text"]
        image = line["image"]
        q_id = line["question_id"]
        question_and_images_mapping[q_id] = (image, qs)

    return question_and_images_mapping


def get_mmmu_dataset(): 
    categories = ['Accounting', 'Agriculture', 'Architecture_and_Engineering', 'Art', 'Art_Theory', 'Basic_Medical_Science', 'Biology', 'Chemistry', 'Clinical_Medicine', 'Computer_Science', 'Design', 'Diagnostics_and_Laboratory_Medicine', 'Economics', 'Electronics', 'Energy_and_Power', 'Finance', 'Geography', 'History', 'Literature', 'Manage', 'Marketing', 'Materials', 'Math', 'Mechanical_Engineering', 'Music', 'Pharmacy', 'Physics', 'Psychology', 'Public_Health', 'Sociology']
    datasets_list = [load_dataset("/local/zemel/nikita/datasets/MMMU/", category, split="validation") for category in categories]
    full_validation_set = concatenate_datasets(datasets_list)

    print(len(full_validation_set))
    return full_validation_set

def get_vqa_rad_dataset(): 
    train_ds = load_dataset("/local/zemel/nikita/datasets/vqa-rad")["train"] 
    test_ds = load_dataset("/local/zemel/nikita/datasets/vqa-rad")["test"]
    ds = concatenate_datasets([train_ds, test_ds])

    return ds

# format things to question_file and image_folder here. 
def format_torch_datasets(dataset): 
    image_folder = None
    question_file = None
    rel_paths = get_rel_paths(dataset)
    if dataset == "vqa-rad":
        train_ds = load_dataset("/local/zemel/nikita/datasets/vqa-rad")["train"] 
        test_ds = load_dataset("/local/zemel/nikita/datasets/vqa-rad")["test"]
        ds = concatenate_datasets([train_ds, test_ds])
        image_folder = rel_paths["image_folder"]
        question_file = rel_paths["question_file"]

        total_len = len(ds)
        print(total_len)
        # save to question_file and save the images to an image folder. 
    
    if dataset == "MMMU": 
        ds = get_mmmu_dataset()
        image_folder = rel_paths["image_folder"]
        question_file = rel_paths["question_file"]

    idx = -1
    question_file = open(question_file, "w")
    for item in ds: 
        idx+=1
        question_id = idx
        if dataset == "MMMU": 
            ds_id = item["id"]
            image_file = f"{ds_id}.jpg"
            image_path = f"{image_folder}/{image_file}"
            image = item["image_1"]
            other_image_not_none = False
            for i in range(2,8): 
                other_img_idx = f"image_{i}"
                other_image = item[other_img_idx]
                if other_image is not None: 
                    #print(item)
                    print("not none")
                    other_image_not_none = True
            if other_image_not_none: 
                continue

            text = item["question"]
            if item["question_type"] == 'multiple-choice': 
                print("options")
                option_labels = ["A", "B", "C", "D"]
                options = ast.literal_eval((item["options"]))
                formatted_options = "\n".join([f"({label}) {choice}" for label, choice in zip(option_labels,options)])
                text = f"Q: {text}\n{formatted_options}\n"

        else: # vqa-rad
            image_file = f"{question_id}.jpg"
            image_path = f"{image_folder}/{image_file}"
            text = item["question"]
            image = item["image"]
        
        image = image.convert("RGB")
        image.save(image_path)
        question_file.write(
                        json.dumps(
                            {
                                "question_id": question_id,
                                "image": image_file,
                                "text": text,
                            }
                        )
                        + "\n"
                    )
    
    question_file.close()

def format_text_vqa_question_file(): 
    rel_paths = get_rel_paths("textvqa")
    annotation_file = rel_paths["annotations"]
    annotation_file_open = open(annotation_file).read()
    annotations = json.loads(annotation_file_open)["data"]
    questions = [
        json.loads(q) for q in open(os.path.expanduser("/local/zemel/nikita/datasets/textvqa/incorr_textvqa_val_v051_ocr.jsonl", "r"))
    ]

    image_id_questions = {}
    for q in questions: 
        image_id_questions[q[question_id]] # this is actualyl image id. 
    for annotation in annotations: 
        image_file = annotation['image_id'] + '.jpg'
        question_id = annotation['question_id'] 


def save_random_chosen_images(dataset, image_folder):
    image_folder_path = Path(image_folder)
    all_image_files = [f.name for f in image_folder_path.iterdir() if f.is_file()]
    if dataset == 'textvqa': #because you only want val images. 
        with open("/local/zemel/nikita/datasets/textvqa/TextVQA_0.5.1_val.json", "r") as f: 
            textvqa_json = json.load(f)
        
        all_image_files = [item["image_id"] + '.jpg' for item in textvqa_json["data"]] # there are mul questions per image. 
    
    all_image_files = list(set(all_image_files)) # unique list 
    print("num of unique images", len(all_image_files))

    # get 2000 random images. 
    sample_files_idxs = random.sample(range(len(all_image_files)), NUM_TRAIN_IMAGES + NUM_VAL_IMAGES) 
    # split them into train and test. 
    train_split = sample_files_idxs[0:NUM_TRAIN_IMAGES]
    rand_imgs = [all_image_files[idx] for idx in train_split]
    rand_idx_img_mapping = {}
    idx = 0
    for img in rand_imgs:
        rand_idx_img_mapping[idx] = img
        idx += 1
    
    rand_chosen_images_file = f"/local/zemel/nikita/all-multi-modals/online_learning/train_{dataset}_images.json" 
    
    with open(rand_chosen_images_file, "w") as f:
        print("dump file", rand_chosen_images_file)
        json.dump(rand_idx_img_mapping, f)
    

    # and just use their subsets. 
    val_split = sample_files_idxs[NUM_TRAIN_IMAGES:NUM_TRAIN_IMAGES+NUM_VAL_IMAGES]
    rand_imgs = [all_image_files[idx] for idx in val_split]
    rand_idx_img_mapping = {}
    idx = 0
    for img in rand_imgs:
        rand_idx_img_mapping[idx] = img
        idx += 1

    rand_chosen_images_file = f"/local/zemel/nikita/all-multi-modals/online_learning/val_{dataset}_images.json"  # 100 for now.
    with open(rand_chosen_images_file, "w") as f:
        print("dump file", rand_chosen_images_file)
        json.dump(rand_idx_img_mapping, f)


def check_val_and_train_no_overlap(dataset):
    print('dataset', dataset)
    train_rand_chosen_images_file = f"/local/zemel/nikita/all-multi-modals/online_learning/train_{dataset}_images.json"  # 100 for now.
    with open(train_rand_chosen_images_file, "r") as f:
        train_images = json.load(f)  # {'0': img_file_name, ..}

    val_images_file = f"/local/zemel/nikita/all-multi-modals/online_learning/val_{dataset}_images.json"  # 100 for now.
    with open(val_images_file, "r") as f:
        val_images = json.load(f)  # {'0': img_file_name, ..}

    t_images_set = set(train_images.values())
    v_images_set = set(val_images.values())

    common_images = t_images_set & v_images_set
    print(common_images)

    print("Is disjoint", t_images_set.isdisjoint(v_images_set))
    assert t_images_set.isdisjoint(v_images_set), "They have an overlap"

def find_sublist(main_list, sub_list):
    # Check if the sub_list exists within the main_list
    for i in range(len(main_list) - len(sub_list) + 1):
        if main_list[i : i + len(sub_list)] == sub_list:
            return i  # Return the starting index
    return -1  # Return -1 if the sublist is not found


def intialize_clip_model(dataset='non-med', use_siglip=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if dataset == 'vqa-rad': 
        print("bio med clip")
        clip_model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        clip_processor = (preprocess, tokenizer)
        clip_model.to(device)
        print("bio clip model device", print(next(clip_model.parameters()).device))
        clip_model_max_length = 256
    elif use_siglip: 
        print("using siglip")
        clip_model_path =  "/local/zemel/weights/siglip-base-patch16-224"
        clip_processor = AutoProcessor.from_pretrained(clip_model_path)
        clip_model = AutoModel.from_pretrained(clip_model_path).to(device)
        clip_model_max_length = 64
    else: 
        print("open clip")
        clip_model_path = "/local/zemel/weights/clip-vit-large-patch14-336"
        clip_model = CLIPModel.from_pretrained(clip_model_path).to(device)
        print("Clip model device", clip_model.device)
        clip_processor = CLIPProcessor.from_pretrained(clip_model_path)
        clip_model_max_length = 77

    return clip_model, clip_processor, clip_model_max_length


def initialize_obj_det_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = DetrImageProcessor.from_pretrained(
        "/local/zemel/weights/detr-resnet-101/", revision="no_timm"
    )
    model = DetrForObjectDetection.from_pretrained(
        "/local/zemel/weights/detr-resnet-101/", revision="no_timm"
    ).to(device)

    return processor, model


def get_baseline_prompts():
    prompts = [
        "What is happening in this image?",
        "Describe the main subject of this image in detail.",
        "What objects or people are visible in this image?",
        "What actions are the subjects performing in this image?",
        "What does the background reveal about this image?",
        "What is unusual or unique about this image?",
        "What details in this image might someone easily overlook?",
        "Are there any signs, symbols, or text in this image? If so, what do they say?",
        "Explain the possible relationships or roles of the people, animals, or objects in this scene. What hints or clues suggest these relationships?",
        "Based on visual cues, infer what might have happened just before and what might happen right after this image was captured.",
    ]

    return prompts

def get_prompts():
    prompts = [
        "What is happening in this image?",
        "Describe the main subject of this image in detail.",
        "What objects or people are visible in this image?",
        "What actions are the subjects performing in this image?",
        "What does the background reveal about this image?",
        "What is unusual or unique about this image?",
        "What details in this image might someone easily overlook?",
        "Are there any signs, symbols, or text in this image? If so, what do they say?",
        "Explain the possible relationships or roles of the people, animals, or objects in this scene. What hints or clues suggest these relationships?",
        "Based on visual cues, infer what might have happened just before and what might happen right after this image was captured.",
    ]

    return prompts
    

'''
def get_prompts():
    prompts = [
    "What is happening in this image?",
    "Describe the main subject of this image in detail.",
    "What emotions does this image evoke, and why?",
    "What activity is taking place in this scene?",
    "Describe the environment or setting in the picture.",
    "What time of day do you think it is, based on this image?",
    "What objects or people are visible in this image?",
    "What could be the purpose of the items shown in this scene?",
    "What is the relationship between the people in this image?",
    "What can you infer about the location of this image?",
    "If this image were part of a story, what would the story be about?",
    "What mood does this image convey, and how do you know?",
    "What are the colors in the image, and how do they contribute to the overall composition?",
    "What actions are the subjects performing in this image?",
    "What event might this image be documenting?",
    "Are there any signs, symbols, or text in this image? If so, what do they say?",
    "Describe any animals in the image and their behavior.",
    "What cultural or historical context can you derive from this image?",
    "What objects are most prominent in this image, and why do you think that is?",
    "What might the photographer or creator of this image want to communicate?",
    "What type of scene or event does this image depict?",
    "What is unusual or unique about this image?",
    "If this were a still from a movie, what genre would it belong to?",
    "How might the weather or climate be affecting the scene in this image?",
    "Are there any notable textures or patterns in the image? Describe them.",
    "What kind of interaction is occurring between the subjects in the image?",
    "What perspective or angle was used to take this image?",
    "What does the background reveal about this image?",
    "What sounds might you associate with this image?",
    "Describe the clothing or attire visible in this image.",
    "What season do you think this image represents, and why?",
    "Are there any vehicles in this image? If so, describe them.",
    "What appears to be the focus or emphasis of this image?",
    "How does light or shadow affect the atmosphere in this image?",
    "If this were an advertisement, what product or message might it promote?",
    "What action just happened before this image was taken?",
    "What might happen next in this scene?",
    "How are space and distance used in this image?",
    "What does this image remind you of or resemble?",
    "Are there any reflections visible in this image? If so, describe them.",
    "What kind of story can you create based on this image?",
    "Are there any emotions being expressed by the people or animals in the image?",
    "What role does symmetry or asymmetry play in this image?",
    "What is the overall theme or concept of this image?",
    "Describe any man-made structures visible in the image.",
    "What would a child notice first in this image?",
    "If you could add something to this image, what would it be?",
    "What details in this image might someone easily overlook?",
    "How does this image make you feel, and why?",
    "What aspect of this image seems most important or striking to you?"
    ]

    return prompts
'''

def old_get_prompts():
    prompts = [
        "Please describe the image in 2-3 sentences",
        "Can you please provide a caption for the following image in 2-3 sentences?",
        "What is happening in this image in 2-3 sentences?",  # overall
        "What are the prominent objects and their positions in the image in 2-3 sentences?",
        "Please describe all the details in this image in 3-4 sentences.",
    ]  # detail-oriented
    return prompts


def eval_obj_det(coco, scene_graph_data, dataset_type, image_path, model_objects):
    if dataset_type == "vqav2":
        # use coco
        image_filename = os.path.basename(image_path).split(".")[0]
        numeric_id = int(image_filename.split("_")[-1])
        img_ids = coco.getImgIds(imgIds=[numeric_id])
        image_info = coco.loadImgs(img_ids)  # Retrieve image info using filename

        # If image is found, proceed with annotations retrieval
        if image_info:
            image_id = image_info[0]["id"]

            # Get annotations for this image
            ann_ids = coco.getAnnIds(imgIds=image_id)
            annotations = coco.loadAnns(ann_ids)

            # Get object labels (category names) for this image
            obj_labels = [
                coco.loadCats(ann["category_id"])[0]["name"] for ann in annotations
            ]

            print("Object labels in the image:", obj_labels)
            num_objs_recalled = 0.0
            for l in obj_labels:
                if l in model_objects:
                    num_objs_recalled += 1

            return num_objs_recalled / len(obj_labels)
        else:
            raise ValueError("image not found")

    if dataset_type == "gqa":
        image_id = image_path.split("/")[-1]
        image_id = image_id.split(".")[0]
        image_scene_graph = None
        image_scene_graph = scene_graph_data[image_id]["objects"]

        if image_scene_graph:
            obj_labels = [
                obj["name"] for obj_id, obj in image_scene_graph.items()
            ]  # Extract object names
            print(f"Object labels in the image:")
            print(obj_labels)
            num_objs_recalled = 0.0
            for l in obj_labels:
                if l in model_objects:
                    num_objs_recalled += 1

            return num_objs_recalled / len(obj_labels)
        else:
            print(f"Scene graph for image {image_path} not found.")

def is_token_match(pred_answer: str, gt: str) -> bool:
    """
    Checks if the predicted answer and ground truth answer match:
    - exact token match
    - or one is a subset of the other (token-wise)

    Assumes inputs are already normalized (e.g., lowercased, stripped of punctuation).
    """
    pred_tokens = set(pred_answer.strip().split())
    gt_tokens = set(gt.strip().split())

    if pred_tokens == gt_tokens:
        return True
    if gt_tokens.issubset(pred_tokens):
        return True
    if pred_tokens.issubset(gt_tokens):
        return True
    return False
