from PIL import Image
import torch
import numpy as np
from PIL import Image
import os
import json
import random
from utils import initialize_obj_det_model
from utils import NUM_AUX_LABELS_PER_IMAGE

# deterministic behaviour
np.random.seed(30)
random.seed(30)
torch.manual_seed(30)
torch.cuda.manual_seed(30)
torch.cuda.manual_seed_all(30)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class ObjectDetector:
    def __init__(self, obj_det_model, obj_det_processor):
        self.obj_det_model = obj_det_model
        self.obj_det_processor = obj_det_processor

    def get_obj_detection_results(self, image):
        processor, model = initialize_obj_det_model()
        if image.mode != "RGB":
            image = image.convert("RGB")

        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        # convert outputs (bounding boxes and class logits) to COCO API
        # let's only keep detections with score > 0.9
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.9
        )[0]

        detection_results = []
        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            box = [round(i, 2) for i in box.tolist()]
            print(
                f"Detected {model.config.id2label[label.item()]} with confidence "
                f"{round(score.item(), 3)} at location {box}"
            )

            d_res = {}
            d_res["class"] = model.config.id2label[label.item()]
            d_res["confidence"] = round(score.item(), 3)
            d_res["bbox"] = box
            detection_results.append(d_res)

        return detection_results

    def process_det_results_for_qa(self, detection_results):
        # Initialize the dictionary to store the results
        objects_image = {}

        # Process the detection results
        for result in detection_results:
            object_class = result["class"]
            bbox = result["bbox"]
            confidence = result["confidence"]

            # If the class is already in the dictionary, append the bbox
            if object_class in objects_image:
                objects_image[object_class][0] += 1  # Increase count
                objects_image[object_class][1].append(bbox)  # Add new bbox
            else:
                # If the class is not in the dictionary, add it with initial count and bbox
                objects_image[object_class] = [1, [bbox], confidence]

        return objects_image

    def generate_qa(self, image_objects, image_width, image_height):
        qa_pairs = []

        # Helper function to convert bounding box to natural language location
        def get_relative_position(xmin, ymin, xmax, ymax, image_width, image_height):
            horizontal_position = "center"
            vertical_position = "center"

            # Horizontal position: left, right, or center (narrower range for center)
            center_left = image_width * 0.40
            center_right = image_width * 0.60
            if xmax < center_left:
                horizontal_position = "left"
            elif xmin > center_right:
                horizontal_position = "right"
            elif xmin >= center_left and xmax <= center_right:
                horizontal_position = "center"

            # Vertical position: top, bottom, or center (narrower range for center)
            center_top = image_height * 0.40
            center_bottom = image_height * 0.60
            if ymax < center_top:
                vertical_position = "top"
            elif ymin > center_bottom:
                vertical_position = "bottom"
            elif ymin >= center_top and ymax <= center_bottom:
                vertical_position = "center"

            return horizontal_position, vertical_position

        # Function to generate a question based on object type and count
        def add_object_questions(object_type, count, bounding_boxes, confidence):
            if count > 1:
                qa_pairs.append(
                    {
                        "question": f"How many {object_type}s are in the image?",
                        "answer": f"There are {count} {object_type}s in the image.",
                    }
                )
            else:
                qa_pairs.append(
                    {
                        "question": f"Is there a {object_type} in the image?",
                        "answer": f"There is {count} {object_type} in the image.",
                    }
                )

            # Confidence-based question
            qa_pairs.append(
                {
                    "question": f"What is the confidence level of the {object_type} detection?",
                    "answer": f"The confidence level of the {object_type} detection is {confidence}.",
                }
            )

            # Location-based questions using bounding box
            for bbox in bounding_boxes:
                xmin, ymin, xmax, ymax = bbox
                horizontal_position, vertical_position = get_relative_position(
                    xmin, ymin, xmax, ymax, image_width, image_height
                )

                # Generate spatial relationship questions
                qa_pairs.append(
                    {
                        "question": f"Where is the {object_type} located in the image?",
                        "answer": f"The {object_type} is located towards the {horizontal_position} and {vertical_position} of the image.",
                    }
                )

                qa_pairs.append(
                    {
                        "question": f"Is the {object_type} closer to the left or right of the image?",
                        "answer": f"The {object_type} is closer to the {horizontal_position} side of the image.",
                    }
                )

                qa_pairs.append(
                    {
                        "question": f"Is the {object_type} closer to the top or bottom of the image?",
                        "answer": f"The {object_type} is closer to the {vertical_position} of the image.",
                    }
                )

                # Generating relationship-based questions (object-to-object)
                for other_object_type, (
                    other_count,
                    other_bboxes,
                    _,
                ) in image_objects.items():
                    if object_type != other_object_type:
                        for other_bbox in other_bboxes:
                            other_xmin, other_ymin, other_xmax, other_ymax = other_bbox
                            other_horizontal, other_vertical = get_relative_position(
                                other_xmin,
                                other_ymin,
                                other_xmax,
                                other_ymax,
                                image_width,
                                image_height,
                            )

                            if horizontal_position == other_horizontal:
                                horizontal_relation = "in line with"
                            elif (
                                horizontal_position == "left"
                                and other_horizontal == "right"
                            ):
                                horizontal_relation = "to the left of"
                            elif (
                                horizontal_position == "right"
                                and other_horizontal == "left"
                            ):
                                horizontal_relation = "to the right of"
                            else:
                                horizontal_relation = f"to the {horizontal_position} of"

                            if vertical_position == other_vertical:
                                vertical_relation = "in line with"
                            elif (
                                vertical_position == "top"
                                and other_vertical == "bottom"
                            ):
                                vertical_relation = "above"
                            elif (
                                vertical_position == "bottom"
                                and other_vertical == "top"
                            ):
                                vertical_relation = "below"
                            else:
                                vertical_relation = f"to the {vertical_position} of"

                            # Relationship-based question
                            qa_pairs.append(
                                {
                                    "question": f"Where is the {object_type} in relation to the {other_object_type}?",
                                    "answer": f"The {object_type} is {horizontal_relation} of the {other_object_type}, and {vertical_relation} it.",
                                }
                            )

        # Loop through the detected objects and generate questions
        for object_type, (count, bounding_boxes, confidence) in image_objects.items():
            add_object_questions(object_type, count, bounding_boxes, confidence)

        # Additional general questions (optional)
        if len(image_objects) > 0:
            qa_pairs.append(
                {
                    "question": "What objects are visible in the image?",
                    "answer": f"The visible objects are {', '.join(image_objects.keys())}.",
                }
            )

        return qa_pairs

    # Function to generate QA pairs for detected objects
    def dep_generate_qa(self, image_objects, image_width, image_height):
        qa_pairs = []

        # Helper function to convert bounding box to natural language location
        def get_relative_position(xmin, ymin, xmax, ymax, image_width, image_height):
            horizontal_position = "center"
            vertical_position = "center"

            # Horizontal position: left, right, or center
            if xmax < image_width / 3:
                horizontal_position = "left"
            elif xmin > 2 * image_width / 3:
                horizontal_position = "right"
            elif xmin < image_width / 3 and xmax > image_width / 3:
                horizontal_position = "center"

            # Vertical position: top, bottom, or center
            if ymax < image_height / 3:
                vertical_position = "top"
            elif ymin > 2 * image_height / 3:
                vertical_position = "bottom"
            else:
                vertical_position = "center"  # object is near the middle

            return horizontal_position, vertical_position

        # Function to generate a question based on object type and count
        def add_object_questions(object_type, count, bounding_boxes, confidence):
            if count > 1:
                qa_pairs.append(
                    {
                        "question": f"How many {object_type}s are in the image?",
                        "answer": f"There are {count} {object_type}s in the image.",
                    }
                )
            else:
                qa_pairs.append(
                    {
                        "question": f"Is there a {object_type} in the image?",
                        "answer": f"There is {count} {object_type} in the image.",
                    }
                )

            # Confidence-based question
            qa_pairs.append(
                {
                    "question": f"What is the confidence level of the {object_type} detection?",
                    "answer": f"The confidence level of the {object_type} detection is {confidence}.",
                }
            )

            # Location-based questions using bounding box
            for bbox in bounding_boxes:
                xmin, ymin, xmax, ymax = bbox
                horizontal_position, vertical_position = get_relative_position(
                    xmin, ymin, xmax, ymax, image_width, image_height
                )

                # Generate spatial relationship questions
                qa_pairs.append(
                    {
                        "question": f"Where is the {object_type} located in the image?",
                        "answer": f"The {object_type} is located towards the {horizontal_position}-{vertical_position} of the image.",
                    }
                )

                qa_pairs.append(
                    {
                        "question": f"Is the {object_type} closer to the left or right of the image?",
                        "answer": f"The {object_type} is closer to the {horizontal_position} side of the image.",
                    }
                )

                qa_pairs.append(
                    {
                        "question": f"Is the {object_type} closer to the top or bottom of the image?",
                        "answer": f"The {object_type} is closer to the {vertical_position} of the image.",
                    }
                )

        # Loop through the detected objects and generate questions
        for object_type, (count, bounding_boxes, confidence) in image_objects.items():
            add_object_questions(object_type, count, bounding_boxes, confidence)

        # Additional general questions (optional)
        if len(image_objects) > 0:
            qa_pairs.append(
                {
                    "question": "What objects are visible in the image?",
                    "answer": f"The visible objects are {', '.join(image_objects.keys())}.",
                }
            )

        return qa_pairs


def offline_object_detection_qa_pairs(
    dataset_type,
    obj_det_model,
    obj_det_processor,
    train_images,
    val_images,
    image_folder,
):
    o_d = ObjectDetector(
        obj_det_model,
        obj_det_processor,
    )

    train_file = f"/local/zemel/nikita/all-multi-modals/online_learning/offline_obj_detection/{dataset_type}/train.jsonl"
    val_file = f"/local/zemel/nikita/all-multi-modals/online_learning/offline_obj_detection/{dataset_type}/val.jsonl"

    if not os.path.exists(train_file):
        train_file = open(train_file, "w")
        qa_pair_idx = 0
        for img_idx, img_file in train_images.items():  # 100 for now.
            print("img_idx", img_idx, flush=True)
            img_path = os.path.join(image_folder, img_file)
            image = Image.open(img_path)
            detection_results = o_d.get_obj_detection_results(image)
            image_width, image_height = image.size
            objects_in_image = o_d.process_det_results_for_qa(detection_results)
            qa_pairs = o_d.generate_qa(objects_in_image, image_width, image_height)
            for pair in qa_pairs:
                question = pair["question"]
                answer = pair["answer"]
                train_file.write(
                    json.dumps(
                        {
                            "img_id": img_idx,
                            "img_path": img_path,
                            "obj_qa_pair_idx": qa_pair_idx,
                            "question": question,
                            "answer": answer,
                        }
                    )
                    + "\n"
                )
            qa_pair_idx += 1

        train_file.close()
    else:
        print(f"obj detection for train dataset_type: {dataset_type} exists.")

    if not os.path.exists(val_file):
        val_file = open(val_file, "w")
        qa_pair_idx = 0
        for img_idx, img_file in val_images.items():  # 100 for now.
            print("img_idx", img_idx, flush=True)
            img_path = os.path.join(image_folder, img_file)
            image = Image.open(img_path)
            detection_results = o_d.get_obj_detection_results(image)
            image_width, image_height = image.size
            objects_in_image = o_d.process_det_results_for_qa(detection_results)
            qa_pairs = o_d.generate_qa(objects_in_image, image_width, image_height)
            for pair in qa_pairs:
                question = pair["question"]
                answer = pair["answer"]
                val_file.write(
                    json.dumps(
                        {
                            "img_id": img_idx,
                            "img_path": img_path,
                            "obj_qa_pair_idx": qa_pair_idx,
                            "question": question,
                            "answer": answer,
                        }
                    )
                    + "\n"
                )

            qa_pair_idx += 1

        val_file.close()
    else:
        print(f"obj detection for val dataset_type: {dataset_type} exists.")
