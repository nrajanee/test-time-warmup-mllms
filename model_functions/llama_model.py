from model_functions.base_model import BaseModel
from utils import find_sublist
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForPreTraining, AutoProcessor
import torch
import copy
from PIL import Image
import copy
import numpy as np
from utils import NUM_AUX_LABELS_PER_IMAGE
import random

np.random.seed(30)
random.seed(30)
torch.manual_seed(30)
torch.cuda.manual_seed(30)
torch.cuda.manual_seed_all(30)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class LlamaVisionModel(BaseModel):
    def __init__(self, model_type, model_path, quantized=False):
        print("model_type", model_type)
        super().__init__(model_type)
        if quantized: 
            print("quantized")
            self.model = AutoModelForPreTraining.from_pretrained(
            model_path, device_map="auto", torch_dtype=torch.bfloat16
        )
        else: 
            print("not quantized")
            self.model = AutoModelForPreTraining.from_pretrained(
                model_path, device_map="auto"
            )
        self.model_path = model_path
        if model_type == "gqa_best_epoch_llama" or "gqa_epoch_3_llama": 
            self.model_type = "llama"
        else: 
            self.model_type = model_type
        self.model.tie_weights()
        self.mm_processor = AutoProcessor.from_pretrained(model_path)
        print("model device", self.model.device)
        self.temperature = None
        self.max_new_tokens = 128

    def get_inputs(self, prompt, image, response):
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
            },
            {"role": "assistant", "content": [{"type": "text", "text": response}]},
        ]

        input_text = self.mm_processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )
        # print("input_text", input_text)
        inputs = self.mm_processor(
            image, input_text, add_special_tokens=False, return_tensors="pt"
        )
        return inputs
    
    def get_model_formatted_prompt(self, prompt):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"}, 
                    {"type": "text", "text": prompt}, 
                    ],
            }      ]

        input_text = self.mm_processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )
        return input_text

    def get_input_ids(self, inputs):
        return inputs.input_ids
    

    def _modify_single_item_in_rl_batch(self, item): 
        prompt_all_completion_ids, clip_scores, img_path = item
        #print("prompt_all_completion_ids", prompt_all_completion_ids[0].shape)
        start_caption_tokens = [128006, 78191, 128007, 271]
        # choose one for find_start_caption_idx. it'll be the same for others. 
        find_start_caption_idx = find_sublist(
            prompt_all_completion_ids[0][0].tolist(), start_caption_tokens
        )
        assert find_start_caption_idx != -1
        #print("find_start_caption_idx", find_start_caption_idx)
        # just for a quick sanity check
        find_start_caption_idx_2 = find_sublist(
            prompt_all_completion_ids[2][0].tolist(), start_caption_tokens
        )
        assert find_start_caption_idx_2 == find_start_caption_idx, "find start caption should be equal for all captions if same prompt and image"

        # make all of these a tensor first
        prompt_ids = prompt_all_completion_ids[0][:, :find_start_caption_idx] # choose one of them. 
        prompt_attention_mask = torch.ones(prompt_ids.size())
        all_completion_ids = [prompt_completion[:, find_start_caption_idx:] for prompt_completion in prompt_all_completion_ids]
        max_length = 0
        # also replace compl_ids image_token with 
        for compl_ids in all_completion_ids:
            compl_ids = compl_ids[0]
            if len(compl_ids) > max_length:
                max_length = len(compl_ids)
                    
        #print("max length", max_length)

        tokenizer = self.mm_processor.tokenizer
        append_val = tokenizer.pad_token_id
        padded_all_completed_ids = []

        # remove the image_token. 
        padded_completed_attention_mask = []
        for compl_ids in all_completion_ids: 
            padded_tensor = torch.cat(
                        (
                            compl_ids[0][:max_length],
                            torch.tensor(
                                [append_val] * (max_length - len(compl_ids[0])),
                                dtype=torch.int64,
                            ),
                        )
                    )
            padded_tensor = padded_tensor.expand(1, max_length)
            attention_tensor = torch.ones(compl_ids.size())
            padded_attention_tensor = torch.cat(
                (
                            attention_tensor[0][:max_length],
                            torch.tensor(
                                [append_val] * (max_length - len(compl_ids[0])),
                                dtype=torch.int64,
                            ),
                        )
            )

            padded_attention_tensor = padded_attention_tensor.expand(1, max_length) # models expect batch_size, len
            padded_all_completed_ids.append(padded_tensor)
            padded_completed_attention_mask.append(padded_attention_tensor)


        padded_all_completed_ids = torch.vstack(padded_all_completed_ids)
        #print("padded_all_completed_ids.shape", padded_all_completed_ids.shape)
        padded_completed_attention_mask = torch.vstack(padded_completed_attention_mask)
        #print("padded_completed_attention_mask.shape", padded_completed_attention_mask.shape)

        
        #print("prompt_ids", prompt_ids.shape)
        #print("prompt_attn_mask", prompt_attention_mask.shape)

        repeated_prompt_ids = prompt_ids.repeat(padded_all_completed_ids.shape[0], 1)
        repeated_prompt_attn_mask =  prompt_attention_mask.repeat(padded_all_completed_ids.shape[0], 1)

        padded_prompt_all_completion_ids = torch.cat(
            [repeated_prompt_ids, padded_all_completed_ids], dim=1
        )

        padded_prompt_all_completion_mask = torch.cat(
            [repeated_prompt_attn_mask, padded_completed_attention_mask], dim=1
        )
        
        #print("padded_prompt_all_completion_ids", padded_prompt_all_completion_ids.shape)
        #print("padded_prompt_all_completion_mask", padded_prompt_all_completion_mask.shape)
        inputs = {}
        inputs["prompt_completion_ids"] = padded_prompt_all_completion_ids
        inputs["prompt_ids"] = prompt_ids
        inputs["completion_ids"] = padded_all_completed_ids
        inputs["prompt_attn_mask"] = prompt_attention_mask
        inputs["completion_ids_attn_mask"] = padded_completed_attention_mask
        inputs["prompt_cids_attn_mask"] = padded_prompt_all_completion_mask
        inputs["clip_scores"] = clip_scores
        inputs["img_path"] = img_path



        return inputs


    
    def rl_custom_collate_fn(self, batch):
        padded_batch_with_attn_mask = []
        for item in batch: 
            padded_item_with_attn_mask = self._modify_single_item_in_rl_batch(item)
            padded_batch_with_attn_mask.append(padded_item_with_attn_mask)
            
        return padded_batch_with_attn_mask

    def custom_collate_fn(self, batch):
        #print("batch", batch)
        padded_batch_inputs_mapping = {}
        for key, _ in batch[0][0].items():
            padded_batch_inputs_mapping[key] = []
        
        max_length = 0
        for item in batch:
            input_ids = item[0].input_ids[0] # input_ids is a 2D tensor. 
            #print("input_ids in batch", input_ids)
            if len(input_ids) > max_length:
                max_length = len(input_ids)

        #print("max_length for padding", max_length)

        tokenizer = self.mm_processor.tokenizer
        # print("len batch", len(batch))
        for i in range(0, len(batch)):
            for key, i_t in batch[i][0].items():
                if key == "input_ids" or key == "attention_mask":
                    # print("i_t shape", i_t.shape)
                    # Pad or truncate the tensor
                    # print(key)
                    # print("shape of tresn", i_t.shape)
                    append_val = tokenizer.pad_token_id
                    if key == "attention_mask":
                        append_val = 0
                    padded_tensor = torch.cat(
                        (
                            i_t[0][:max_length],
                            torch.tensor(
                                [append_val] * (max_length - len(i_t[0])),
                                dtype=torch.int64,
                            ),
                        )
                    )
                    padded_tensor = padded_tensor.expand(1, max_length)

                    # print("padded tensor shape", padded_tensor.shape)
                    # print("padded_atch_inputs shape", padded_batch_inputs[i][0][key].shape)
                    assert torch.equal(
                        padded_tensor[0][0 : len(i_t[0])],
                        i_t[0],
                    ), "initial and non-pad tokens should equal"
                    # Append to the corresponding list in padded_batch
                    padded_batch_inputs_mapping[key].append(padded_tensor)

                elif (
                    key == "cross_attention_mask"
                ):  # cross_attention_mask shape eg: [1, 100, 1, 4] -> want to change second dim (input_ids). 4 is for 4 patch image embeddings.
                    pad_tensor = torch.zeros(
                        i_t.size(0),
                        max_length - i_t.size(1),
                        i_t.size(2),
                        i_t.size(3),
                    )
                    padded_tensor = torch.cat((i_t, pad_tensor), dim=1)
                    padded_batch_inputs_mapping[key].append(padded_tensor)

                else:  # if not input_ids or attention_mask then don't pad anything just include it.
                    # print("key", key)
                    # print("tensor shape", i_t.shape)
                    padded_batch_inputs_mapping[key].append(i_t)

        # print("pixel values shape", padded_batch_inputs[0][0]["pixel_values"].shape)
        # print("input ids", padded_batch_inputs[0][0]["input_ids"].shape)
        # print("cross attention mask", padded_batch_inputs[0][0]["cross_attention_mask"].shape)

        # start_of_caption_tokens = tokenizer.encode("<|start_header_id|>assistant<|end_header_id|>")

        start_caption_tokens = [128006, 78191, 128007, 271]
        padded_batch_inputs_mapping["labels"] = []

        # ignore all prompt and image tokens
        batch_labels = copy.deepcopy(padded_batch_inputs_mapping["input_ids"])
        for labels in batch_labels:
            find_start_caption_idx = find_sublist(
                labels[0].tolist(), start_caption_tokens
            )
            assert find_start_caption_idx != -1
            labels[0][0:find_start_caption_idx] = -100
            # print("find_start_caption_idx", find_start_caption_idx)
            labels[labels == tokenizer.pad_token_id] = -100
            image_tokens = [
                tokenizer.convert_tokens_to_ids(self.mm_processor.image_token)
            ]
            # the following might be unneccesary given the above.
            for image_token_id in image_tokens:
                labels[labels == image_token_id] = -100

            padded_batch_inputs_mapping["labels"].append(labels)

        for k, v in padded_batch_inputs_mapping.items():
            padded_batch_inputs_mapping[k] = torch.vstack(
                padded_batch_inputs_mapping[k]
            )

        assert len(batch) == len(padded_batch_inputs_mapping["input_ids"])

        # offload this batch:

        # print("batch[0].shape input_ids", batch[0][0]["input_ids"].shape)
        # print("padded_batch_inputs[0] input_ids", padded_batch_inputs_mapping["input_ids"].shape)
        # print("batch[0].shape attention_mask", batch[0][0]["attention_mask"].shape)
        # print("padded_batch_inputs[0] attention_mask", padded_batch_inputs_mapping["attention_mask"][0].shape)
        # print("batch[0].shape cross attention_mask", batch[0][0]["cross_attention_mask"][0].shape)
        # print("padded_batch_inputs[0] cross attention_mask", padded_batch_inputs_mapping["cross_attention_mask"][0].shape)

        img_paths = []
        for b in batch:
            img_paths.append(b[2])

        # print("img_paths in custom collate", img_paths)

        return padded_batch_inputs_mapping, img_paths

    def get_tokenizer(self):
        self.mm_processor.tokenizer

    
    def get_mmmu_prompt(self, qs, per_img_wsl_add_info):
        #print("question given to mmmu prompt", qs)
        if per_img_wsl_add_info is None: 
            per_img_wsl_add_info = ""
        else: 
            print("added info")
            per_img_wsl_add_info = f"{per_img_wsl_add_info} " # add the space after here. 

        prompt = f"""{qs}

Answer the above question following these guidelines. You must not deviate from the following guidelines. Strictly follow it. 

1. Be concise.
   - Provide a single word or phrase for the answer whenever possible. Follow the final answer format described below. 

2. For multiple-choice questions (A, B, C, D):
   - Respond only with the letter in parentheses (e.g., (A)).

3. If reasoning is needed:
   - Explain your thought process clearly and step by step.
   - Ensure logical progression in reasoning.
   - Don't end the response with reasoning. Ensure the answer is within the response in the format described below. 

4. Final Answer Format:
   - The correct answer must always be on the last line.
   - It should be either a single word/phrase or the option should be in parenthesis (for e.g., (A)) for multiple-choice.
   - Ensure the answer is within the response, if you are not sure take an intelligent guess. 
   - The answer should be be written after Correct answer: 
"""
        print("MMMU prompt", prompt)
        return prompt

    def get_answer_for_question_img(
        self, qs, img_path, short_response=True, per_img_wsl_add_info=None
    ):
        self.model.eval()
        if "MMMU" in img_path:  
            prompt = self.get_mmmu_prompt(qs, per_img_wsl_add_info)
            mmmu_max_new_tokens_for_answer = 512
            #print("in mmmu")
            mmmu_num_beams = 1
            mmmu_temperature = 0.3
        elif short_response: # This is for normal qa. 
            if per_img_wsl_add_info is not None:
                prompt = f"{per_img_wsl_add_info} {qs}\nAnswer the question using a single word or phrase."
            else:
                prompt = f"{qs}\nAnswer the question using a single word or phrase."
        else:  # better for obj det
            prompt = f"{qs}\nAnswer the question using a full sentence."

        image = Image.open(img_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
            }
        ]
        input_text = self.mm_processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self.mm_processor(
            image, input_text, add_special_tokens=False, return_tensors="pt"
        ).to(self.model.device)

        if "MMMU" not in img_path: 
            with torch.no_grad():
                # set model to eval mode here
                assert self.model.training == False
                input_ids = inputs.input_ids[0]
                output = self.model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=self.max_new_tokens,  # greedy, no temperature etc
                )
            
        else: 
             print("generating mmmu")
             with torch.no_grad():
                # set model to eval mode here
                assert self.model.training == False
                input_ids = inputs.input_ids[0]
                output = self.model.generate(
                    **inputs,
                    do_sample=True,
                    num_beams=mmmu_num_beams, 
                    temperature=mmmu_temperature,
                    top_k = 50, 
                    top_p = 0.8, 
                    max_new_tokens=mmmu_max_new_tokens_for_answer,  
                )

        output_ids = output[0][
            len(input_ids) :
        ]  # don't want to include input prompt here.
        answer = self.mm_processor.decode(output_ids, skip_special_tokens=True)
        return answer

    def get_caption(self, prompt, img_path):
        #print("model self.temperature", self.temperature)
        self.model.eval()
        image = Image.open(img_path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
            }
        ]
        input_text = self.mm_processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self.mm_processor(
            image, input_text, add_special_tokens=False, return_tensors="pt"
        ).to(self.model.device)

        # print("inputs.device", inputs.input_ids.device)
        # print("mm model device", self.mm_model.model.device)

        with torch.no_grad():
            input_ids = inputs.input_ids[0]
            assert self.model.training == False
            output = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=self.temperature,
                max_new_tokens=128,
            )

        output_ids = output[0][
            len(input_ids) :
        ]  # don't want to include input prompt here.
        caption = self.mm_processor.decode(output_ids, skip_special_tokens=True)

        return caption

    def get_llm_name(self):
        return "language_model"

    def get_connector_name(self):
        return "multi_modal_projector"

    def get_vision_name(self):
        return "vision_model"
