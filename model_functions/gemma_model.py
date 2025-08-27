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

class GemmaModel(BaseModel):
    def __init__(self, model_type, model_path, quantized=False):
        super().__init__(model_type)
        self.model = AutoModelForPreTraining.from_pretrained(
                model_path, device_map="auto"
            )
        self.mm_processor = AutoProcessor.from_pretrained(model_path)
        print("model device", self.model.device)
        self.temperature = None
        self.max_new_tokens = 128
        self.model_path = model_path
        
    def get_inputs(self, prompt, image, response): 
        messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}],
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
    
    def get_input_ids(self, inputs):
        return inputs.input_ids
    

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
                if key == "input_ids" or key == "attention_mask" or key == "token_type_ids":
                    # print("i_t shape", i_t.shape)
                    # Pad or truncate the tensor
                    # print(key)
                    # print("shape of tresn", i_t.shape)
                    append_val = tokenizer.pad_token_id
                    if key == "attention_mask" or key == "token_type_ids":
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

                else:  # if not input_ids or attention_mask then don't pad anything just include it.
                    # print("key", key)
                    # print("tensor shape", i_t.shape)
                    padded_batch_inputs_mapping[key].append(i_t)

        #print("pixel values shape", padded_batch_inputs_mapping["pixel_values"][0.])
        #print("input ids", padded_batch_inputs[0][0]["input_ids"].shape)
        # print("cross attention mask", padded_batch_inputs[0][0]["cross_attention_mask"].shape)

        start_caption_tokens = [106,    107,    105,   4368, 107] 
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
            #print("key",k)
            padded_batch_inputs_mapping[k] = torch.vstack(
                padded_batch_inputs_mapping[k]
            )

        assert len(batch) == len(padded_batch_inputs_mapping["input_ids"])

        # offload this batch:

        #print("batch[0].shape input_ids", batch[0][0]["input_ids"].shape)
        #print("padded_batch_inputs[0] input_ids", padded_batch_inputs_mapping["input_ids"].shape)
        #print("batch[0].shape attention_mask", batch[0][0]["attention_mask"].shape)
        #print("padded_batch_inputs[0] attention_mask", padded_batch_inputs_mapping["attention_mask"][0].shape)
        #print("batch[0].shape token_type_ids", batch[0][0]["token_type_ids"].shape)
        #print("padded_batch_inputs[0] token_type_ids", padded_batch_inputs_mapping["token_type_ids"][0].shape)


        img_paths = []
        for b in batch:
            img_paths.append(b[2])

        # print("img_paths in custom collate", img_paths)

        return padded_batch_inputs_mapping, img_paths

    def get_tokenizer(self):
        self.mm_processor.tokenizer
    

    def get_tokenizer(self):
        self.mm_processor.tokenizer

    
    def get_mmmu_prompt(self, qs):
        #print("question given to mmmu prompt", qs)
        prompt = f"""{qs}

        Answer the question above by strictly following the guidelines below. Your main goal is to provide the correct answer in the response. Do not deviate from the guidelines below.

        1. Be Concise
        - Provide a single word or brief phrase for the answer whenever possible, adhering to the final answer format.

        2. Multiple-Choice (A, B, C, D)
        - Respond only with the correct letter in square brackets, for example, [A].

        3. Reasoning
        - Include reasoning where possible. 
        - Do not end your response with reasoning alone; always include the final answer as specified below.

        4. Final Answer Format
        - The correct answer must appear on the last line, preceded by the text: "Correct answer:"
        - The answer should be either a single word/phrase or a letter in square brackets (e.g., [A]).
        - If unsure, provide your best logical guess.
        """
        return prompt    

    def get_answer_for_question_img(
        self, qs, img_path, short_response=True, per_img_wsl_add_info=None
    ):
        self.model.eval()
        if "MMMU" in img_path: 
            prompt = self.get_mmmu_prompt(qs)
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
        #print("inputs device", inputs.input_ids.device)
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
             #print("generating mmmu")
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
        return "vision_tower"

        
    
    

