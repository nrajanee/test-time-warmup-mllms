from model_functions.base_model import BaseModel
from utils import find_sublist
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForPreTraining, AutoProcessor, AutoModelForCausalLM
import torch
import copy
from PIL import Image
import copy
import numpy as np
import random
from qwen_vl_chat.qwen_generation_utils import get_inputs_from_query
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import NUM_AUX_LABELS_PER_IMAGE

np.random.seed(30)
random.seed(30)
torch.manual_seed(30)
torch.cuda.manual_seed(30)
torch.cuda.manual_seed_all(30)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class QwenModel(BaseModel):
    def __init__(self, model_type, model_path):
        super().__init__(model_type)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained( # TODO: try this AutoModelForPretraining?
            model_path, trust_remote_code=True, device_map="auto"
        )
        self.im_start, self.im_end = "<|im_start|>", "<|im_end|>"
        self.temperature = None

    def get_inputs(self, prompt, image_path, response):
        history = [(f"Picture 1: <img>{image_path}</img>\n{prompt}", response)]
        inputs = get_inputs_from_query(
            self.model, self.tokenizer, None, history=history
        )
        return inputs

    def get_input_ids(self, inputs):
        return inputs

    def custom_collate_fn(self, batch):
        # there's only input_ids
        if batch[0][1] == "per_img_wsl":
            batch = batch[0][0]  # cause you want it to look like a batch of size 5.
            assert len(batch) == NUM_AUX_LABELS_PER_IMAGE, len(batch)

        padded_batch_inputs = ([], [], [])  # input_ids, attention_mask, labels
        max_length = 0
        for item in batch:
            # print("item", item)
            for input_ids in item[0]:
                if len(input_ids) > max_length:
                    max_length = len(input_ids)
        # print("max_length", max_length)
        for i in range(0, len(batch)):  # there's only input-ids here
            input_ids = batch[i][0]
            # print("input_ids", input_ids)
            attention_mask = torch.ones(input_ids.shape)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.encode(self.im_end)[0]
            assert self.tokenizer.pad_token_id is not None
            append_input_val = self.tokenizer.pad_token_id
            append_attn_val = 0

            padded_input_tensor = torch.cat(
                (
                    input_ids[0][:max_length],
                    torch.tensor(
                        [append_input_val] * (max_length - len(input_ids[0])),
                        dtype=torch.int64,
                    ),
                )
            )

            padded_input_tensor = padded_input_tensor.expand(1, max_length)

            padded_attn_tensor = torch.cat(
                (
                    attention_mask[0][:max_length],
                    torch.tensor(
                        [append_attn_val] * (max_length - len(attention_mask[0])),
                        dtype=torch.int64,
                    ),
                )
            )

            padded_attn_tensor = padded_attn_tensor.expand(1, max_length)

            padded_batch_inputs[0].append(padded_input_tensor)
            padded_batch_inputs[1].append(padded_attn_tensor)

        copy_batch_inputs = copy.deepcopy(padded_batch_inputs[0])
        start_caption_tokens = [151644, 77091, 198]

        for i in range(0, len(copy_batch_inputs)):
            labels = copy_batch_inputs[i]
            find_start_caption_idx = find_sublist(
                labels[0].tolist(), start_caption_tokens
            )

            assert find_start_caption_idx != -1
            labels[0][0:find_start_caption_idx] = -100
            # print("find_start_caption_idx", find_start_caption_idx)
            labels[labels == self.tokenizer.pad_token_id] = -100
            padded_batch_inputs[2].append(labels)

        # print(padded_batch_inputs)
        final_padded_batch_inputs = [0, 1, 3]
        final_padded_batch_inputs[0] = torch.vstack(padded_batch_inputs[0])
        final_padded_batch_inputs[1] = torch.vstack(padded_batch_inputs[1])
        final_padded_batch_inputs[2] = torch.vstack(padded_batch_inputs[2])

        img_paths = []
        for b in batch:
            img_paths.append(b[2])

        # print("img_paths in custom collate", img_paths)
        return final_padded_batch_inputs, img_paths

    def get_answer_for_question_img(
        self, qs, img_path, short_response=True, per_img_wsl_add_info=None
    ):
        self.model.eval()
        if short_response:
            if per_img_wsl_add_info is not None:
                prompt = f"{per_img_wsl_add_info} {qs}\nAnswer the question using a single word or phrase."
            else:
                prompt = f"{qs}\nAnswer the question using a single word or phrase."

        else:  # better for obj det
            prompt = f"{qs}\nAnswer the question using a full sentence."

        query = self.tokenizer.from_list_format(
            [
                {"image": img_path},
                {"text": prompt},
            ]
        )

        inputs = get_inputs_from_query(self.model, self.tokenizer, query).to(
            self.model.device
        )

        with torch.no_grad():
            # set model to eval mode here
            assert self.model.training == False
            input_ids = inputs
            attention_mask = torch.ones(inputs.shape)
            output = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=128,  # greedy, no temperature etc
            )
        output_ids = output[0][
            len(input_ids[0]) :
        ]  # don't want to include input prompt here.

        answer = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return answer

    def get_caption(self, prompt, img_path):
        query = self.tokenizer.from_list_format(
            [
                {"image": img_path},
                {"text": prompt},
            ]
        )

        inputs = get_inputs_from_query(self.model, self.tokenizer, query).to(
            self.model.device
        )
        with torch.no_grad():
            # set model to eval mode here
            assert self.model.training == False
            input_ids = inputs
            attention_mask = torch.ones(inputs.shape)
            output = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=self.temperature,
                max_new_tokens=128,  # greedy, no temperature etc
            )

        output_ids = output[0][
            len(input_ids[0]) :
        ]  # don't want to include input prompt here.

        caption = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return caption

    def get_llm_name(self):
        return "transformer.h"

    def get_connector_name(self):
        return "transformer.attn_pool"

    def get_vision_name(self):
        return "transformer.visual"
