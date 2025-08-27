from model_functions.base_model import BaseModel
from utils import find_sublist
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForPreTraining, AutoProcessor
import torch
import copy
from argparse import Namespace
from LLaVA.llava.mm_utils import get_model_name_from_path
from LLaVA.llava.model.builder import load_pretrained_model
from LLaVA.llava.eval.run_llava import get_model_inputs
from PIL import Image
from utils import NUM_AUX_LABELS_PER_IMAGE


class LlavaModel(BaseModel):
    def __init__(self, model_type, model_path):
        super().__init__(model_type)
        """
        self.args = Namespace(
            model_path=model_path, conv_mode=None
        )  # llava code infers the correct conv mode by itself.
        self.model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path, None, self.model_name
        )
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        """
        self.model_type = model_type
        self.model = AutoModelForPreTraining.from_pretrained(
            model_path, device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.temperature = None

    def get_inputs(self, prompt, image, response):

        conversation = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
            },
            {"role": "assistant", "content": [{"type": "text", "text": response}]},
        ]

        conv_prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = self.processor(
            images=[image], text=[conv_prompt], padding=True, return_tensors="pt"
        ).to(torch.float16)

        return inputs

    def get_input_ids(self, inputs):
        return inputs.input_ids

    def custom_collate_fn(self, batch):
        if batch[0][1] == "per_img_wsl":
            batch = batch[0][0]  # cause you want it to look like a batch of size 5.
            assert len(batch) == NUM_AUX_LABELS_PER_IMAGE, len(batch)

        padded_batch_inputs_mapping = {}
        # print("batch[0]", batch[0])
        for key, _ in batch[0][0].items():
            # print("key", key)
            padded_batch_inputs_mapping[key] = []

        # print("pixel values shape", padded_batch_inputs[0][0]["pixel_values"].shape)
        # print("input ids", padded_batch_inputs[0][0]["input_ids"].shape)
        # print("cross attention mask", padded_batch_inputs[0][0]["cross_attention_mask"].shape)
        max_length = 0
        for item in batch:
            # print("item", len(item))
            for input_ids in item[0].input_ids:
                if len(input_ids) > max_length:
                    max_length = len(input_ids)

        # print("max_length for padding", max_length)

        tokenizer = self.processor.tokenizer
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

                else:  # if not input_ids or attention_mask then don't pad anything just include it.
                    padded_batch_inputs_mapping[key].append(i_t)
                    # print("key", key)
                    # if key == 'image_sizes':
                    # print("image sizes len", len(i_t))
                    # else:
                    # print("tensor shape", i_t.shape)

        # start_of_caption_tokens = tokenizer.encode("[/INST]")

        start_caption_tokens = [733, 28748, 16289, 28793]
        padded_batch_inputs_mapping["labels"] = []

        # ignore all prompt and image tokens
        batch_labels = copy.deepcopy(padded_batch_inputs_mapping["input_ids"])
        for labels in batch_labels:
            # print("len of labels", len(labels))
            find_start_caption_idx = find_sublist(
                labels[0].tolist(), start_caption_tokens
            )
            assert find_start_caption_idx != -1
            labels[0][0:find_start_caption_idx] = -100
            # print("find_start_caption_idx", find_start_caption_idx)
            labels[labels == tokenizer.pad_token_id] = -100

            padded_batch_inputs_mapping["labels"].append(labels)

        # print("padded_batch_inputs_mapping", padded_batch_inputs_mapping["labels"])

        for k, v in padded_batch_inputs_mapping.items():
            # print("k", k)
            # print("v", len(v))
            if k == "image_sizes":
                concat_is = []
                for i_s in v:
                    concat_is += i_s
                padded_batch_inputs_mapping[k] = concat_is
            else:
                padded_batch_inputs_mapping[k] = torch.vstack(v)
        assert len(batch) == len(padded_batch_inputs_mapping["input_ids"])

        # print("padded_batch_inputs_mapping", padded_batch_inputs_mapping)

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

        return padded_batch_inputs_mapping, img_paths

    def get_tokenizer(self):
        return self.processor.tokenizer

    def get_answer_for_question_img(self, qs, img_path, short_response=True):
        self.model.eval()
        if short_response:
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
        input_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self.processor(
            image, input_text, add_special_tokens=False, return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            # set model to eval mode here
            assert self.model.training == False
            input_ids = inputs.input_ids[0]
            output = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=128,  # greedy, no temperature etc
            )

        output_ids = output[0][
            len(input_ids) :
        ]  # don't want to include input prompt here.
        answer = self.processor.decode(output_ids, skip_special_tokens=True)
        return answer

    def get_answer_for_question_img_with_attention_scores(
        self, qs, img_path, short_response=True
    ):
        self.model.eval()
        if short_response:
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
        input_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self.processor(
            image, input_text, add_special_tokens=False, return_tensors="pt"
        ).to(self.model.device)

        last_img_idx = 0
        first_img_idx = -1
        for i in range(0, len(inputs.input_ids[0])):
            if first_img_idx == -1 and inputs.input_ids[0][i] == 32000:
                first_img_idx = i
            if inputs.input_ids[0][i] == 32000:
                last_img_idx = i

        with torch.no_grad():
            # set model to eval mode here
            assert self.model.training == False
            input_ids = inputs.input_ids[0]
            output = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=128,  # greedy, no temperature etc
                return_dict_in_generate=True,
                output_attentions=True,
            )

        output_ids = output.sequences[0][
            len(input_ids) :
        ]  # don't want to include input prompt here.
        answer = self.processor.decode(output_ids, skip_special_tokens=True)
        return answer, output.attentions, first_img_idx, last_img_idx

    def get_caption(self, prompt, img_path):
        image = Image.open(img_path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
            }
        ]
        input_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self.processor(
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
        caption = self.processor.decode(output_ids, skip_special_tokens=True)

        return caption

    def get_llm_name(self):
        return "language_model"

    def get_connector_name(self):
        return "multi_modal_projector"

    def get_vision_name(self):
        return "vision_tower"
