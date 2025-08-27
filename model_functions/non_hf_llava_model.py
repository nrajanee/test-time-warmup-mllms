from model_functions.base_model import BaseModel
from utils import find_sublist
from torch.nn.utils.rnn import pad_sequence
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
import copy
from argparse import Namespace
from LLaVA.llava.mm_utils import get_model_name_from_path
from LLaVA.llava.model.builder import load_pretrained_model
from LLaVA.llava.eval.run_llava import get_model_inputs
from PIL import Image


class LlavaModel(BaseModel):
    def __init__(self, model_type, model_path):
        super().__init__(model_type)
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

    def get_inputs(self, prompt, image, caption):
        self.args.query = prompt
        self.args.images = [image]
        self.args.answer = caption

        inputs = get_model_inputs(
            self.model, self.image_processor, self.tokenizer, self.args
        )

        return inputs

    def get_input_ids(self, inputs):
        return inputs["input_ids"]

    def custom_collate_fn(self, batch):
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
            for input_ids in item[0]["input_ids"]:
                if len(input_ids) > max_length:
                    max_length = len(input_ids)

        # print("max_length for padding", max_length)

        tokenizer = self.tokenizer
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

        print("padded_batch_inputs_mapping", padded_batch_inputs_mapping["labels"])

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

        return padded_batch_inputs_mapping

    def get_tokenizer(self):
        return self.tokenizer

    def get_answer_for_question_img(self, qs, img_path):
        self.args.query = f"{qs}\nAnswer the question using a single word or phrase."
        image = Image.open(img_path).convert("RGB")
        self.args.images = [image]
        self.args.answer = None

        inputs = get_model_inputs(
            self.model, self.image_processor, self.tokenizer, self.args
        )

        with torch.inference_mode():
            # set model to eval mode here.
            self.model.eval()
            assert self.model.training == False
            input_ids = inputs["input_ids"]
            images_tensor = inputs["images"]
            image_sizes = inputs["image_sizes"]
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

            output = self.model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=image_sizes,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=128,  # greedy, no temperature etc
            )

        output_ids = output[0]
        answer = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        print("answer", answer)
        return answer

    def get_llm_name(self):
        return "model"

    def get_connector_name(self):
        return "model.mm_projector"

    def get_vision_name(self):
        return "model.vision_tower"
