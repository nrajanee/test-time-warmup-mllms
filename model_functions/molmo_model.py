from model_functions.base_model import BaseModel
from utils import find_sublist
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForPreTraining, AutoProcessor
import torch
import copy


class MolmoModel:
    def __init__(self, model_type, model_path):
        super().__init__(model_type)

    def get_inputs(self, prompt, image, caption):
        pass

    def custom_collate_fn(self, batch):
        pass

    def get_tokenizer(self):
        pass

    def get_answer_for_question_img(self, qs, img_path):
        pass

    def get_llm_name(self):
        pass

    def get_connector_name(self):
        pass

    def get_vision_name(self):
        pass
