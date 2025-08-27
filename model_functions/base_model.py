class BaseModel:
    def __init__(self, model_type):
        self.model_type = model_type

    def get_inputs(self, prompt, image, response):
        pass

    def get_input_ids(self, inputs):
        pass

    def custom_collate_fn(self, batch):
        pass

    def get_tokenizer(self):
        pass

    def get_answer_for_question_img(self, qs, img_path):
        pass

    def get_caption(self, prompt, img_path):
        pass

    def get_llm_name(self):
        pass

    def get_connector_name(self):
        pass

    def get_vision_name(self):
        pass
