from PIL import Image
import torch
import numpy as np
from PIL import Image
import os
import json
import random
from utils import NUM_VAL_IMAGES, NUM_TRAIN_IMAGES, NUM_AUX_LABELS_PER_IMAGE
import gc

# deterministic behaviour
np.random.seed(30)
random.seed(30)
torch.manual_seed(30)
torch.cuda.manual_seed(30)
torch.cuda.manual_seed_all(30)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class CaptionGenerator:
    def __init__(
        self,
        mm_model,
        clip_model,
        clip_processor,
        clip_model_max_length,
        prompts,
        gen_num_captions,
        choose_num_captions=5,
        choose_cap_per_prompt=True,
        temperature=0.75,
        vllm_model=None
    ):
        super().__init__()
        self.mm_model = mm_model  
        # set model to eval here:
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self.clip_model_max_length = clip_model_max_length
        self.prompts = prompts
        self.gen_num_captions = gen_num_captions
        print("gen_num_captions", self.gen_num_captions)
        self.choose_num_captions = choose_num_captions
        print("choose_num_captions", self.choose_num_captions)
        self.choose_cap_per_prompt = choose_cap_per_prompt
        self.temperature = temperature
        self.mm_model.temperature = temperature
        self.vllm_model = vllm_model
        self.sampling_params = None
        self.use_vllm = False
        if vllm_model: 
            from vllm import SamplingParams
            self.sampling_params = SamplingParams(temperature=mm_model.temperature, max_tokens=mm_model.max_new_tokens,n=self.gen_num_captions)
            self.use_vllm = True


    def update_mm_model(
        self, mm_model
    ):  # this is so that you're updating the mm_model as you finetune.
        self.mm_model = mm_model

    def generate_caption_from_mm_model(
        self, prompt, img_path
    ):  
        assert self.mm_model.temperature == self.temperature
        self.mm_model.model.eval()
        assert self.mm_model.model.training == False
        caption = self.mm_model.get_caption(prompt, img_path)
        return caption
    
    
    def get_multimodal_prompts_for_all_images(self, image_folder, image_dataset): 
        chosen_prompts = self.prompts 
        #if len(chosen_prompts) > self.choose_num_captions: # figure out for 3 later. 
            #print("random sampling because larger set of prompts")
            #chosen_prompts = random.sample(self.prompts, self.choose_num_captions) # randomly choose 10 from 50 for eg. 
        
        #assert len(chosen_prompts) == self.choose_num_captions # because per prompt choosing from 10.

        multi_model_fmt_imgs_prompts = []
        img_path_prompt_list = []
        for i, image_file in image_dataset.items():
             img_path = os.path.join(image_folder, image_file)
             image = Image.open(img_path).convert("RGB")
             for prompt in chosen_prompts: 
                model_formatted_prompt = self.mm_model.get_model_formatted_prompt(prompt)
                multi_model_fmt_imgs_prompts.append({"prompt": model_formatted_prompt, "multi_modal_data": {"image": image}})
                img_path_prompt_list.append((img_path, prompt))
        
        return multi_model_fmt_imgs_prompts, img_path_prompt_list
    
    def get_mapped_img_prompt_vllm_responses(self, multi_model_fmt_imgs_prompts, img_path_prompt_list): 
        print("calling vllm generate once")
        outputs = self.vllm_model.generate(multi_model_fmt_imgs_prompts, self.sampling_params) 
        mapped_img_prompt_vllm_response = {}
        for i, output in enumerate(outputs): 
            img_path, prompt = img_path_prompt_list[i] # there 10 prompts per image. 
            if img_path not in mapped_img_prompt_vllm_response: 
                mapped_img_prompt_vllm_response[img_path] = {}
            
            mapped_img_prompt_vllm_response[img_path][prompt] = []
            num_gens = 0
            for j, generated in enumerate(output.outputs): 
                    gen_caption = generated.text 
                    mapped_img_prompt_vllm_response[img_path][prompt].append(gen_caption)
                    num_gens+=1
            assert num_gens == self.gen_num_captions
        

        return mapped_img_prompt_vllm_response


    def get_mapped_prompt_n_captions_for_img_prompts(
         self, img_path
    ): 
        
        chosen_prompts = self.prompts 
        #if len(chosen_prompts) > self.choose_num_captions: # figure out for 3 later. 
            ##print("random sampling because larger set of prompts")
            #chosen_prompts = random.sample(self.prompts, self.choose_num_captions) # randomly choose 10 from 50 for eg.
        
        #assert len(chosen_prompts) == self.choose_num_captions

        map_prompt_gen_caption = {}
        if self.use_vllm: 
            # use vllm
            image = Image.open(img_path).convert("RGB")
            #print("img_path", img_path)
            multi_model_fmt_prompts = []
            for prompt in chosen_prompts:
                model_formatted_prompt = self.mm_model.get_model_formatted_prompt(prompt)
                multi_model_fmt_prompts.append({"prompt": model_formatted_prompt, "multi_modal_data": {"image": image}})
            
    
            outputs = self.vllm_model.generate(multi_model_fmt_prompts, self.sampling_params)
            for i, output in enumerate(outputs): 
                prompt = chosen_prompts[i]
                map_prompt_gen_caption[prompt] = []
                num_gens = 0
                for j, generated in enumerate(output.outputs): 
                    gen_caption = generated.text 
                    map_prompt_gen_caption[prompt].append(gen_caption)
                    num_gens+=1
                
                assert num_gens == self.gen_num_captions
            
            #print("diff captions per prompt", map_prompt_gen_caption)


        else: 
            all_caption_idx = 0
            for prompt in chosen_prompts: 
                map_prompt_gen_caption[prompt] = []
                for i in range(0, self.gen_num_captions):
                    gen_caption = self.generate_caption_from_mm_model(prompt, img_path)
                    all_caption_idx += 1
                    map_prompt_gen_caption[prompt].append(gen_caption)

            assert all_caption_idx == self.choose_num_captions * self.gen_num_captions
        
        return map_prompt_gen_caption
    
    def get_unfiltered_n_captions_based_on_img_prompts_with_clip_scores(self, img_path): 
        map_prompt_gen_caption = self.get_mapped_prompt_n_captions_for_img_prompts(img_path)
        map_prompt_gen_caption_with_clip_scores = {}
        for prompt, captions in map_prompt_gen_caption.items(): 
            clip_scores = self.clip_score_of_captions(img_path, captions)
            #print("clip_scores.shape", clip_scores.shape)
            map_prompt_gen_caption_with_clip_scores[prompt] = (captions, clip_scores)
        
        return map_prompt_gen_caption_with_clip_scores

    def get_clip_filtered_n_captions_based_on_img_prompts(
        self, img_path
    ):  # 1 or 10 gen n captions captions  ablations. if 1 no need to use clip
        if self.gen_num_captions == 1:
            prompt_captions = []
            # print("self.prompts", self.prompts)
            map_prompt_gen_caption = self.get_mapped_prompt_n_captions_for_img_prompts(img_path) 
            for prompt, gen_captions in map_prompt_gen_caption.items():
                assert len(gen_captions) == 1
                gen_caption = gen_captions[0]
                prompt_captions.append((prompt, gen_caption))
            
            #print("prompt_captions", prompt_captions)
            assert len(prompt_captions) == self.choose_num_captions
            return prompt_captions

        else:  # more than 1 then you need to strategically pick the captions you finetune on using clip
            map_prompt_gen_caption = self.get_mapped_prompt_n_captions_for_img_prompts(img_path)
            clip_chosen_captions_prompt = self.use_clip_to_choose_best_captions(
                img_path, map_prompt_gen_caption
            )
            return clip_chosen_captions_prompt

    def use_clip_to_choose_best_captions(self, img_path, map_prompt_gen_caption):
        # choose 1 per prompt or choose_num_captions best overall
        best_prompt_captions = []
        if self.choose_cap_per_prompt == True:
            #print("clip per prompt best caption")
            for prompt in map_prompt_gen_caption:
                clip_scores = self.clip_score_of_captions(
                    img_path, map_prompt_gen_caption[prompt]
                )
                max_clip_cap_idx = torch.argmax(clip_scores)
                best_prompt_captions.append(
                    (prompt, map_prompt_gen_caption[prompt][max_clip_cap_idx])
                )

            assert len(best_prompt_captions) == self.choose_num_captions
        else:  # return choose_num_captions best overall --> here using a tuple.
            #print(f"return {self.choose_num_captions} best overall")
            all_captions = []
            cap_to_prompt = {}
            for p, p_caps in map_prompt_gen_caption.items():
                for cap in p_caps:
                    all_captions.append(cap)
                    cap_to_prompt[cap] = p
            clip_scores = self.clip_score_of_captions(img_path, all_captions)
            _, max_clip_indices = torch.topk(clip_scores, self.choose_num_captions)
            max_clip_indices = max_clip_indices[0].tolist()
            best_captions = [all_captions[i] for i in max_clip_indices]
            # print("best_captions", best_captions)
            # reverse figure out the prompt
            assert (len(best_captions)) == self.choose_num_captions
            for b_c in best_captions:
                prompt = cap_to_prompt[b_c]
                best_prompt_captions.append((prompt, b_c))
            # print(f"best_prompt_cs: {best_prompt_captions}")
            assert len(best_prompt_captions) == self.choose_num_captions

        return best_prompt_captions

    def clip_score_of_captions(self, img_path, captions):
        img = Image.open(img_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        if "vqa-rad" in img_path and len(self.clip_processor) == 2: # bio_med_clip
            preprocess, tokenizer = self.clip_processor
            images = torch.stack([preprocess(img)]).to("cuda")
            texts = tokenizer([l for l in captions], context_length=self.clip_model_max_length).to("cuda")
            image_features, text_features, logit_scale = self.clip_model(images, texts)
            #print("logit_scale", logit_scale)
        else: 
            inputs = self.clip_processor(
                text=captions,
                images=img,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.clip_model_max_length,
            ).to("cuda")
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                image_features = outputs.image_embeds
                text_features = outputs.text_embeds

        # Normalize the features
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

        # Compute the similarity

        captions_similarity = (
            100.0 * image_features @ text_features.T
        )  # .softmax(dim=-1)

        print("captions", captions)
        print(captions_similarity)
        return captions_similarity

# FOR GRPO. 
def train_dataset_unfiltered_for_RL_online_caption_generation(
        dataset_type, 
        mm_model,
        clip_model,
        clip_processor,
        clip_model_max_length,
        train_images, 
        image_folder, 
        c_ablation, 
        prompts, 
        use_vllm=False, 
        vllm_model=None): 
    
    gen_num_captions = c_ablation["gen_num_captions"]
    if use_vllm: 
        c_g = CaptionGenerator(
            mm_model,
            clip_model,
            clip_processor,
            clip_model_max_length,
            prompts,
            gen_num_captions, 
            choose_num_captions=None, # no filtering
            choose_cap_per_prompt=None,
            vllm_model=vllm_model,
        )

        caption_gen_idx = 0
        train_unfiltered_captions_clip_scores = []
    
        multi_model_fmt_imgs_prompts, img_path_prompt_list = c_g.get_multimodal_prompts_for_all_images(image_dataset=train_images, image_folder=image_folder)
        mapped_img_prompt_vllm_response = c_g.get_mapped_img_prompt_vllm_responses(multi_model_fmt_imgs_prompts, img_path_prompt_list)
        img_idx = 0
        for image_path in mapped_img_prompt_vllm_response:
            print("img_idx", img_idx, flush=True)
            for prompt, captions in mapped_img_prompt_vllm_response[image_path].items(): 
                clip_scores = c_g.clip_score_of_captions(image_path, captions)
                #print('clip_scores', clip_scores)
                #print('captions', captions)
                train_unfiltered_captions_clip_scores.append(
                {
                    "img_id": img_idx,
                    "img_path": image_path,
                    "c_id": caption_gen_idx,
                    "prompt": prompt,
                    "captions": captions,
                    "clip_scores": clip_scores,
                }
            )
            img_idx+=1

            caption_gen_idx += 1
        
        assert img_idx == len(train_images.keys())

    else: 
        raise ValueError('it should use vllm')

    '''
    caption_gen_idx = 0
    train_unfiltered_captions_clip_scores = []
    for img_idx, img_file in train_images.items():  # 100 for now.
        print("img_idx", img_idx, flush=True)
        img_path = os.path.join(image_folder, img_file)
        prompt_caption_clip_scores_mapping = c_g.get_unfiltered_n_captions_based_on_img_prompts_with_clip_scores(img_path)
        for prompt, caption_clip_scores in prompt_caption_clip_scores_mapping.items():
            train_unfiltered_captions_clip_scores.append(
                {
                    "img_id": img_idx,
                    "img_path": img_path,
                    "c_id": caption_gen_idx,
                    "prompt": prompt,
                    "captions": caption_clip_scores[0],
                    "clip_scores": caption_clip_scores[1],
                }
            )

            caption_gen_idx += 1
        
        #if int(img_idx) == 2:
            #break
    '''
    #print(train_unfiltered_captions_clip_scores)
    return train_unfiltered_captions_clip_scores



def train_dataset_online_caption_generation(
    dataset_type,
    mm_model,
    clip_model,
    clip_processor,
    clip_model_max_length,
    train_images, 
    image_folder,
    c_ablation,
    prompts,use_vllm=False,
    vllm_model=None, 
):
    gen_num_captions = c_ablation["gen_num_captions"]
    choose_cap_per_prompt = c_ablation["choose_cap_per_prompt"]
    choose_num_captions = len(prompts)
    assert choose_num_captions == NUM_AUX_LABELS_PER_IMAGE

    if use_vllm: 
        c_g = CaptionGenerator(
            mm_model,
            clip_model,
            clip_processor,
            clip_model_max_length,
            prompts,
            gen_num_captions,
            choose_num_captions=choose_num_captions,
            choose_cap_per_prompt=choose_cap_per_prompt,
            vllm_model=vllm_model,
        )

        caption_gen_idx = 0
        train_dataset_caption_gens = []
    
        multi_model_fmt_imgs_prompts, img_path_prompt_list = c_g.get_multimodal_prompts_for_all_images(image_dataset=train_images, image_folder=image_folder)
        mapped_img_prompt_vllm_response = c_g.get_mapped_img_prompt_vllm_responses(multi_model_fmt_imgs_prompts, img_path_prompt_list)
        img_idx = 0
        for image_path in mapped_img_prompt_vllm_response:
            print("img_idx", img_idx, flush=True)
            prompt_and_clip_chosen_caps = c_g.use_clip_to_choose_best_captions(image_path,mapped_img_prompt_vllm_response[image_path])
            
            for prompt, caption in prompt_and_clip_chosen_caps:                  
                train_dataset_caption_gens.append(
                {
                    "img_id": img_idx,
                    "img_path": image_path,
                    "c_id": caption_gen_idx,
                    "prompt": prompt,
                    "caption": caption, # clip selected
                }
            )
                caption_gen_idx += 1

            img_idx+=1

        
        assert img_idx == len(train_images.keys())
    
    
    '''
    else: 
        c_g = CaptionGenerator(
            mm_model,
            clip_model,
            clip_processor,
            clip_model_max_length,
            prompts,
            gen_num_captions,
            choose_num_captions=choose_num_captions,
            choose_cap_per_prompt=choose_cap_per_prompt,
        )
    
    # store map_img_prompt_gens = {'img_path': {}}
    caption_gen_idx = 0
    train_dataset_caption_gens = []
    for img_idx, img_file in train_images.items():  # 100 for now.
        print("img_idx", img_idx, flush=True)
        img_path = os.path.join(image_folder, img_file)
        prompt_caption_mapping = c_g.get_clip_filtered_n_captions_based_on_img_prompts(img_path)
        for prompt, caption in prompt_caption_mapping:
            train_dataset_caption_gens.append(
                {
                    "img_id": img_idx,
                    "img_path": img_path,
                    "c_id": caption_gen_idx,
                    "prompt": prompt,
                    "caption": caption,
                }
            )

            caption_gen_idx += 1
    '''
    return train_dataset_caption_gens


def per_image_caption_generation(
    dataset_type,
    mm_model,
    clip_model,
    clip_processor,
    clip_model_max_length,
    img_idx,
    img_path,
    image_folder,
    c_ablation,
    prompts,
    use_vllm=False, 
    vllm_model=None, 
):
    # TODO: use vllm here. 
    gen_num_captions = c_ablation["gen_num_captions"]
    choose_cap_per_prompt = c_ablation["choose_cap_per_prompt"]
    choose_num_captions = len(prompts)
    assert choose_num_captions == NUM_AUX_LABELS_PER_IMAGE

    c_g = CaptionGenerator(
        mm_model,
        clip_model,
        clip_processor,
        clip_model_max_length,
        prompts,
        gen_num_captions,
        choose_num_captions=choose_num_captions,
        choose_cap_per_prompt=choose_cap_per_prompt,
    )

    prompt_caption_mapping = c_g.get_clip_filtered_n_captions_based_on_img_prompts(img_path)

    per_img_train_dataset_caption_gens = []
    for prompt, caption in prompt_caption_mapping:
        per_img_train_dataset_caption_gens.append(
            {
                "img_id": img_idx,
                "img_path": img_path,
                "c_id": caption_gen_idx,
                "prompt": prompt,
                "caption": caption,
            }
        )

        caption_gen_idx += 1

    return per_img_train_dataset_caption_gens


def offline_caption_generation(
    dataset_type,
    mm_model,
    clip_model,
    clip_processor,
    clip_model_max_length,
    train_images,
    val_images,
    image_folder,
    caption_generator_ablations,
    prompts,
):
    # free space --> Gemma 3 is not supported. 
    #del mm_model.model 
    gc.collect()
    torch.cuda.empty_cache()  

    for c_ablation in caption_generator_ablations:
        print(c_ablation, flush=True)
        if c_ablation["id"] == 1:
            train_file = f"/local/zemel/nikita/all-multi-modals/online_learning/baseline/{dataset_type}/{mm_model.model_type}/baseline_train_1_c_abl.jsonl"
            val_file = f"/local/zemel/nikita/all-multi-modals/online_learning/baseline/{dataset_type}/{mm_model.model_type}/baseline_val_1_c_abl.jsonl"
        else:
            train_file = f"/local/zemel/nikita/all-multi-modals/online_learning/offline_train_caption_gens/{dataset_type}/{mm_model.model_type}/{len(prompts)}_setofprompts_{c_ablation['id']}_c_abl.jsonl"
            val_file = '$$'
        gen_num_captions = c_ablation["gen_num_captions"]
        choose_cap_per_prompt = c_ablation["choose_cap_per_prompt"]
        choose_num_captions = NUM_AUX_LABELS_PER_IMAGE
        vllm_model = None # if c_ablation not 1 use vllm_model. 

        if not os.path.exists(train_file) or not os.path.exists(val_file):
            #from vllm import LLM
            #vllm_model = LLM(model=mm_model.model_path, gpu_memory_utilization=0.85, max_model_len=4096, enforce_eager=True, max_num_seqs=20, seed=0) # use 2 devices. 

            c_g = CaptionGenerator(
                mm_model,
                clip_model,
                clip_processor,
                clip_model_max_length,
                prompts,
                gen_num_captions, # change it for a second. 
                choose_num_captions=choose_num_captions,
                choose_cap_per_prompt=choose_cap_per_prompt,
                vllm_model=None,
            )

        if not os.path.exists(train_file):
            os.makedirs(os.path.dirname(train_file), exist_ok=True)
            train_file = open(train_file, "w")
            caption_gen_idx = 0
            num_imgs = 0
            for img_idx, img_file in train_images.items():  # 250 for now.
                print("img_idx", img_idx, flush=True)
                img_path = os.path.join(image_folder, img_file)
                prompt_caption_mapping = c_g.get_clip_filtered_n_captions_based_on_img_prompts(
                    img_path
                )
                for prompt, caption in prompt_caption_mapping:
                    train_file.write(
                        json.dumps(
                            {
                                "img_id": img_idx,
                                "img_path": img_path,
                                "c_id": caption_gen_idx,
                                "prompt": prompt,
                                "caption": caption,
                            }
                        )
                        + "\n"
                    )

                    caption_gen_idx += 1
                
                num_imgs+=1
                if num_imgs == NUM_TRAIN_IMAGES:
                    break

            train_file.close()

        else:
            print(f"train gens exist for {c_ablation['id']}")

        if (
            c_ablation["id"] == 1
        ):  # do val too cause this is baseline for caption order metric.

            if not os.path.exists(val_file):
                os.makedirs(os.path.dirname(val_file), exist_ok=True)
                val_file = open(val_file, "w")
                caption_gen_idx = 0
                num_imgs = 0
                for img_idx, img_file in val_images.items(): 
                    print("img_idx", img_idx, flush=True)
                    img_path = os.path.join(image_folder, img_file)
                    prompt_caption_mapping = c_g.get_clip_filtered_n_captions_based_on_img_prompts(
                        img_path
                    )
                    for prompt, caption in prompt_caption_mapping:
                        val_file.write(
                            json.dumps(
                                {
                                    "img_id": img_idx,
                                    "img_path": img_path,
                                    "c_id": caption_gen_idx,
                                    "prompt": prompt,
                                    "caption": caption,
                                }
                            )
                            + "\n"
                        )

                        caption_gen_idx += 1
                    
                    num_imgs+=1 

                val_file.close()

            else:
                print(f"val gens exist for {c_ablation['id']}")
