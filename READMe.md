# Test-Time warmup for MLLMS 
Authors: Nikita Rajaneesh, Thomas P. Zollo, Richard Zemel

Paper: [Arxiv link]


## Setup
- Please pip install conda_requirements.txt
- Using utils.py you can generate json files for a set of n images (in the paper we do 500) and questions you want to apply test-time warmup to. 
- Each step in test-time-warmup saves data in a folder. You can find more details on that in each script. 

## Auxiliary task data generation. 
 The following script generates the captions for each image. 
 /../test-time-warmup-mllms/auxiliary_data/offline_caption_generation.py 

Example run: 

## Baseline evaluation. 
The following script provides the baseline accuracies for datasets (on 500 images). 
/../nikita/test-time-warmup-mllms/run_baseline_inference.py

Example run: 

## Test-time warmup (gradient steps) 
The following script runs test-time-warmup on 500 images. 
/../test-time-warmup-mllms/run_and_eval_domain_adapter.py

Our main method tta_adapt is in adapt_infer.py. 

## Exploratory code. 
There's more information about this in the limitations and future work section of the paper. 

We ran experiments for unsupervised domain adaptation; instead of throwing away the model weights after each image we kept updating the weights of the model. You'll find that in adapt_infer.py. 

We also tried generation captions in an online fashion: after each update to model re-generate captions per image for the next epoch. You'll find that in online_adapt_infer.py

Additionaly, we also tried doing RL using GRPO loss. 