# Get the accuracy, correctness
import json
from tqdm import tqdm
import sys
from accuracy_for_datasets.m4c_evaluator import EvalAIAnswerProcessor

def get_correctness_dict(annotations, results):

    correctness_w_question_id_dict = {}
    c_w_q_id_pred_ans_ann = {}
    num_pred_eval = 0
    for qid, annotation in tqdm(annotations.items()):
        gold = annotation["answer"].lower()
        if results.get(qid) is None:
            continue
        pred = results[qid].lower()
        num_pred_eval += 1

        if gold in pred:
            correctness_w_question_id_dict[qid] = 1
        else:
            # print("gold", gold)
            # print("\n")
            # print("pred", pred)
            correctness_w_question_id_dict[qid] = 0

        c_w_q_id_pred_ans_ann[qid] = (correctness_w_question_id_dict[qid], pred, gold)
    
    print("num_pred_eval", num_pred_eval)
    assert num_pred_eval == len(results.keys())
    return correctness_w_question_id_dict, c_w_q_id_pred_ans_ann


def get_gqa_accuracy(annotations_file, results):
    with open(annotations_file) as file:
        annotations = json.load(file)

    correctness_w_question_id_dict, c_w_qid_pred_gold = get_correctness_dict(
        annotations, results
    )
    total = 0.0
    correct = 0.0
    for q_id, match in correctness_w_question_id_dict.items():
        if match == 1:
            correct += 1

        total += 1

    return correct / total, c_w_qid_pred_gold