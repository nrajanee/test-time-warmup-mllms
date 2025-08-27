# Get the accuracy

import json
import sys
from tqdm import tqdm
from accuracy_for_datasets.m4c_evaluator import EvalAIAnswerProcessor

# from rouge_score import rouge_scorer
from bert_score import score

answer_processor = EvalAIAnswerProcessor()
# scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)


def _compute_answer_scores(raw_answers):
    """
    compute the accuracy (soft score) of human answers
    """
    answers = [answer_processor(a) for a in raw_answers]
    assert len(answers) == 10
    gt_answers = list(enumerate(answers))
    unique_answers = set(answers)
    unique_answer_scores = {}

    for unique_answer in unique_answers:
        accs = []
        for gt_answer in gt_answers:
            other_answers = [item for item in gt_answers if item != gt_answer]
            matching_answers = [
                item for item in other_answers if item[1] == unique_answer
            ]
            acc = min(1, float(len(matching_answers)) / 3)
            accs.append(acc)
        unique_answer_scores[unique_answer] = sum(accs) / len(accs)

    return unique_answer_scores


def eval_pred_list(pred_list):
    pred_scores = []
    pred_scores_w_qid_pred_gold = {}
    print("using min(human gt soft scores) for eval")
    for entry in tqdm(pred_list):
        pred_answer = answer_processor(entry["pred_answer"])
        # print("pred_answer", pred_answer)
        question_id = entry["question_id"]
        # print("gt_answers", entry["gt_answers"])
        unique_answer_scores = _compute_answer_scores(
            entry["gt_answers"]
        )  # annotations.
        # print("unique_answer_scores", unique_answer_scores)

        pred_not_in_annotations = True
        gt_scores = []
        for unique_answer, score in unique_answer_scores.items():
            if unique_answer in pred_answer:
                gt_scores.append(score)
                pred_not_in_annotations = False

        if not pred_not_in_annotations:  # matched a gt answer.
            pred_scores.append(
                min(gt_scores)
            )  # doing minimum because exact match would take the lower score too. so this is closer to that.

        if pred_not_in_annotations:  # it's only 0 when it's umatched to any gt.
            pred_scores.append(0)
            assert len(gt_scores) == 0

        pred_scores_w_qid_pred_gold[question_id] = (
            pred_scores[-1],
            pred_answer,
            entry["gt_answers"],
        )  # last one added is for this question.

    accuracy = sum(pred_scores) / len(pred_scores)
    return accuracy, pred_scores_w_qid_pred_gold


def eval_get_accuracy(annotation_file, results):  # results is a dict
    annotation_file_open = open(annotation_file).read()
    if "TextVQA" in annotation_file: 
            annotations = json.loads(annotation_file_open)["data"]
    else: 
        annotations = json.loads(annotation_file_open)["annotations"]
    annotations_dict = {}
    for annotation in annotations:
        if "TextVQA" in annotation_file:
            q_id = annotation["image_id"]
        else: 
            q_id = annotation["question_id"]
        annotations_dict[q_id] = annotation
    annotations = annotations_dict
    pred_list = []
    for question_id in results.keys():
        if "TextVQA" in annotation_file: 
            annotation = annotations[question_id]
        else: 
            annotation = annotations[int(question_id)]
        
        if "TextVQA" in annotation_file: 
            annotation = annotations[question_id]
            gt_answers = [
                    answer for answer in annotation["answers"]
                ]
        else: 
            gt_answers = [
                    answer_obj["answer"] for answer_obj in annotation["answers"]
                ]
        pred_list.append(
            {
                "question_id": question_id,
                "pred_answer": results[question_id],
                "gt_answers": gt_answers,
            }
        )

    accuracy, pred_scores_w_qid_pred_gold = eval_pred_list(pred_list)
    print("Samples: {}\n Accuracy: {:.2f}%\n".format(len(pred_list), 100.0 * accuracy))
    return accuracy, pred_scores_w_qid_pred_gold
