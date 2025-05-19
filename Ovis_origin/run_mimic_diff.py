import argparse
import torch
import os
import json
from tqdm import tqdm

from PIL import Image, ImageDraw
import math
from transformers import set_seed, logging
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from ovis.serve.runner import RunnerArguments, OvisRunner

def eval_model(args):
    runner_args = RunnerArguments(model_path='AIDC-AI/Ovis2-8B')
    # runner_args = RunnerArguments(model_path='/workspace/vlm-med/Ovis/model_0518')
    runner = OvisRunner(runner_args)

    with open("./mimic_diff_vqa_test_500.json", "r") as f:
        json_data = json.load(f)

    right = 0
    wrong = 0
    total_bleu = 0
    smoothie = SmoothingFunction().method4
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    total_rouge1 = 0
    total_rougeL = 0
    idx = 0
    for item in tqdm(json_data):
        image_paths = item["image"]
        if len(image_paths) < 2:
            continue
        image1 = Image.open(image_paths[0]).convert("RGB")
        image2 = Image.open(image_paths[1]).convert("RGB")
        qs = item["conversations"][0]["value"]
        qs += "\nAnswer in a single sentence."
        gt = item["conversations"][1]["value"]

        outputs = runner.run([image1, image2, qs])['output'].strip().strip(".")

        bleu = sentence_bleu([gt.lower().split()], outputs.lower().split(), smoothing_function=smoothie)
        total_bleu += bleu
        scores = rouge.score(gt.lower(), outputs.lower())
        total_rouge1 += scores['rouge1'].fmeasure
        total_rougeL += scores['rougeL'].fmeasure
        if bleu > 0.5:
            right += 1
        else:
            wrong += 1

    print(f"######################################")
    print(f"Average BLEU: {total_bleu / (right + wrong)}")
    print(f"Average ROUGE-1: {total_rouge1 / (right + wrong)}")
    print(f"Average ROUGE-L: {total_rougeL / (right + wrong)}")
    print(f"######################################")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/workspace/Slake1.0/imgs")
    parser.add_argument("--question-file", type=str, default="/workspace/Slake1.0/test.json")
    parser.add_argument("--bbox_detail", action="store_true")
    parser.add_argument("--bbox_box_only", action="store_true")
    parser.add_argument("--bbox_draw", action="store_true")
    parser.add_argument("--bbox_draw_and_mention", action="store_true")
    args = parser.parse_args()

    eval_model(args)
