import argparse
import torch
import os
import json
from tqdm import tqdm

from PIL import Image, ImageDraw
import math
from transformers import set_seed, logging
from datasets import load_dataset

from ovis.serve.runner import RunnerArguments, OvisRunner

def eval_model(args):
    # runner_args = RunnerArguments(model_path='/workspace/vlm-med/Ovis/model_0518')
    runner_args = RunnerArguments(model_path='AIDC-AI/Ovis2-8B')
    runner = OvisRunner(runner_args)

    # HuggingFace의 path-vqa 데이터셋 불러오기
    dataset = load_dataset("flaviagiammarino/path-vqa", split="test")

    right = 0
    wrong = 0
    for item in tqdm(dataset):
        qs = item["question"].strip()
        gt = item["answer"].strip()

        if len(gt.split(' ')) != 1:
            continue
        # if gt.lower() not in ['yes', 'no']:
        #     continue

        image = item["image"]  # 이미 PIL Image로 들어옴

        qs += "\nAnswer the question using a single word or phrase."

        outputs = runner.run([image, qs])['output'].strip().strip(".")

        # print("Model:", outputs)
        # print("GT: ", gt)

        if gt.lower() in outputs.lower() or outputs.lower() in gt.lower():
            right += 1
        else:
            wrong += 1

    print(f"######################################")
    print(f"Accuracy: {right / (right + wrong)}")
    print(f"######################################")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-base", type=str, default=None)
    args = parser.parse_args()

    eval_model(args)