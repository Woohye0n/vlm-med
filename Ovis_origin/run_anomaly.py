import argparse
import torch
import os
import json
from tqdm import tqdm

from PIL import Image, ImageDraw
import math
from transformers import set_seed, logging

from ovis.serve.runner import RunnerArguments, OvisRunner

def eval_model(image_folder):
    runner_args = RunnerArguments(model_path='AIDC-AI/Ovis2-8B')
    runner = OvisRunner(runner_args)

    image_list = []
    for label in ["defect", "good"]:
        folder_path = os.path.join(image_folder, label)
        for filename in os.listdir(folder_path):
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                image_list.append(os.path.join(folder_path, filename))

    right = 0
    wrong = 0
    # for item in json_data:
    for item in tqdm(image_list):
        image = Image.open(item)
        qs = "Is there any anomaly in the image?"
        if "defect" in item:
            gt = "yes"
        else:
            gt = "no"

        qs += "\nAnswer the question using a single word or phrase."

        # print("###########################################")
        # print(prompt)
        # print("###########################################")
        # exit()

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
    parser.add_argument("--image-folder", type=str, default="/workspace/med_anomaly/")
    parser.add_argument("--bbox_detail", action="store_true")
    parser.add_argument("--bbox_box_only", action="store_true")
    parser.add_argument("--bbox_draw", action="store_true")
    parser.add_argument("--bbox_draw_and_mention", action="store_true")
    args = parser.parse_args()

    for subfolder in ["brain_mri", "headct", "br35h"]:
        print(f"=== Evaluating {subfolder} ===")
        eval_model(os.path.join(args.image_folder, subfolder, "test"))
