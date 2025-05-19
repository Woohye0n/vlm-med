import argparse
import torch
import os
import json
from tqdm import tqdm

from PIL import Image, ImageDraw
import math
from transformers import set_seed, logging

from ovis.serve.runner import RunnerArguments, OvisRunner

def eval_model(args):
    runner_args = RunnerArguments(model_path='AIDC-AI/Ovis2-8B')
    # runner_args = RunnerArguments(model_path='/workspace/vlm-med/Ovis/model_0518')
    runner = OvisRunner(runner_args)

    with open(os.path.expanduser(args.question_file), 'r') as f:
        json_data = json.load(f)

    right = 0
    wrong = 0
    # for item in json_data:
    for item in tqdm(json_data):
        if item["answer_type"] != "CLOSED":
            continue
        if item["q_lang"] != "en":
            continue

        image_file = item["img_name"]
        qs = item["question"].strip()
        gt = item["answer"].strip()

        with open(os.path.join(args.image_folder, "xmlab" + str(item["img_id"]), "detection.json")) as det:
            detection = json.load(det)

        image = Image.open(os.path.join(args.image_folder, image_file))

        if args.bbox_detail:
            qs += "\nBounding box guidelines:"
            for d in detection:
                qs += "\n" + str(d)
                # print(d.values())
        if args.bbox_box_only:
            qs += "\nBounding box guidelines:"
            for d in detection:
                for box in d.values():
                    qs += "\n" + str(box)
        if args.bbox_draw:
            draw = ImageDraw.Draw(image)
            for d in detection:
                for box in d.values():
                    bbox = (box[0], box[1], box[0] + box[2], box[1] + box[3])
                    draw.rectangle(bbox, outline="green", width=2)
                    # image.save("/workspace/output.jpg")
        if args.bbox_draw_and_mention:
            qs += "\nAnomalies are highlighted with green boxes."
            draw = ImageDraw.Draw(image)
            for d in detection:
                for box in d.values():
                    bbox = (box[0], box[1], box[0] + box[2], box[1] + box[3])
                    draw.rectangle(bbox, outline="green", width=2)
                    # image.save("/workspace/output.jpg")

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
    parser.add_argument("--image-folder", type=str, default="/workspace/Slake1.0/imgs")
    parser.add_argument("--question-file", type=str, default="/workspace/Slake1.0/test.json")
    parser.add_argument("--bbox_detail", action="store_true")
    parser.add_argument("--bbox_box_only", action="store_true")
    parser.add_argument("--bbox_draw", action="store_true")
    parser.add_argument("--bbox_draw_and_mention", action="store_true")
    args = parser.parse_args()

    eval_model(args)
