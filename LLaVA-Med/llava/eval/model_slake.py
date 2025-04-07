import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images

from PIL import Image, ImageDraw
import math
from transformers import set_seed, logging

logging.set_verbosity_error()


# def split_list(lst, n):
#     """Split a list into n (roughly) equal-sized chunks"""
#     chunk_size = math.ceil(len(lst) / n)  # integer division
#     return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


# def get_chunk(lst, n, k):
#     chunks = split_list(lst, n)
#     return chunks[k]


def eval_model(args):
    set_seed(0)
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    with open(os.path.expanduser(args.question_file), 'r') as f:
        json_data = json.load(f)

    right = 0
    wrong = 0
    for item in json_data:
        if item["answer_type"] != "CLOSED":
            continue
        if item["q_lang"] != "en":
            continue

        image_file = item["img_name"]
        qs = item["question"].strip()
        gt = item["answer"].strip()

        with open(os.path.join(args.image_folder, "xmlab" + str(item["img_id"]), "detection.json")) as det:
            detection = json.load(det)

        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

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
        
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # print("###########################################")
        # print(prompt)
        # print("###########################################")
        # exit()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image_tensor = process_images([image], image_processor, model.config)[0]

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        outputs = outputs.strip(".")

        # print("Model: ", outputs)
        # print("Answer: ", gt)

        if gt == outputs:
            right += 1
        else:
            wrong += 1

    print(f"######################################")
    print(f"Accuracy: {right / (right + wrong)}")
    print(f"######################################")


    # questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    # questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    # answers_file = os.path.expanduser(args.answers_file)
    # os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    # ans_file = open(answers_file, "w")
    # for line in tqdm(questions):
    #     idx = line["question_id"]
    #     image_file = line["image"]
    #     qs = line["text"].replace(DEFAULT_IMAGE_TOKEN, '').strip()
    #     cur_prompt = qs
    #     if model.config.mm_use_im_start_end:
    #         qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    #     else:
    #         qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    #     conv = conv_templates[args.conv_mode].copy()
    #     conv.append_message(conv.roles[0], qs)
    #     conv.append_message(conv.roles[1], None)
    #     prompt = conv.get_prompt()

    #     input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    #     image = Image.open(os.path.join(args.image_folder, image_file))
    #     image_tensor = process_images([image], image_processor, model.config)[0]

    #     stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    #     keywords = [stop_str]
    #     stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    #     with torch.inference_mode():
    #         output_ids = model.generate(
    #             input_ids,
    #             images=image_tensor.unsqueeze(0).half().cuda(),
    #             do_sample=True if args.temperature > 0 else False,
    #             temperature=args.temperature,
    #             top_p=args.top_p,
    #             num_beams=args.num_beams,
    #             # no_repeat_ngram_size=3,
    #             max_new_tokens=1024,
    #             use_cache=True)

    #     outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    #     ans_id = shortuuid.uuid()
    #     ans_file.write(json.dumps({"question_id": idx,
    #                                "prompt": cur_prompt,
    #                                "text": outputs,
    #                                "answer_id": ans_id,
    #                                "model_id": model_name,
    #                                "metadata": {}}) + "\n")
    #     ans_file.flush()
    # ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Veda0718/llava-med-v1.5-mistral-7b-finetuned")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/workspace/Slake1.0/imgs")
    parser.add_argument("--question-file", type=str, default="/workspace/Slake1.0/test.json")
    # parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    # parser.add_argument("--num-chunks", type=int, default=1)
    # parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--bbox_detail", action="store_true")
    parser.add_argument("--bbox_box_only", action="store_true")
    parser.add_argument("--bbox_draw", action="store_true")
    parser.add_argument("--bbox_draw_and_mention", action="store_true")
    args = parser.parse_args()

    eval_model(args)
