import argparse
import torch
import os
import numpy as np
import json
import cv2
from tqdm import tqdm

from PIL import Image, ImageDraw
import math
from transformers import set_seed, logging
from collections import Counter

from ovis.serve.runner import RunnerArguments, OvisRunner

def eval_model(args):
    # runner_args = RunnerArguments(model_path='AIDC-AI/Ovis2-8B')
    runner_args = RunnerArguments(model_path='/workspace/vlm-med2/Ovis/temp/')
    runner = OvisRunner(runner_args)

    for numbering in [1, 2, 3, 4]:
        image = Image.open(f"./abnormal{numbering}.png")

        # white, background, 
        s = "ab_normal"
        # qs = f"{s} {s} {s} {s} {s} {s} {s} {s}"
        qs = "Is there any anomaly normal in the image?\nAnswer the question using a single a single word or phrase."

        # print("###########################################")
        # print(prompt)
        # print("###########################################")
        # exit()

        outputs = runner.run([image, qs])['output'].strip().strip(".")

        # anomaly: 62948
        # normal: 4622
        # abnormal: 34563
        # print(runner.model.abnormal_embed.tolist())
        # for idx, i in enumerate(runner.model.abnormal_embed):
        #     if i >= 0:
        #         target = runner.text_tokenizer.decode(i, skip_special_tokens=False).strip()
        #         # print(target)
        #         if target == "anomaly":
        #             temp = runner.model.get_wte().weight[62948]
        #             print(i, idx)
        #             break

        # print(runner.model.temp_feature.shape)
        # print(len(runner.model.max_sim_index))
        # top5 = [idx for idx, _ in Counter(runner.model.max_sim_index).most_common(20)]
        # decoded = runner.model.text_tokenizer.batch_decode(top5, skip_special_tokens=False)
        # print(decoded)

        # L2
        # dists = torch.norm(runner.model.temp_feature - runner.model.abnormal_embed, dim=1)

        # cossim
        x1_norm = torch.nn.functional.normalize(runner.model.get_wte()(torch.tensor(34563).to(device=runner.device)), dim=0).unsqueeze(0)         # (3584,)
        x2_norm = torch.nn.functional.normalize(runner.model.get_wte()(torch.tensor(4622).to(device=runner.device)), dim=0).unsqueeze(0)         # (3584,)
        Y_norm = torch.nn.functional.normalize(runner.model.temp_feature, dim=1)         # (256, 3584)
        dist1 = torch.nn.functional.cosine_similarity(x1_norm, Y_norm, dim=1).squeeze()
        dist2 = torch.nn.functional.cosine_similarity(x2_norm, Y_norm, dim=1).squeeze()

        # heatmap_tensor = (dist1.reshape(16, 16) - dist2.reshape(16, 16))
        # print(heatmap_tensor.min(), heatmap_tensor.max())
        heatmap_tensor = dist1.reshape(16, 16)
        # print(heatmap_tensor.min(), heatmap_tensor.max())
        # heatmap_tensor = (dists1.reshape(16, 16) - dists2.reshape(16, 16) + 2) / 4

        # 3. heatmap 텐서를 PIL 이미지로 변환 (normalize to 0~255)
        heatmap_np = (heatmap_tensor - heatmap_tensor.min()) / (heatmap_tensor.max() - heatmap_tensor.min())
        # heatmap_np = 1 - heatmap_np
        # NumPy 변환
        heatmap_np = heatmap_np.to(torch.float32).cpu().detach().numpy()
        heatmap_np = (heatmap_np * 255).astype(np.uint8)

        # 4. NumPy 상태에서 리사이즈 (cv2 사용)
        # image: PIL.Image → 크기 확인
        target_size = image.size  # (width, height)
        heatmap_np_resized = cv2.resize(heatmap_np, dsize=target_size, interpolation=cv2.INTER_CUBIC)
        # heatmap_np_resized[heatmap_np_resized < 0.8 * 255] = 0

        # 5. 컬러맵 적용 (cv2 → RGB)
        heatmap_color = cv2.applyColorMap(heatmap_np_resized, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        # 6. PIL 이미지로 변환
        heatmap_color_img = Image.fromarray(heatmap_color)

        # 7. overlay
        overlay = Image.blend(image.convert('RGB'), heatmap_color_img.convert('RGB'), alpha=0.5)
        overlay.save(f"{s}_temp{numbering}.png")

    exit()

    # print("Model:", outputs)
    # print("GT: ", gt)

    if gt.lower() in outputs.lower() or outputs.lower() in gt.lower():
        right += 1
    else:
        wrong += 1




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
