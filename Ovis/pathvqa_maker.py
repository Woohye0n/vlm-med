import json
import os
from datasets import load_dataset
from tqdm import tqdm

# 수정해줘야 하는 경로
image_root_path = "/workspace/vlm-med/Ovis/data/PathVQA/images"  # 로컬 이미지 위치
os.makedirs(image_root_path, exist_ok=True)
output_json_path = "/workspace/vlm-med/Ovis/data/pathvqa_vit_format.json"

# 데이터셋 로드
dataset = load_dataset("flaviagiammarino/path-vqa", split="train")  # 필요한 split 선택 ('train', 'validation', 'test')

# 결과 저장할 리스트
output_data = []

for idx, item in tqdm(enumerate(dataset)):
    image_name = "pathvqa_" + str(idx).zfill(6) + ".png"
    question = item["question"].strip()
    answer = item["answer"].strip()
    question_id = "pathvqa_" + str(idx).zfill(6)
    image = item["image"]
    if image.mode == "CMYK":
        image = image.convert("RGB")
    image.save(os.path.join(image_root_path, image_name))

    # 데이터 변환
    entry = {
        "id": question_id,
        "image": os.path.join(image_root_path, image_name),
        "conversations": [
            {
                "from": "human",
                "value": f"<image>\n{question} Answer in a single word or phrase."
            },
            {
                "from": "gpt",
                "value": answer
            }
        ]
    }

    output_data.append(entry)

# JSON 저장
with open(output_json_path, "w") as f:
    json.dump(output_data, f, indent=2)

print(f"Saved {len(output_data)} examples to {output_json_path}")