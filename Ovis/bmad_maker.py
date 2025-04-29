import os
import json
import random

# 폴더의 루트 경로 설정
root_dir = '/workspace/vlm-med/Ovis/data/'  # 여기에 루트 폴더 경로를 넣어

# 저장할 리스트
data_list = []

dataset_type = {
    "BraTS2021_slice": "Brain MRI",
    "camelyon16_256": "Pathology",
    "Chest-RSNA": "Chest X-Ray",
    "hist_DIY": "Liver CT",
    "OCT2017": "Retinal OCT",
    "RESC": "Retinal OCT"
}

# 루트부터 모든 파일 순회
for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        # 이미지 파일인 경우만 (원하면 확장자 필터링 가능)
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
            if "train" not in dirpath:
                continue
            # 파일 전체 경로
            file_path = os.path.join(dirpath, filename)
            # 파일 경로를 루트 기준 상대 경로로 변환
            # rel_path = os.path.relpath(file_path, root_dir)
            rel_path = file_path

            # id는 확장자 없는 파일명
            file_id = os.path.splitext(filename)[0]

            t = dataset_type[dirpath.split("/")[-3]]

            if random.random() < 0.5:
                # Short
                if random.random() < 0.5:
                    conversation = [
                        {
                            "from": "human",
                            "value": f"This is an image of {t}. <image>\nIs there any anomalies in the image? Answer in a single word or phrase."
                        }
                    ]
                else:
                    conversation = [
                        {
                            "from": "human",
                            "value": f"This is an image of {t}. <image>\nCan you see anomaly feature in the image? Answer in a single word or phrase."
                        }
                    ]

                if "ungood" in dirpath:
                    conversation.append(
                        {
                            "from": "gpt",
                            "value": "Yes"
                        }
                    )
                else:
                    conversation.append(
                        {
                            "from": "gpt",
                            "value": "No"
                        }
                    )
            else:
                # Long
                if random.random() < 0.5:
                    conversation = [
                        {
                            "from": "human",
                            "value": f"This is an image of {t}. <image>\nIs there any anomalies in the image?"
                        }
                    ]
                else:
                    conversation = [
                        {
                            "from": "human",
                            "value": f"This is an image of {t}. <image>\nCan you see anomaly feature in the image?"
                        }
                    ]
                if "ungood" in dirpath:
                    conversation.append(
                        {
                            "from": "gpt",
                            "value": "Yes. The image is an anomaly."
                        }
                    )
                else:
                    conversation.append(
                        {
                            "from": "gpt",
                            "value": "No. The image is normal."
                        }
                    )


            # 하나의 항목 생성
            entry = {
                "id": file_id,
                "image": rel_path.replace("\\", "/"),  # Windows에서도 경로를 /로
                "conversations": conversation
            }

            data_list.append(entry)

            conversation = [
                {
                    "from": "human",
                    "value": "<image>\nWhat type of medical image is this?"
                },
                {
                    "from": "gpt",
                    "value": f"{t}"
                }
            ]

            if random.random() < 0.5:
                entry = {
                    "id": file_id,
                    "image": rel_path.replace("\\", "/"),  # Windows에서도 경로를 /로
                    "conversations": conversation
                }

                data_list.append(entry)

# 결과를 json 파일로 저장
output_path = 'bmad_vqa.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(data_list, f, indent=4, ensure_ascii=False)

print(f"Saved {len(data_list)} entries to {output_path}")
