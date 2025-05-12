import cv2
from PIL import Image
import numpy as np

# 1. 이미지 불러오기
base_img = Image.open('./data_sample/Chest_AD/test/Ungood/img/abnormal3.png').convert('RGB')  # 원본 이미지
base_img = Image.open('./data_sample/Liver_AD/test/Ungood/img/abnormal4.png').convert('RGB')  # 원본 이미지
base_img = Image.open('./data_sample/Brain_AD/test/Ungood/img/abnormal1.png').convert('RGB')  # 원본 이미지
base_img = Image.open('./data_sample/Brain_AD/test/Ungood/img/abnormal2.png').convert('RGB')  # 원본 이미지
heatmap_img = Image.open('temp.png').convert('L')  # 0~255 grayscale heatmap

# 2. heatmap 크기를 base 이미지에 맞게 리사이즈
heatmap_img_resized = heatmap_img.resize(base_img.size, resample=Image.BICUBIC)

# 3. numpy로 변환
base_np = np.array(base_img)
heatmap_np = np.array(heatmap_img_resized)

# 4. heatmap 컬러맵 적용 (OpenCV: COLORMAP_JET 등)
heatmap_color = cv2.applyColorMap(heatmap_np, cv2.COLORMAP_JET)  # BGR
heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)   # RGB

# 5. 알파 블렌딩 (overlay)
alpha = 0.5
overlay_np = (alpha * heatmap_color + (1 - alpha) * base_np).astype(np.uint8)

# 6. 저장 및 시각화
overlay_img = Image.fromarray(overlay_np)
overlay_img.save('overlay_result.png')
overlay_img.show()
