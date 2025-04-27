import os
import shutil
import random

# 원본 폴더와 대상 폴더 경로
src_folder = './abnormal'
dst_folder = './valid/Ungood'
n = 16  # 옮길 파일 개수

# 원본 폴더 내 파일 리스트 얻기
all_files = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]

# 랜덤하게 n개 선택
selected_files = random.sample(all_files, n)

# 선택된 파일 옮기기
for file_name in selected_files:
    src_path = os.path.join(src_folder, file_name)
    dst_path = os.path.join(dst_folder, file_name)
    shutil.move(src_path, dst_path)  # 복사하려면 move 대신 copy 사용
    print(f'{file_name} 옮김 → {dst_path}')