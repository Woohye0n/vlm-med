import json

# 파일 경로와 바꿀 문자열 정의
file_path = './bmad_zero_shot.json'
old_text = 'bmad/'
new_text = '/workspace/anomaly_dataset/bmad/'

# 1. 파일을 문자열로 읽기
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 2. 문자열 내 텍스트 치환
modified_content = content.replace(old_text, new_text)

# 3. 덮어쓰기
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(modified_content)