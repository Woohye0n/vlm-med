echo "python llava/eval/model_slake.py --model-path /workspace/vlm-med/MedTrinity-25M/llava-llama-med-8b-stage2-finetune-vqa_rad_orift"
python llava/eval/model_slake.py --model-path /workspace/vlm-med/MedTrinity-25M/llava-llama-med-8b-stage2-finetune-vqa_rad_orift

echo "python llava/eval/model_slake.py --model-path /workspace/vlm-med/MedTrinity-25M/llava-llama-med-8b-stage2-finetune-vqa_rad_orift --bbox_box_only"
python llava/eval/model_slake.py --model-path /workspace/vlm-med/MedTrinity-25M/llava-llama-med-8b-stage2-finetune-vqa_rad_orift --bbox_box_only

echo "python llava/eval/model_slake.py --model-path /workspace/vlm-med/MedTrinity-25M/llava-llama-med-8b-stage2-finetune-vqa_rad_orift --bbox_detail"
python llava/eval/model_slake.py --model-path /workspace/vlm-med/MedTrinity-25M/llava-llama-med-8b-stage2-finetune-vqa_rad_orift --bbox_detail

echo "python llava/eval/model_slake.py --model-path /workspace/vlm-med/MedTrinity-25M/llava-llama-med-8b-stage2-finetune-vqa_rad_orift --bbox_draw"
python llava/eval/model_slake.py --model-path /workspace/vlm-med/MedTrinity-25M/llava-llama-med-8b-stage2-finetune-vqa_rad_orift --bbox_draw

echo "python llava/eval/model_slake.py --model-path /workspace/vlm-med/MedTrinity-25M/llava-llama-med-8b-stage2-finetune-vqa_rad_orift --bbox_draw_and_mention"
python llava/eval/model_slake.py --model-path /workspace/vlm-med/MedTrinity-25M/llava-llama-med-8b-stage2-finetune-vqa_rad_orift --bbox_draw_and_mention
