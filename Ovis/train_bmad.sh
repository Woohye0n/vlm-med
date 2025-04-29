CUDA_VISIBLE_DEVICES=0,1,3,7 torchrun --nproc_per_node=4 ovis/train/train.py --output_dir ./temp/

# CUDA_VISIBLE_DEVICES=4 python ovis/train/train.py --output_dir ./temp/