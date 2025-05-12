CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 ovis/train/train.py --output_dir ./temp/

# CUDA_VISIBLE_DEVICES=0 python ovis/train/train.py --output_dir ./temp/