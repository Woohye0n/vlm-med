import os
import argparse
import random
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from dataset.medical_few import MedDataset
from CLIP.clip import create_model
from CLIP.tokenizer import tokenize
from CLIP.adapter import CLIP_Inplanted
from PIL import Image
from sklearn.metrics import roc_auc_score, precision_recall_curve, pairwise
from loss import FocalLoss, BinaryDiceLoss
from utils import augment, cos_sim, encode_text_with_prompt_ensemble
from prompt import REAL_NAME
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import umap
from open_clip import create_model_from_pretrained, get_tokenizer

import warnings
warnings.filterwarnings("ignore")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# CLASS_INDEX = {'Brain':3, 'Liver':2, 'Retina_RESC':1, 'Retina_OCT2017':-1, 'Chest':-2, 'Histopathology':-3, "Test" :-4}
CLASS_INDEX = {'Brain':-1, 'Liver':-1, 'Retina_RESC':-1, 'Retina_OCT2017':-1, 'Chest':-2, 'Histopathology':-3, "Test" :-4}

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def main():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--model_name', type=str, default='ViT-L-14-336', help="ViT-B-16-plus-240, ViT-L-14-336")
    parser.add_argument('--pretrain', type=str, default='openai', help="laion400m, openai")
    parser.add_argument('--obj', type=str, default='Retina_OCT2017')
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--save_model', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='./ckpt/few-shot/')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument("--epoch", type=int, default=50, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--shot', type=int, default=4)
    parser.add_argument('--iterate', type=int, default=0)
    parser.add_argument('--umap', type=int, default=0)
    args = parser.parse_args()

    setup_seed(args.seed)

    classes = ['Retina_RESC', 'Liver', 'Retina_OCT2017', 'Brain', 'Chest', 'Histopathology', "Test"]

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    clips = []
    dets = []
    clips_label = []
    dets_label = []
    
    for idx, obj in enumerate(classes):
        # fixed feature extractor
        clip_model, process = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        clip_model.to(device)
        clip_model.eval()

        # load test dataset
        kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
        test_dataset = MedDataset(args.data_path, obj, args.img_size, args.shot, args.iterate)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)


        # few-shot image augmentation
        augment_abnorm_img, augment_abnorm_mask = augment(test_dataset.fewshot_abnorm_img, test_dataset.fewshot_abnorm_mask)
        augment_normal_img, augment_normal_mask = augment(test_dataset.fewshot_norm_img)

        augment_fewshot_img = torch.cat([augment_abnorm_img, augment_normal_img], dim=0)
        augment_fewshot_mask = torch.cat([augment_abnorm_mask, augment_normal_mask], dim=0)
        
        augment_fewshot_label = torch.cat([torch.Tensor([1] * len(augment_abnorm_img)), torch.Tensor([0] * len(augment_normal_img))], dim=0)

        train_dataset = torch.utils.data.TensorDataset(augment_fewshot_img, augment_fewshot_mask, augment_fewshot_label)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, **kwargs)


        # memory bank construction
        support_dataset = torch.utils.data.TensorDataset(augment_normal_img)
        support_loader = torch.utils.data.DataLoader(support_dataset, batch_size=1, shuffle=True, **kwargs)
        
        clips_data, clips_labels = test(args, obj, clip_model, test_loader)

        clips += clips_data
        clips_label += clips_labels

    data = np.stack(clips)

    if args.umap:
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=50, min_dist=0.8)
        embedding = reducer.fit_transform(data)  # shape: [50, 2]
    else:
        tsne = TSNE(n_components=2, random_state=42, perplexity=500, n_iter=3000)
        embedding = tsne.fit_transform(data)

    plt.figure(figsize=(8, 6))
    colors = ['blue', 'orange']
    for class_idx in [1, 0]:
        indices = np.array(clips_label) == class_idx
        plt.scatter(embedding[indices, 0],
                    embedding[indices, 1],
                    s=40,
                    c=colors[class_idx],
                    label=f'Class {class_idx}',
                    alpha=0.5)

    plt.title('umap Visualization with 2 Classes')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    algorithm = "umap" if args.umap else "tSNE"
    plt.savefig(f'all_{algorithm}_CLIP_2D.png', dpi=300)
    plt.clf()



def test(args, obj, model, test_loader):
    gt_list = []
    gt_mask_list = []

    det_image_scores_zero = []
    det_image_scores_few = []
    
    seg_score_map_zero = []
    seg_score_map_few= []

    clips = []
    clips_normal = []
    clips_anomaly = []

    det = []
    det_normal = []
    det_anomaly = []

    for (image, y, mask) in tqdm(test_loader):
        image = image.to(device)
        mask[mask > 0.5], mask[mask <= 0.5] = 1, 0

        with torch.no_grad(), torch.cuda.amp.autocast():
            # _, seg_patch_tokens, det_patch_tokens = model(image)

            # image = image.to(torch.float16)

            clip_feature = model.encode_image(image, normalize=True)
            clip_feature = clip_feature.squeeze(0).cpu().numpy()
            
            if y == 0:
                clips_normal.append(clip_feature)
            elif y == 1:
                clips_anomaly.append(clip_feature)
    
    ##########################################################
    # (2) [N, D] 형태로 쌓기
    clips = clips_normal + clips_anomaly
    clips_labels = [0] * len(clips_normal) + [1] * len(clips_anomaly)
    clips_data = np.stack(clips)  # shape: [50, 1024]

    if args.umap:
        # umap
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=50, min_dist=0.8)
        embedding = reducer.fit_transform(clips_data)  # shape: [50, 2]

    else:
        # t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=500, n_iter=3000)
        embedding = tsne.fit_transform(clips_data)


    # (4) 시각화
    plt.figure(figsize=(8, 6))
    colors = ['blue', 'orange']
    for class_idx in [1, 0]:
        indices = np.array(clips_labels) == class_idx
        plt.scatter(embedding[indices, 0],
                    embedding[indices, 1],
                    s=40,
                    c=colors[class_idx],
                    label=f'Class {class_idx}',
                    alpha=0.5)

    plt.title('umap Visualization with 2 Classes')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    algorithm = "umap" if args.umap else "tSNE"
    plt.savefig(f'{obj}_{algorithm}_CLIP_2D.png', dpi=300)
    plt.clf()
    ##########################################################
    # exit()

    return clips, clips_labels



if __name__ == '__main__':
    main()


