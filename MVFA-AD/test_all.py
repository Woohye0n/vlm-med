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

from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score

def dunn_index(X, labels):
    """
    X: numpy array of shape [N, D] - feature vectors
    labels: numpy array of shape [N] - cluster labels

    returns: Dunn Index (float)
    """
    X = np.array(X)
    print(X.shape)
    labels = np.array(labels)
    print(labels.shape)

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    # 1. Intra-cluster distances (diameters)
    diameters = []
    for label in unique_labels:
        cluster_points = X[labels == label]
        if len(cluster_points) <= 1:
            diameters.append(0.0)
            continue
        distances = cdist(cluster_points, cluster_points, metric='euclidean')
        diameters.append(np.max(distances))
    max_diameter = max(diameters)

    # 2. Inter-cluster distances
    min_intercluster_dist = np.inf
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            points_i = X[labels == unique_labels[i]]
            points_j = X[labels == unique_labels[j]]
            distances = cdist(points_i, points_j, metric='euclidean')
            min_dist = np.min(distances)
            if min_dist < min_intercluster_dist:
                min_intercluster_dist = min_dist
    if max_diameter <= 0:
        print('max_diameter', max_diameter)
    # 3. Dunn Index
    return min_intercluster_dist / max_diameter if max_diameter > 0 else 0.0

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
    parser.add_argument('--img_size', type=int, default=240)
    parser.add_argument("--epoch", type=int, default=50, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--shot', type=int, default=4)
    parser.add_argument('--iterate', type=int, default=0)
    parser.add_argument('--umap', type=int, default=0)
    args = parser.parse_args()

    setup_seed(args.seed)

    classes = ['Test','Retina_OCT2017', 'Chest', 'Retina_RESC', 'Liver', 'Brain', 'Histopathology']
    classes = ['Retina_OCT2017', 'Chest', 'Retina_RESC', 'Liver', 'Brain', 'Histopathology']

    clips = []
    dets = []
    clips_label = []
    dets_label = []
    
    for idx, obj in enumerate(classes):
        # fixed feature extractor
        clip_model = create_model(model_name=args.model_name, img_size=args.img_size, device=device, pretrained=args.pretrain, require_pretrained=True)
        clip_model.eval()

        model = CLIP_Inplanted(clip_model=clip_model, features=args.features_list).to(device)
        model.eval()

        checkpoint = torch.load(os.path.join(f'{args.save_path}', f'{obj}.pth'))
        model.seg_adapters.load_state_dict(checkpoint["seg_adapters"])
        model.det_adapters.load_state_dict(checkpoint["det_adapters"])

        for name, param in model.named_parameters():
            param.requires_grad = True

        # optimizer for only adapters
        seg_optimizer = torch.optim.Adam(list(model.seg_adapters.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))
        det_optimizer = torch.optim.Adam(list(model.det_adapters.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))



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


        # losses
        loss_focal = FocalLoss()
        loss_dice = BinaryDiceLoss()
        loss_bce = torch.nn.BCEWithLogitsLoss()


        # text prompt
        with torch.cuda.amp.autocast(), torch.no_grad():
            text_features = encode_text_with_prompt_ensemble(clip_model, REAL_NAME[obj], device)

        best_result = 0

        seg_features = []
        det_features = []
        for image in support_loader:
            image = image[0].to(device)
            with torch.no_grad():
                _, seg_patch_tokens, det_patch_tokens = model(image)
                seg_patch_tokens = [p[0].contiguous() for p in seg_patch_tokens]
                det_patch_tokens = [p[0].contiguous() for p in det_patch_tokens]
                seg_features.append(seg_patch_tokens)
                det_features.append(det_patch_tokens)
        seg_mem_features = [torch.cat([seg_features[j][i] for j in range(len(seg_features))], dim=0) for i in range(len(seg_features[0]))]
        det_mem_features = [torch.cat([det_features[j][i] for j in range(len(det_features))], dim=0) for i in range(len(det_features[0]))]
        
        result, clips_data, det_data, clips_labels, dets_labels = test(args, obj, model, test_loader, text_features, seg_mem_features, det_mem_features)

        clips += clips_data
        dets += det_data
        clips_label += clips_labels
        dets_label += dets_labels

    # data = np.stack(clips)

    # if args.umap:
    #     reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=50, min_dist=0.8)
    #     embedding = reducer.fit_transform(data)  # shape: [50, 2]
    # else:
    #     tsne = TSNE(n_components=2, random_state=42, perplexity=500, n_iter=3000)
    #     embedding = tsne.fit_transform(data)


    # plt.figure(figsize=(8, 6))
    # colors = ['blue', 'orange']
    # for class_idx in [1, 0]:
    #     indices = np.array(clips_label) == class_idx
    #     plt.scatter(embedding[indices, 0],
    #                 embedding[indices, 1],
    #                 s=40,
    #                 c=colors[class_idx],
    #                 label=f'Class {class_idx}',
    #                 alpha=0.5)

    # plt.title('umap Visualization with 2 Classes')
    # plt.xlabel('Component 1')
    # plt.ylabel('Component 2')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # algorithm = "umap" if args.umap else "tSNE"
    # plt.savefig(f'all_{algorithm}_CLIP_2D.png', dpi=300)
    # plt.clf()

    # data = np.stack(dets)
    # reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=50, min_dist=0.8)
    # embedding = reducer.fit_transform(data)  # shape: [50, 2]

    # plt.figure(figsize=(8, 6))
    # colors = ['blue', 'orange']
    # for class_idx in [1, 0]:
    #     indices = np.array(dets_label) == class_idx
    #     plt.scatter(embedding[indices, 0],
    #                 embedding[indices, 1],
    #                 s=40,
    #                 c=colors[class_idx],
    #                 label=f'Class {class_idx}',
    #                 alpha=0.5)

    # plt.title('umap Visualization with 2 Classes')
    # plt.xlabel('Component 1')
    # plt.ylabel('Component 2')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig('all_tSNE_det_2D.png', dpi=300)
    # plt.clf()



def test(args, obj, model, test_loader, text_features, seg_mem_features, det_mem_features):
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
            _, seg_patch_tokens, det_patch_tokens = model(image)

            clip_feature = model.clipmodel.encode_image(image, out_layers=[6, 12, 18, 24], normalize=True)
            clip_feature = clip_feature.squeeze(0)[1:, :].mean(dim=0).cpu().numpy()
            # clip_feature = clip_feature.squeeze(0).mean(dim=0).cpu().numpy()
            
            if y == 0:
                clips_normal.append(clip_feature)
            elif y == 1:
                clips_anomaly.append(clip_feature)

            temp_tensor = torch.cat(det_patch_tokens)
            # temp_tensor = det_patch_tokens[-1].squeeze(0)
            temp_tensor = temp_tensor.mean(dim=0)
            temp_tensor = temp_tensor[1:, :].mean(dim=0)
            # temp_tensor = temp_tensor[1:, :]
            if y == 0:
                det_normal.append(temp_tensor.cpu().numpy())
            elif y == 1:
                det_anomaly.append(temp_tensor.cpu().numpy())

            seg_patch_tokens = [p[0, 1:, :] for p in seg_patch_tokens]
            det_patch_tokens = [p[0, 1:, :] for p in det_patch_tokens]

            # few-shot, det head
            temp_list = []
            anomaly_maps_few_shot = []
            for idx, p in enumerate(det_patch_tokens):
                cos = cos_sim(det_mem_features[idx], p)
                height = int(np.sqrt(cos.shape[1]))
                temp_list.append(torch.min((1 - cos), 0)[0].unsqueeze(0))
                anomaly_map_few_shot = torch.min((1 - cos), 0)[0].reshape(1, 1, height, height)
                anomaly_map_few_shot = F.interpolate(torch.tensor(anomaly_map_few_shot),
                                                        size=args.img_size, mode='bilinear', align_corners=True)
                anomaly_maps_few_shot.append(anomaly_map_few_shot[0].cpu().numpy())

            # temp_list = torch.cat(temp_list, dim=0)
            # det_feature = temp_list.mean(dim=0).cpu().numpy()
            # if y == 0:
            #     det_normal.append(det_feature)
            # elif y == 1:
            #     det_anomaly.append(det_feature)
                
            anomaly_map_few_shot = np.sum(anomaly_maps_few_shot, axis=0)
            score_few_det = anomaly_map_few_shot.mean()
            det_image_scores_few.append(score_few_det)

            # zero-shot, det head
            anomaly_score = 0
            for layer in range(len(det_patch_tokens)):
                det_patch_tokens[layer] /= det_patch_tokens[layer].norm(dim=-1, keepdim=True)
                anomaly_map = (100.0 * det_patch_tokens[layer] @ text_features).unsqueeze(0)
                anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                anomaly_score += anomaly_map.mean()
            det_image_scores_zero.append(anomaly_score.cpu().numpy())

            
            gt_mask_list.append(mask.squeeze().cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())

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


    # (2) [N, D] 형태로 쌓기
    dets = det_normal + det_anomaly
    det_labels = [0] * len(det_normal) + [1] * len(det_anomaly)
    det_data = np.stack(dets)  # shape: [50, 1024]

    clips_center = np.mean(np.stack(clips), axis=0)
    clips_n_center = np.mean(np.stack(clips_normal), axis=0)
    clips_a_center = np.mean(np.stack(clips_anomaly), axis=0)

    det_center = np.mean(np.stack(dets), axis=0)
    det_n_center = np.mean(np.stack(det_normal), axis=0)
    det_a_center = np.mean(np.stack(det_anomaly), axis=0)


    # cossim_list = []
    # for p in clips_anomaly:
    #     p_n = clips_n_center - p
    #     p_d = clips_a_center - p
    #     norms = np.linalg.norm(p_n) * np.linalg.norm(p_d)
    #     cossim = np.dot(p_n, p_d) / norms
    #     cossim_list.append(cossim)

    # dunn = dunn_index(clips, clips_labels)
    score = silhouette_score(np.array(clips), np.array(clips_labels), metric='euclidean')
    print(obj, "clips silhouette:", score)

    # cossim_list = []
    # for p in det_anomaly:
    #     p_n = det_n_center - p
    #     p_d = det_a_center - p
    #     norms = np.linalg.norm(p_n) * np.linalg.norm(p_d)
    #     cossim = np.dot(p_n, p_d) / norms
    #     cossim_list.append(cossim)

    # dunn = dunn_index(dets, det_labels)
    score = silhouette_score(np.array(dets), np.array(det_labels), metric='euclidean')
    print(obj, "det silhouette:", score)


    if args.umap:
        # umap
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=50, min_dist=0.8)
        embedding = reducer.fit_transform(det_data)  # shape: [50, 2]
    else:
        # t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=500, n_iter=3000)
        embedding = tsne.fit_transform(det_data)


    # 3D 시각화
    plt.figure(figsize=(8, 6))
    colors = ['blue', 'orange']
    for class_idx in [1, 0]:
        indices = np.array(det_labels) == class_idx
        plt.scatter(
            embedding[indices, 0],
            embedding[indices, 1],
            c=colors[class_idx],
            label=f'Class {class_idx}',
            s=40,
            alpha=0.5
        )

    plt.title('umap Visualization with 2 Classes')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{obj}_{algorithm}_det_2D.png', dpi=300)
    plt.clf()
    ##########################################################
    # exit()

    gt_list = np.array(gt_list)
    gt_mask_list = np.asarray(gt_mask_list)
    gt_mask_list = (gt_mask_list>0).astype(np.int_)


    if CLASS_INDEX[obj] > 0:

        seg_score_map_zero = np.array(seg_score_map_zero)
        seg_score_map_few = np.array(seg_score_map_few)

        seg_score_map_zero = (seg_score_map_zero - seg_score_map_zero.min()) / (seg_score_map_zero.max() - seg_score_map_zero.min())
        seg_score_map_few = (seg_score_map_few - seg_score_map_few.min()) / (seg_score_map_few.max() - seg_score_map_few.min())
    
        segment_scores = 0.5 * seg_score_map_zero + 0.5 * seg_score_map_few
        seg_roc_auc = roc_auc_score(gt_mask_list.flatten(), segment_scores.flatten())
        print(f'{obj} pAUC : {round(seg_roc_auc,4)}')

        segment_scores_flatten = segment_scores.reshape(segment_scores.shape[0], -1)
        roc_auc_im = roc_auc_score(gt_list, np.max(segment_scores_flatten, axis=1))
        print(f'{obj} AUC : {round(roc_auc_im, 4)}')

        return seg_roc_auc + roc_auc_im

    else:

        det_image_scores_zero = np.array(det_image_scores_zero)
        det_image_scores_few = np.array(det_image_scores_few)

        det_image_scores_zero = (det_image_scores_zero - det_image_scores_zero.min()) / (det_image_scores_zero.max() - det_image_scores_zero.min())
        det_image_scores_few = (det_image_scores_few - det_image_scores_few.min()) / (det_image_scores_few.max() - det_image_scores_few.min())
    
        image_scores = 0.5 * det_image_scores_zero + 0.5 * det_image_scores_few
        img_roc_auc_det = roc_auc_score(gt_list, image_scores)
        print(f'{obj} AUC : {round(img_roc_auc_det,4)}')

        return img_roc_auc_det, clips, dets, clips_labels, det_labels


if __name__ == '__main__':
    main()


