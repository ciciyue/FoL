# -*- coding: UTF-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
import torchvision.transforms as T
import argparse
import network_FoL
import util
import cv2
import numpy as np
from PIL import Image
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def input_transform():
    return T.Compose([
        T.Resize((322, 322)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

similarity_threshold = 0.99975
OUTPUT_DIR = "visualize"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "foL_visualize.jpg")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Visualize image matching guided by Discriminative Region Guidance.")
    image_path0 = "../image/match_pair/query.jpg"
    image_path1 = "../image/match_pair/database.jpg"
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--image_path0", type=str, default=image_path0, help="Path to the first image")
    parser.add_argument("--image_path1", type=str, default=image_path1, help="Path to the second image")
    # parser.add_argument("--resume", type=str, default="../weights/FoL_large.pth", help="Path to the trained model checkpoint")
    parser.add_argument("--resume", type=str, default="/media/data1/chenshunpeng1/project_v2/open_source/FoL_opensorce/weights/FoL_large.pth", help="Path to the trained model checkpoint")
    return parser.parse_args()

def get_keypoints(img_size):
    H,W = img_size
    patch_size = 4
    N_h = H//patch_size
    N_w = W//patch_size
    keypoints = np.zeros((2, N_h*N_w), dtype=int)
    keypoints[0] = np.tile(np.linspace(patch_size//2, W-patch_size//2, N_w, dtype=int), N_h)
    keypoints[1] = np.repeat(np.linspace(patch_size//2, H-patch_size//2, N_h, dtype=int), N_w)
    return np.transpose(keypoints)

def match_batch_tensor(fm1, fm2, img_path0, img_path1, mask1, mask2, img_size):
    l = fm1.shape[0]
    N = fm2.shape[0]

    mask1_flat = mask1.reshape(l, -1)
    mask2_flat = mask2.reshape(N, l, -1)

    # Match only inside discriminative regions (mask-based filtering)
    mask1_indices = mask1_flat.bool().any(dim=1)
    mask2_indices = mask2_flat.bool().any(dim=2)

    fm1_masked = fm1[mask1_indices]
    fm2_masked = torch.stack([fm2[i, mask2_flat[i].bool().any(dim=1)] for i in range(N)])

    # Patch similarity matrix
    M = torch.matmul(fm2_masked, fm1_masked.T)

    max1 = torch.argmax(M, dim=2)
    max2 = torch.argmax(M, dim=1)

    # Mutual nearest neighbor constraint
    m = max2[torch.arange(N).reshape((-1, 1)), max1]
    device = fm1.device
    valid = (torch.arange(M.shape[1], device=device).repeat((N, 1)) == m)

    # Similarity threshold filtering
    valid_similarities = M[torch.arange(M.shape[0]).reshape((-1, 1)), torch.arange(M.shape[1]), max1]
    valid = valid & (valid_similarities > similarity_threshold)

    for i in range(N):
        idx2 = torch.nonzero(valid[i, :]).squeeze()
        idx1 = max1[i, :][idx2]

        original_idx1 = mask1_indices.nonzero(as_tuple=False).view(-1).cpu().numpy()[idx1.cpu().numpy()]
        original_idx2 = mask2_indices[i].nonzero(as_tuple=False).view(-1).cpu().numpy()[idx2.cpu().numpy()]
        assert original_idx1.shape == original_idx2.shape

        cv_im_one = cv2.resize(cv2.imread(img_path0), img_size)
        cv_im_two = cv2.resize(cv2.imread(img_path1), img_size)

        kps = get_keypoints(img_size)
        inlier_keypoints_one = kps[original_idx1]
        inlier_keypoints_two = kps[original_idx2]

        kp_all1 = []
        kp_all2 = []
        matches_all = []
        print("Number of matched point pairs:", len(inlier_keypoints_one))

        for k in range(inlier_keypoints_one.shape[0]):
            kp_all1.append(cv2.KeyPoint(inlier_keypoints_one[k, 0].astype(float), inlier_keypoints_one[k, 1].astype(float), 1, -1, 0, 0, -1))
            kp_all2.append(cv2.KeyPoint(inlier_keypoints_two[k, 0].astype(float), inlier_keypoints_two[k, 1].astype(float), 1, -1, 0, 0, -1))
            matches_all.append(cv2.DMatch(k, k, 0))

        im_allpatch_matches = cv2.drawMatches(cv_im_one, kp_all1, cv_im_two, kp_all2, matches_all, None, matchColor=(0, 255, 0), flags=2)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        cv2.imwrite(OUTPUT_PATH, im_allpatch_matches)


def match_batch_tensor_ransac(fm1, fm2, img_path0, img_path1, mask1, mask2, img_size):
    l = fm1.shape[0]
    N = fm2.shape[0]

    mask1_flat = mask1.reshape(l, -1)
    mask2_flat = mask2.reshape(N, l, -1)

    # Match only inside discriminative regions (mask-based filtering)
    mask1_indices = mask1_flat.bool().any(dim=1)
    mask2_indices = mask2_flat.bool().any(dim=2)

    fm1_masked = fm1[mask1_indices]
    fm2_masked = torch.stack([fm2[i, mask2_flat[i].bool().any(dim=1)] for i in range(N)])

    # Patch similarity matrix
    M = torch.matmul(fm2_masked, fm1_masked.T)

    max1 = torch.argmax(M, dim=2)
    max2 = torch.argmax(M, dim=1)

    # Mutual nearest neighbor constraint
    m = max2[torch.arange(N).reshape((-1, 1)), max1]
    device = fm1.device if hasattr(fm1, 'device') else 'cuda'
    valid = (torch.arange(M.shape[1], device=device).repeat((N, 1)) == m)

    # Similarity threshold filtering
    valid_similarities = M[torch.arange(M.shape[0]).reshape((-1, 1)), torch.arange(M.shape[1]), max1]
    valid = valid & (valid_similarities > similarity_threshold)

    for i in range(N):
        idx2 = torch.nonzero(valid[i, :]).squeeze()
        idx1 = max1[i, :][idx2]

        original_idx1 = mask1_indices.nonzero(as_tuple=False).view(-1).cpu().numpy()[idx1.cpu().numpy()]
        original_idx2 = mask2_indices[i].nonzero(as_tuple=False).view(-1).cpu().numpy()[idx2.cpu().numpy()]
        assert original_idx1.shape == original_idx2.shape

        cv_im_one = cv2.resize(cv2.imread(img_path0), img_size)
        cv_im_two = cv2.resize(cv2.imread(img_path1), img_size)

        kps = get_keypoints(img_size)
        inlier_keypoints_one = kps[original_idx1]
        inlier_keypoints_two = kps[original_idx2]

        print("Number of matched point pairs (before RANSAC):", len(inlier_keypoints_one))

        # --- 1. Draw and save the matching visualization before RANSAC ---
        kp_all1_before = [cv2.KeyPoint(float(pt[0]), float(pt[1]), 1, -1, 0, 0, -1) for pt in inlier_keypoints_one]
        kp_all2_before = [cv2.KeyPoint(float(pt[0]), float(pt[1]), 1, -1, 0, 0, -1) for pt in inlier_keypoints_two]
        matches_before = [cv2.DMatch(k, k, 0) for k in range(len(inlier_keypoints_one))]

        im_matches_before = cv2.drawMatches(cv_im_one, kp_all1_before, cv_im_two, kp_all2_before, matches_before, None, matchColor=(0, 255, 0), flags=2)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "foL_visualize.jpg"), im_matches_before)

        # --- RANSAC Geometric Verification ---
        # Requires at least 8 points to compute the Fundamental Matrix
        if len(inlier_keypoints_one) >= 8:
            src_pts = inlier_keypoints_one.astype(np.float32)
            dst_pts = inlier_keypoints_two.astype(np.float32)

            # Apply USAC_MAGSAC for robust outlier rejection (8.0px threshold)
            F, mask_ransac = cv2.findFundamentalMat(src_pts, dst_pts, cv2.USAC_MAGSAC, 8.0, 0.99)

            if mask_ransac is not None:
                mask_ransac = mask_ransac.ravel().astype(bool)
                inlier_keypoints_one = inlier_keypoints_one[mask_ransac]
                inlier_keypoints_two = inlier_keypoints_two[mask_ransac]

        print("Number of matched point pairs (after RANSAC):", len(inlier_keypoints_one))

        # --- 2. Draw and save the matching visualization after RANSAC ---
        kp_all1_after = [cv2.KeyPoint(float(pt[0]), float(pt[1]), 1, -1, 0, 0, -1) for pt in inlier_keypoints_one]
        kp_all2_after = [cv2.KeyPoint(float(pt[0]), float(pt[1]), 1, -1, 0, 0, -1) for pt in inlier_keypoints_two]
        matches_after = [cv2.DMatch(k, k, 0) for k in range(len(inlier_keypoints_one))]

        im_matches_after = cv2.drawMatches(cv_im_one, kp_all1_after, cv_im_two, kp_all2_after, matches_after, None, matchColor=(0, 255, 0), flags=2)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "foL_visualize_ransac.jpg"), im_matches_after)


def display_and_match_features(model, img_path0, img_path1, transform):
    img0 = Image.open(img_path0)
    img1 = Image.open(img_path1)
    img0_t = transform(img0).unsqueeze(0).to('cuda')
    img1_t = transform(img1).unsqueeze(0).to('cuda')

    with torch.no_grad():
        out0 = model(img0_t)
        local_feature0, mask1 = out0[4][0], out0[4][1]

        out1 = model(img1_t)
        local_feature1, mask2 = out1[4][0], out1[4][1]

    feature0 = local_feature0.view(1, -1, local_feature0.shape[3])
    feature1 = local_feature1.view(1, -1, local_feature0.shape[3])

    print("Size of patch tokens:", feature0.shape[1:])

    # Legacy: Standard feature matching without geometric constraints
    # match_batch_tensor(feature0[0], feature1, img_path0, img_path1, mask1, mask2, img_size=(356, 356))

    # Robust matching: Integrates RANSAC for outlier rejection
    match_batch_tensor_ransac(feature0[0], feature1, img_path0, img_path1, mask1, mask2, img_size=(356, 356))

def main():
    args = parse_arguments()
    model = network_FoL.FoLNet()
    model = util.resume_model(args, model)
    model.eval()
    model = model.to(args.device)
    transform = input_transform()
    display_and_match_features(model, args.image_path0, args.image_path1, transform)

if __name__ == '__main__':
    main()