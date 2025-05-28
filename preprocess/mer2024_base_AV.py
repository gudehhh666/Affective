import glob
import os
import shutil
import sys

sys.path.append("/mnt/public/gxj/EmoNets/")
import argparse
import os.path as osp
import random

import numpy as np
import torch

import config

# from toolkit.utils.read_files import *
from toolkit.utils.read_files import func_read_key_from_csv


def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def read_data_into_lists(filename, feat_dir, emo_rule):
    sample_ids = []
    numbers = []
    emotions = []
    float_values = []

    with open(filename, "r") as file:
        for line in file:
            parts = line.strip().split()

            if osp.exists(osp.join(feat_dir, parts[0] + ".npy")):
                if parts[2] not in config.EMO_RULE[emo_rule]:
                    continue

                sample_ids.append(parts[0])
                numbers.append(int(parts[1]))
                emotions.append(parts[2])
                float_values.append(float(parts[3]))

    return sample_ids, numbers, emotions, float_values


def normalize_dataset_format(
    emo_rule,
    gt_path,
    video_root,
    feat_video_root,
    feat_audio_root,
    description,
):

    all_names, _, all_emos, all_vals = read_data_into_lists(
        gt_path, feat_video_root, emo_rule
    )

    # --------------- split train & test ---------------
    split_ratio = 0.2
    whole_num = len(all_names)

    # gain indices for cross-validation
    indices = np.arange(whole_num)
    random.shuffle(indices)

    # split indices into 1-fold
    each_folder_num = int(whole_num * split_ratio)
    test_idxs = indices[0:each_folder_num]
    train_idxs = indices[each_folder_num:]

    split_train_names = []
    split_train_emos = []
    split_train_vals = []
    split_train_video_feat = []
    split_train_audio_feat = []
    for idx in train_idxs:
        name = all_names[idx]
        train_name = osp.join(video_root, "video", name)
        split_train_names.append(train_name)
        split_train_emos.append(all_emos[idx])
        split_train_vals.append(all_vals[idx])
        split_train_video_feat.append(osp.join(feat_video_root, name + ".npy"))
        split_train_audio_feat.append(osp.join(feat_audio_root, name + ".npy"))

    video_feat_dim = np.load(split_train_video_feat[0]).shape[0]
    audio_feat_dim = np.load(split_train_audio_feat[0]).shape[0]

    split_test_names = []
    split_test_emos = []
    split_test_vals = []
    split_test_video_feat = []
    split_test_audio_feat = []
    for idx in test_idxs:
        name = all_names[idx]
        test_name = osp.join(video_root, name)
        split_test_names.append(test_name)
        split_test_emos.append(all_emos[idx])
        split_test_vals.append(all_vals[idx])
        split_test_video_feat.append(osp.join(feat_video_root, name + ".npy"))
        split_test_audio_feat.append(osp.join(feat_audio_root, name + ".npy"))

    if len(osp.basename(feat_video_root)) == 0:
        video_feat_description = osp.basename(osp.dirname(feat_video_root))
    else:
        video_feat_description = osp.basename(feat_video_root)
    if len(osp.basename(feat_audio_root)) == 0:
        audio_feat_description = osp.basename(osp.dirname(feat_audio_root))
    else:
        audio_feat_description = osp.basename(feat_audio_root)

    feat_dim = {
        "video_dim": video_feat_dim,
        "video_feat_description": video_feat_description,
        "audio_dim": audio_feat_dim,
        "audio_feat_description": audio_feat_description,
    }
    for key in ["video", "audio"]:
        print(
            "# {}: dim={}".format(
                feat_dim[f"{key}_feat_description"], feat_dim[f"{key}_dim"]
            )
        )
    train_info = {
        "names": split_train_names,
        "emos": split_train_emos,
        "vals": split_train_vals,
        "video_feat": split_train_video_feat,
        "audio_feat": split_train_audio_feat,
    }
    valid_info = {
        "names": split_test_names,
        "emos": split_test_emos,
        "vals": split_test_vals,
        "video_feat": split_test_video_feat,
        "audio_feat": split_test_audio_feat,
    }
    return {
        "description": description,
        "feat_dim": feat_dim,
        "train_info": train_info,
        "valid_info": valid_info,
    }


# run -d toolkit/preprocess/mer2024.py
if __name__ == "__main__":
    random_seed = 7
    set_seed(random_seed)
    print("random seed: {}".format(random_seed))

    # --- data config
    emo_rule = "MER"
    gt_path = "/mnt/public/share/data/Dataset/MER2024/MER2024-labeled.txt"
    video_root = "/mnt/public/share/data/Dataset/MER2024/video/"

    # --- feature config
    description = "mer24-train"
    feat_video_root = (
        "/mnt/public/share/data/Dataset/MER2024/features/clip-vit-large-patch14-UTT/"
    )
    feat_audio_root = (
        "/mnt/public/share/data/Dataset/MER2024/features/chinese-hubert-large-UTT/"
    )

    # --- save config
    save_root = "/mnt/public/gxj_2/EmoNet_2B/lst_train/"
    save_name = "seed{}_MER24_AV".format(random_seed)

    # --- run
    dataset_info = normalize_dataset_format(
        emo_rule,
        gt_path,
        video_root,
        feat_video_root,
        feat_audio_root,
        description,
    )

    description = dataset_info["description"]
    feat_dim = dataset_info["feat_dim"]
    train_info = dataset_info["train_info"]
    valid_info = dataset_info["valid_info"]

    train_info["vals"] = [-100] * len(train_info["vals"])
    valid_info["vals"] = [-100] * len(valid_info["vals"])

    save_path = os.path.join(save_root, save_name + ".npy")
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    np.save(
        save_path,
        {
            "description": description,
            "feat_dim": feat_dim,
            "train_info": train_info,
            "valid_info": valid_info,
        },
    )

    print(
        "split_train: {}\nsplit_valid: {}".format(
            len(train_info["names"]), len(valid_info["names"])
        )
    )
