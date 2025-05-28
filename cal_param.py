import argparse
import os
import os.path as osp

import numpy as np
from torchsummary import summary

from models.AV_Base import AV_Base
from models.AV_Base_v2 import AV_Base_v2
from models.AV_Base_v3 import AV_Base_v3


# 计算参数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


parser = argparse.ArgumentParser()
args = parser.parse_args()

load_key = (
    "/mnt/public/gxj_2/EmoNet_2B/saved/seed1_MER24_AV/AV_Base/2025-04-15-12-07-24"
)
load_args_path = osp.join(load_key, "best_args.npy")
load_args = np.load(load_args_path, allow_pickle=True).item()["args"]
load_args_dic = vars(load_args)
args_dic = vars(args)
for key in args_dic:
    load_args_dic[key] = args_dic[key]
args = argparse.Namespace(**load_args_dic)
model = AV_Base(args)
total_params = count_parameters(model) / 1e6
print(f"Total trainable parameters: {total_params} M")
