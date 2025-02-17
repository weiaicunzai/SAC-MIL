import os
import random
import sys
import xml.etree.ElementTree as ET

import cv2
import h5py
import numpy as np
import openslide
import pandas as pd
import torch
import torch.utils.data as data
from pytorch_grad_cam import GradCAM

from models.mixer import MlpMixer


# main()
class WSIDataset(data.Dataset):
    def __init__(self, data_root, state, fold, dataset):
        super().__init__()
        # Set all input args as attributes
        self.data_root = data_root

        data_set = state

        self.feat_dir = os.path.join(
            self.data_root,
            # 'feat_clam_resnet50',
            "patch_size_256_at_mag_20",
            "pt_files",
        )

        self.idx = os.path.join(self.data_root, "idx_64")

        self.pos = os.path.join(self.data_root, "pos")

        # dataset_name = os.path.basename(self.data_root)
        dataset_name = dataset
        self.file_list_csv = os.path.join(
            "dataset_csv", "dataset_list", dataset_name, "{}.csv".format(dataset_name)
        )
        self.split_dir = os.path.join("dataset_csv", "splits", dataset_name)
        wsis = self.get_wsis(data_set, fold)
        self.wsis = wsis
        self.slide_names = list(self.wsis.keys())

        self.data_set = data_set

        self.tmp_data = {}

        # for idx in range(len(self)):
        #     slide_id = self.slide_names[idx]
        #     # label = self.wsis[slide_id]
        #     full_path = os.path.join(self.feat_dir, '{}.pt'.format(slide_id))
        #     self.get_data(full_path)

    def get_wsis(self, data_set, fold):
        file_list = pd.read_csv(self.file_list_csv)
        file_list["slide_id"] = file_list["slide_id"].apply(
            lambda x: os.path.splitext(x)[0]
        )

        split_file = os.path.join(self.split_dir, "splits_{}.csv".format(fold))
        splits = pd.read_csv(split_file)
        train_split = splits["train"].dropna()
        val_split = splits["val"].dropna()
        test_split = splits["test"].dropna()

        if data_set == "train":
            mask1 = file_list["slide_id"].isin(train_split)
            mask2 = file_list["slide_id"].isin(val_split)
            slide_ids = file_list["slide_id"][mask1 | mask2]
        else:
            mask1 = file_list["slide_id"].isin(test_split)
            slide_ids = file_list["slide_id"][mask1]

        # slide_ids = slide_ids[:3]
        wsis = {}
        for slide_id in slide_ids:
            mask = file_list["slide_id"] == slide_id
            label = int(file_list[mask].iloc[0]["label"])
            wsis[slide_id] = label

        return wsis

    def dict2tensor(self, features):
        assert isinstance(features, dict)
        return torch.tensor(list(features.values()))

    def get_data(self, full_path):
        features = self.tmp_data.get(full_path, None)
        # print('start loading...')

        if features is None:
            features = torch.load(full_path)
            if isinstance(features, dict):
                features = self.dict2tensor(features)
            self.tmp_data[full_path] = features

        # print('done loading.......')
        return features

    def __len__(self):
        return len(self.wsis.keys())

    def __getitem__(self, idx):
        slide_id = self.slide_names[idx]
        label = self.wsis[slide_id]
        full_path = os.path.join(self.feat_dir, "{}.pt".format(slide_id))
        # print(full_path)

        # features = torch.load(full_path)
        features = self.get_data(full_path)

        # idx = torch.load(os.path.join(self.idx, slide_id + '.pt')).long()
        pos = torch.load(os.path.join(self.pos, slide_id + ".pt"))

        pos_len = pos.shape[0]
        feat_len = features.shape[0]
        if pos_len > feat_len:
            # print('>>')
            pos = pos[:feat_len]
        else:
            # print('<<')
            # print('before')
            # print(pos.shape, pos[:feat_len - pos_len].shape, pos_len, feat_len)
            # pos = torch.cat([pos, pos[:feat_len - pos_len]], dim=0)

            repeat_times = feat_len // pos_len
            remainder = feat_len % pos_len

            # 重复 pos 并拼接
            pos = torch.cat([pos] * repeat_times + [pos[:remainder]], dim=0)
            # print(pos.shape)

        features = features.clone()
        # print(os.path.join(self.idx, slide_id + '.pt'))
        # print(full_path)
        # print(pos_len, feat_len)

        # features = features[idx]
        # pos = pos[idx]

        # print(full_path)
        # print(os.path.join(self.pos, slide_id + '.pt'))
        # print(pos.shape, features.shape)

        # print(pos.shape, features.shape)
        assert pos.shape[0] == features.shape[0]
        if self.data_set == "train":
            index = [x for x in range(features.shape[0])]
            random.shuffle(index)
            features = features[index]
            pos = pos[index]

        assert pos.shape[0] == features.shape[0]
        return features, pos, label, slide_id


def hex_to_bgr(hex_color):
    # 去掉十六进制颜色前缀‘#’
    hex_color = hex_color.lstrip("#")
    # 解析十六进制颜色值，并转换为整数
    rgb = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    # 调整颜色顺序从RGB到BGR
    bgr = (rgb[2], rgb[1], rgb[0])
    return bgr


def read_tif(img_path):
    wsi = openslide.OpenSlide(img_path)

    level = 0
    for lvl, dims in enumerate(wsi.level_dimensions):
        level = lvl
        if max(dims) < 8000:
            break

    # print(len(wsi.level_dimensions))
    dims = wsi.level_dimensions[level]
    downsample_factor = wsi.level_downsamples[level]

    img_org = wsi.read_region((0, 0), level, dims).convert("RGB")
    return np.array(img_org, dtype=np.uint8), downsample_factor


def draw_points(img_path, coords, heatmap: np.ndarray, xml_path):
    heatmap = (heatmap * 255).astype(np.uint8)
    colormap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    img_org, downsample_factor = read_tif(img_path)  # RGB
    img_org = img_org[:, :, ::-1]  # BGR

    img_org = draw_anno(img_org, xml_path, downsample_factor)

    img = img_org

    overlay = np.zeros(img.shape, dtype=np.uint8)
    num_instances = coords.shape[0]
    for idx in range(num_instances):

        coord = coords[idx]
        color = colormap[idx][0].tolist()
        # print(idx,heatmap[idx])
        coord = coord / downsample_factor
        coord = coord.tolist()
        coord = [int(c) for c in coord]
        coord2 = [int(c + 512 / downsample_factor) for c in coord]
        # print(coord, hex_to_bgr(color))
        # cv2.circle(img, coord, 3, hex_to_bgr(color), 3)
        cv2.rectangle(overlay, coord, coord2, color, -1)
    overlay = draw_anno(overlay, xml_path, downsample_factor)
    # 将overlay层与原图合并
    alpha = 0.8
    img = cv2.addWeighted(img, alpha, overlay, 1 - alpha, 0)

    return img, img_org, overlay


def parse_anno(xml_path):
    annos = []
    if os.path.exists(xml_path) == False:
        return annos
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # print(root)
    # print(xml_path)
    for anno in root.iter("Annotation"):
        res = []
        for coord in anno.iter("Coordinate"):
            res.append([float(coord.attrib["X"]), float(coord.attrib["Y"])])

        annos.append(res)
    return annos


def draw_anno(img, anno_path, downsample_factor):
    annos = parse_anno(anno_path)

    pts = []
    for region in annos:
        region = np.array(region)
        region = region.reshape(-1, 1, 2)
        region = region / downsample_factor
        pts.append(region.astype(np.int32))
    img = img.copy()
    # template = np.zeros(dims[::-1])
    img = cv2.polylines(img, pts, True, (255, 255, 255), thickness=1)
    return img


data_root = "/data/smb/syh/WSI_cls/"
dataset = "cam16"
feat = "res50"
k = 64
ckpt = "ckpt/cam16/0/Friday_06_December_2024_15h_53m_10s/epoch=130-val_acc=0.8730.ckpt"

if feat == "uni":
    feat_ext = "feat_clam_uni"
elif feat == "res50":
    feat_ext = "feat_clam_resnet50"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


out_dir = "grad-cam-out"
os.makedirs(out_dir, exist_ok=True)
dataset_root = os.path.join(data_root, dataset, feat_ext)
h5_dir = os.path.join(
    data_root, dataset, "feat_clam_resnet50", "patch_size_256_at_mag_20", "h5_files"
)
img_dir = os.path.join(data_root, dataset, "img")
xml_dir = os.path.join(data_root, dataset, "anno")


dataset = WSIDataset(dataset_root, "test", 0, "cam16")
checkpoint = torch.load(ckpt, map_location=torch.device("cpu"))
state_dict = checkpoint["state_dict"]

state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
# model = MlpMixer(input_dim=1024, hidden_dim=512, scale=32768)
model = MlpMixer(input_dim=1024, hidden_dim=512, scale=512)
model.load_state_dict(state_dict)
model.to(device)
cam = GradCAM(
    model=model,
    target_layers=[model.intra_region3.intra_fc[0]],
    reshape_transform=lambda x: x.permute(0, 2, 1).unsqueeze(3),
)  # b,n,d -> b,d,n,1


targets = None  # 默认使用模型预测的类别

for features, coords, label, slide_id in dataset:
    h5_path = os.path.join(h5_dir, slide_id + ".h5")
    img_path = os.path.join(img_dir, slide_id + ".tif")
    xml_path = os.path.join(xml_dir, slide_id + ".xml")
    # img_org, downsample_factor = read_tif(img_path) #RGB
    # img_org = img_org[:, :, ::-1] #BGR
    if os.path.exists(xml_path) == False:
        continue

    batch = torch.concat([features, coords], dim=1).unsqueeze(0)  # B,N,D
    batch = batch.permute(0, 2, 1).unsqueeze(3).to(device)  # B,D,N,1   <=> B,C,H,W
    # print(features.shape, coords.shape,batch.shape, label,slide_id)
    with h5py.File(h5_path, "r") as f:
        coords = torch.tensor(f["coords"])

    # showArrayOrTensor(img_org,'img')
    # showArrayOrTensor(coords,'coords')

    grayscale_cam = cam(batch, None).reshape(-1)
    print(slide_id, label, grayscale_cam.shape[0], grayscale_cam.sum())

    img, img_org, heat = draw_points(img_path, coords, grayscale_cam, xml_path)
    # 将热力图叠加到原图

    cv2.imwrite(f"{out_dir}/{slide_id}.jpg", img)
    cv2.imwrite(f"{out_dir}/{slide_id}_org.jpg", img_org)
    cv2.imwrite(f"{out_dir}/{slide_id}_heat.jpg", heat)
