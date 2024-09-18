from pathlib import Path
import time
import argparse

import torch
import torchvision
import torchvision.transforms as T
from torchmetrics.detection import MeanAveragePrecision

from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm, trange

import json
import albumentations as A
import cv2

import matplotlib.pyplot as plt

from adv_utils.applier import PatchApplier
from adv_utils.dataset import PatchedDataset

import sys
sys.path.append('./RT-DETR/rtdetr_pytorch')

from src.core.yaml_config import YAMLConfig
from src.core.yaml_utils import create
from src.solver import TASKS

def evaluate(args):
    PARAMS_FILE = args.params_file
    device = args.device
    
    if args.model:
        model_name = args.model
    else:
        params_file = Path(PARAMS_FILE)
        with params_file.open() as fd:
            params = json.load(fd)

    transforms = A.Compose([
            A.LongestMaxSize(640),
            A.PadIfNeeded(640,640, border_mode=cv2.BORDER_CONSTANT)
        ], bbox_params=A.BboxParams(format='coco', min_area=0, min_visibility=0.1))
    
    def do_transform(img, anno):
        img, xywh = transforms(
            image=np.array(
                img.getdata())
            .reshape(img.height, img.width, 3)
            .astype(np.uint8), 
            bboxes=[[
                max(0.0, a['bbox'][0]),
                max(0.0, a['bbox'][1]),
                min(a['bbox'][2], img.width),
                min(a['bbox'][3], img.height),
                a['category_id']
            ] for a in anno if a['category_id'] == 1]
        ).values()
        img = torch.tensor(img).permute(2,0,1)/255
        return img, np.array([a[:-1] for a in xywh]).reshape(-1,4).astype(np.int64)
        
    
    ds = torchvision.datasets.CocoDetection(
        '/data/coco2017/val2017',
        '/data/coco2017/annotations/instances_val2017.json',
        transforms=do_transform)
    
    
    
    class BboxConvertDataset(torch.utils.data.Dataset):
        def __init__(self, source_ds, in_fmt='xywh'):
            self.ds = source_ds
    
        def __len__(self):
            return len(self.ds)
    
        def __getitem__(self, idx):
            img, xywh = self.ds[idx]
            xyxy = torchvision.ops.box_convert(torch.from_numpy(xywh), in_fmt='xywh', out_fmt='xyxy')
            return img, xyxy

    import torch.backends.cudnn as cudnn
    
    cfg = YAMLConfig(args.config)
    solver = TASKS['detection'](cfg)
    solver.setup()
    solver.load_state_dict(torch.load(model_name))
    model = solver.ema.module.eval()
    imgsz = 640
    
    def evaluate_patch(patch=None):
        applier = PatchApplier(
            patch, 
            patch_transforms=None,
            mode=PatchApplier.MODE_CENTER_BBOX,
            resize_range=(0.75,0.75)
        )
        
        ds_patched = PatchedDataset(BboxConvertDataset(ds),applier)
        def collatefn(data):
            imgs = [d[0] for d in data]
            labels = [d[1] for d in data]
            return torch.stack(imgs), labels
        
        dl = torch.utils.data.DataLoader(ds_patched, batch_size=32, num_workers=8, collate_fn=collatefn)

        mAP = MeanAveragePrecision()
        for imgs, labels in dl:
            pred = model(imgs.to(device))
            valid = torch.logical_and((pred['pred_logits'].max(2)[0].sigmoid() > 0.5), pred['pred_logits'].max(2)[1] == 0)
            mAP.update([{
                'scores': o[valid[i]].max(1)[0].sigmoid().to('cpu'),
                'boxes': b[valid[i]].to('cpu'),
                'labels': o[valid[i]].max(1)[1].to('cpu'),
            } for i, (o, b) in enumerate(zip(pred['pred_logits'], (torchvision.ops.box_convert(pred['pred_boxes'], 'cxcywh', 'xyxy') * 640).long()))],
            [{
                'boxes': l,
                'labels': torch.zeros(len(l), dtype=torch.long),
            } for l in labels])
        return mAP.compute()
    
    patches = sorted(list(Path(args.patches).rglob('patch_???.png')))
    for f in tqdm(patches):
        output_file = (Path(args.output) / Path(model_name).stem / f.parent.stem / f.stem).with_suffix('.json')
        output_file.parent.mkdir(parents=True, exist_ok=True)
        if output_file.exists():
            continue
        patch = torchvision.io.read_image(str(f))/255
        results = evaluate_patch(patch)
        with output_file.open('w') as fd:
            json.dump({k: v.item() for k, v in results.items()}, fd)

    output_file = (Path(args.output) / Path(model_name).stem / 'none.json')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if not output_file.exists():
        results = evaluate_patch(None)
        with output_file.open('w') as fd:
            json.dump({k: v.item() for k, v in results.items()}, fd)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('patches', type=str)
    parser.add_argument('--config', type=str)
    parser.add_argument('--params_file', type=str)
    parser.add_argument('--output', type=str, default='test_output')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--model', type=str, default=None)
    args = parser.parse_args()
    evaluate(args)

    
