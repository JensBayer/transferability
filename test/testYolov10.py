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


def evaluate(args):
    device = args.device

    model_name = args.model
    
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
        
    import sys
    from ultralytics import YOLOv10
    import torch.backends.cudnn as cudnn
    
    model = YOLOv10(model_name)
    imgsz = 640
    _ = model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())), verbose=False)
    
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
            output = model(imgs.to(device), verbose=False)
            mAP.update([{
                'scores': o.boxes.conf[o.boxes.cls == 0].to('cpu'),
                'boxes': o.boxes.xyxy[o.boxes.cls == 0].to('cpu'),
                'labels': o.boxes.cls.long()[o.boxes.cls == 0].to('cpu'),
            } for o in output],
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
    parser.add_argument('--output', type=str, default='test_output')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--model', type=str, default=None)
    args = parser.parse_args()
    evaluate(args)

    
