from pathlib import Path
import time
import argparse

import torch
import torchvision
import torchvision.transforms as T

from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm, trange

import json
import albumentations as A
import cv2

import matplotlib.pyplot as plt

from adv_utils.applier import PatchApplier

def get_run_id(output_path='outputs'):
    output_path = Path(output_path)
    id = len(list(output_path.glob('*')))
    return id

    
def gen_patch(args):
    device = args.device
    NPATCHES = args.npatches
    EPOCHS = args.epochs
    PATCH_INIT = args.patch_init
    PATCH_SHAPE = args.patch_shape
    LR = args.lr
    SCHEDULER_STEP_SIZE = args.scheduler_step_size
    RESIZE_RANGE = args.resize_range
    FILL_VALUE = args.fill_value
    SMT_WEIGHT = args.smt_weight
    VAL_WEIGHT = args.val_weight
    OUTPUT_PATH = args.output_path
    MODEL = args.model
    
    RUN_ID = get_run_id(OUTPUT_PATH)
    
    transforms = A.Compose([
        A.LongestMaxSize(640),
        A.PadIfNeeded(640,640, border_mode=cv2.BORDER_CONSTANT)
    ], bbox_params=A.BboxParams(format='coco', min_area=1024, min_visibility=0.1))
    
    def do_transform(img, anno):
        img, xywh = transforms(
            image=np.array(
                img.getdata())
            .reshape(img.height, img.width, 3)
            .astype(np.uint8), 
            bboxes=[[
                max(0.0, a['bbox'][0]),
                max(0.0, a['bbox'][1]),
                min(a['bbox'][2], img.width) - a['bbox'][0],
                min(a['bbox'][3], img.height) - a['bbox'][1],
                a['category_id']
            ] for a in anno]
        ).values()
        img = torch.tensor(img).permute(2,0,1)/255
        return img, np.array([a[:-1] for a in xywh]).reshape(-1,4).astype(np.int64)
        
    
    ds = torchvision.datasets.CocoDetection(
        '/data/INRIAPerson/Train/pos', 
        '/data/INRIAPerson/inriaperson.json', 
        transforms=do_transform)
    
    
    def collate(data):
        images = torch.stack([img for img, _ in data])
        xyxy = [xyxy for _, xyxy in data]
        return images, xyxy
    
    dl = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True, pin_memory=True, num_workers=16, collate_fn=collate)

    import sys
    from ultralytics import NAS
    import torch.backends.cudnn as cudnn
    
    model = NAS(MODEL)
    imgsz = 640
    _ = model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())), verbose=False)
    
    
    applier = PatchApplier(
        None, 
        patch_transforms=T.Compose([
            T.ColorJitter(0.1, 0.05, 0.03, 0.05),
            T.RandomRotation(30, fill=FILL_VALUE),
            T.RandomPerspective(fill=FILL_VALUE),
        ]),
        mode=PatchApplier.MODE_RANDOM_BBOX,
        resize_range=RESIZE_RANGE
    )
    
    class Meter:
        def __init__(self):
            self.sum = 0
            self.n = 0
            
        def __call__(self, value):
            self.sum += value
            self.n += 1
            return self.get()
        
        def get(self):
            return self.sum / self.n
        
        def reset(self):
            self.sum = 0
            self.n = 0
    
    def smoothness(patch):
        return (patch[:,:,1:] - patch[:,:,:-1]).abs().mean() + (patch[:,1:] - patch[:,:-1]).abs().mean()
    
    def objectness(outs):
        return outs[0][0][1].flatten(1).max(1)[0].mean()
        
    def validity(patch):
        return torch.nn.functional.mse_loss(patch, patch.clamp(0.1,0.9))
    
    cudnn.benchmark = True
    def train(epoch=0, save_patches=True):
        outs = []
        def hookfn(modul, inf, outf):
            outs.append(outf)
    
        hook = model.model.heads.register_forward_hook(hookfn)
    
        meters = {
            key : Meter() for key in ['obj', 'smt', 'val']
        }
        
        for i, (imgs, labels) in enumerate(dl):
            optimizer.zero_grad()
            outs = []
            imgs = torch.stack([
                applier(
                    img,
                    torchvision.ops.box_convert(
                        torch.from_numpy(label),
                        in_fmt='xywh',
                        out_fmt='xyxy'),
                    patch=patch,
                    normalized_annotations=False)
                for img, label in zip(imgs, labels)
            ])
            
            imgs = imgs.clamp(0,1)
            with torch.cuda.amp.autocast():
                model(imgs.to(device), verbose=False)
                obj = objectness(outs)
                smt = smoothness(patch)
                val = validity(patch)
                loss = obj + smt * SMT_WEIGHT + val * VAL_WEIGHT
            loss.backward()
           
            optimizer.step()
            if (len(dl) * epoch + i) % 250 == 0:
                scheduler.step()
    
        if save_patches:
            T.ToPILImage()(imgs[0].detach().cpu()).save(f'./example.png')
            T.ToPILImage()(patch.detach().cpu()).save(f'./patch.png')
    
        hook.remove()
        return {'obj': meters['obj'](obj.item()), 'smt': meters['smt'](smt.item()), 'val': meters['val'](val.item())}
    
    path = Path(OUTPUT_PATH) / f"{RUN_ID:03d}"
    path.mkdir(parents=True, exist_ok=True)
    
    for n in range(NPATCHES):
        if PATCH_INIT == 'rand':
            patch = torch.rand(PATCH_SHAPE)
        else:
            patch = torch.full(PATCH_SHAPE, PATCH_INIT)
        
        patch = patch.requires_grad_()
    
        optimizer = torch.optim.AdamW([patch], LR)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, SCHEDULER_STEP_SIZE)
        applier.patch = patch

        tloader = tqdm(range(0, EPOCHS))
        for epoch in tloader:
            #patch = patch.clamp(0,1)
            train_loss_dict = train(epoch)
            tloader.set_postfix(train_loss_dict)
            T.ToPILImage()(patch.clamp(0,1).detach().cpu()).save(path / f'patch_{n:03d}.png')
            torch.save(patch.detach().cpu(), path / f'patch_{n:03d}.pt')

        params = {
            'run_id': RUN_ID,
            'epochs': EPOCHS,
            'patch/shape': str(PATCH_SHAPE),
            'patch/initial': PATCH_INIT,
            'batch_size': dl.batch_size,
            'optimizer/type': type(optimizer).__name__,
            'optimizer/lr': optimizer.defaults['lr'],
            'scheduler/type': type(scheduler).__name__,
            'scheduler/step_size': scheduler.state_dict().get('step_size', -1),
            'scheduler/T_max': scheduler.state_dict().get('T_max', -1),
            'applier/resize_range/min': applier.resize_range[0],
            'applier/resize_range/max': applier.resize_range[0],
            'applier/mode': applier.mode,
            'applier/transforms': str(applier.patch_transforms),
            'model': MODEL,
            'n_patches': NPATCHES,
        }
        with (path / "params.json").open('w') as fd:
            json.dump(params, fd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--npatches', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patch_init', type=str, default='rand')
    parser.add_argument('--patch-shape', type=tuple, default=(3, 256, 256))
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--scheduler_step_size', type=int, default = 25)
    parser.add_argument('--resize_range', type=tuple, default=(0.75, 1.0))
    parser.add_argument('--fill_value', type=float, default=-1)
    parser.add_argument('--smt_weight', type=float, default=2)
    parser.add_argument('--val_weight', type=float, default=1)
    parser.add_argument('--output_path', type=str, default='outputs')
    args = parser.parse_args()
    gen_patch(args)

    
