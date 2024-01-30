import torch
import sx_quant as sx_quant
from sx_quant import QuantConv2d
from sx_quant import Linker
import os
from tqdm import tqdm
import my_deploy_model

#========dataset新增============
import torchvision.transforms as T
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets.imagenet import ImageFolder
#===============================

# import debugpy
# device = int(os.environ['LOCAL_RANK'])
# base_port = 50000
# debugpy.listen(base_port + device)
# debugpy.wait_for_client()
# print("debugpy connected")

# --------------------------------------------------------
# RepVGG: Making VGG-style ConvNets Great Again (https://openaccess.thecvf.com/content/CVPR2021/papers/Ding_RepVGG_Making_VGG-Style_ConvNets_Great_Again_CVPR_2021_paper.pdf)
# Github source: https://github.com/DingXiaoH/RepVGG
# Licensed under The MIT License [see LICENSE for details]
# The training script is based on the code of Swin Transformer (https://github.com/microsoft/Swin-Transformer)
# --------------------------------------------------------
import sys
sys.path.append('../')
import time
import argparse
import datetime
import numpy as np
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter
from train.config import get_config
from data import build_loader
from train.lr_scheduler import build_scheduler
from train.logger import create_logger
from utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor, save_latest, update_model_ema, unwrap_model, load_weights
import copy
from train.optimizer import build_optimizer
from repvggplus import create_RepVGGplus_by_name

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


#=======================coco数据集====================
import torch
# from models.yolo import Model
import argparse
from sx_quant import Linker
import yaml
# from utils.dataloaders import create_dataloader
from tqdm import tqdm
import time
import os
# import val as val
from copy import deepcopy
from pathlib import Path
import time

def parse_option():
    parser = argparse.ArgumentParser('RepOpt-VGG training script built on the codebase of Swin Transformer', add_help=False)
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--summary', type=str, default="ptq_summary1.json", help="summary save file")
    parser.add_argument('--arch', default=None, type=str, help='arch name')
    parser.add_argument('--batch-size', default=128, type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', default='/home/lr/workspace/data/imagenet/', type=str, help='path to dataset')
    parser.add_argument('--scales-path', default=None, type=str, help='path to the trained Hyper-Search model')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O0', choices=['O0', 'O1', 'O2'],  #TODO Note: use amp if you have it
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='./output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    parser.add_argument("--local_rank", type=int, default=0, help='local rank for DistributedDataParallel')


    # 量化需要新增参数
    parser.add_argument('--calib_batch_size',default=200, type=int, help="gradient accumulation steps")
    parser.add_argument('--save_ptq', action='store_true', help='use zipped dataset instead of folder dataset')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config

def collect_stats(model, data_loader, device, num_batch=200):
    model.eval()
    model.cuda()
    
    # 模型前向收集信息 test
    with torch.no_grad():
        for i, datas in tqdm(enumerate(data_loader), total=num_batch):
            imgs = datas[0].to(device, non_blocking=True).float()
            # datas是一个列表。第一个元素是(4, 3, 640, 640)的照片。 
            # 第二个元素是边框（39， 6） # # 图像索引、类别索引、x_center, y_center, width, height
            # 第三个元素是长度为4的列表真实图片路径
            # 第四个元素是通常是一个列表，包含了批次中每个图像的原始形状
            model(imgs)
            if i >= num_batch:
                break


def get_model(resume, deploy, use_checkpoint):
    from repvgg import create_RepVGG_A0
    model = create_RepVGG_A0(deploy=deploy, use_checkpoint=use_checkpoint)
    load_weights(model, resume)
    return model

def prepare_dataset_dataloader(args):
    transform = T.Compose([
        T.Resize(224 + 32, interpolation=T.InterpolationMode.BILINEAR),
        T.RandomCrop(224),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    trainset = ImageFolder(os.path.join(args.data_path, "train"), transform)
    trainloader = DataLoader(trainset, args.batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
    valset = ImageFolder(os.path.join(args.data_path, "val"), transform)
    valloader = DataLoader(valset, args.batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)



    return trainset, valset, trainloader, valloader

def evaluate(model, dataloader):
    correct = 0
    total   = 0
    dtype = next(model.parameters()).dtype
    bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Evaluate")
    for ibatch, (image, target) in bar:
        B = image.size(0)

        with torch.no_grad():
            predict = model(image.cuda().to(dtype)).view(B, -1).argmax(1)
        
        correct += (predict == target.cuda()).sum()
        total += B
        accuracy = correct / total
        bar.set_description(f"Evaluate accuracy is: {accuracy:.6f}")
    
    accuracy = correct / total
    print(f"Top1 accuracy is: {accuracy:.6f}")
    return accuracy

def pipeline(args, config):
    print("1.0 create model......")
    model = my_deploy_model.deploy_model().eval()
    sx_quant.replace_modules(model)
    print(model)

    print("2.0 Prepare Dataset ....")
    trainset, valset, trainloader, valloader = prepare_dataset_dataloader(args)
    print(
        "=======================================================\n" +
        f"Run at: \n" + 
        f" model                = {args.arch}\n" + 
        f" batch                = {args.batch_size}\n" + 
        f" len_train_dataloader = {len(trainloader)}\n" + 
        f" len_val_dataloader   = {len(valloader)}\n" + 
        "======================================================="
    )

    print("3.0 Begining Calibration ....")
    Linker(model).do_collect = True
    start1 = time.time()
    collect_stats(model, trainloader, "cuda", args.calib_batch_size)
    print(f"collect cost {time.time() - start1} s")
    Linker(model).do_collect = False

    print("4.0 Begining post_compute ....")
    start2 = time.time()
    Linker(model).post_compute()
    print(f"compute amax cost {time.time() - start2} s")
    print(f"collect + compute amax cost {time.time() - start1}")

    print("5.0 quant infer ....")

    Linker(model).do_quant = True
    # 开启量化后，关闭yolov5最后一个detect层。
    # Linker(model.model[-1].m).do_quant = False
    ap = evaluate(model, valloader)
    # summary.append(["sx_quant", ap])
    Linker(model).do_quant = False

    print("6.0 ori infer ....")
    ap = evaluate(model, valloader)

    if args.save_ptq:
        print("Export PTQ...")
        Linker(model).do_export = True
        # rules.run_export(model, args.ptq)
        device = next(model.parameters()).device
        model.float()
        input_dummy = torch.zeros(1,3,640,640, device=device)
        from pytorch_quantization import nn as quant_nn
        model.eval()
        with torch.no_grad():
            torch.onnx.export(model, input_dummy, ptq, opset_version=14, )
        Linker(model).do_export = False

if __name__ == "__main__":
    args, config = parse_option()
    print(config.dump())
    pipeline(args, config)