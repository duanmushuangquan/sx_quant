import inspect
import torch
import os

import json
class SummaryTool:
    def __init__(self, file):
        self.file = file
        self.data = []
        
    def append(self, item):
        self.data.append(item)
        json.dump(self.data, open(self.file, "w"), indent = 4) # indent=4更容易阅读

class Logger:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def set_verbose(self, enable=True):
        self.verbose = enable

    def log(self, *msg):
        if self.verbose:
            stack  = inspect.stack()[1]
            lineno = stack.lineno
            filename = os.path.basename(stack.filename)
            formatted_msg = " ".join(str(m) for m in msg)
            print(f"[{filename}:{lineno}]:{formatted_msg}")

import torch
import torch.nn as nn
from torchvision.models import resnet
from torchvision.transforms import transforms
from PIL import Image
import math
from enum import Enum
from typing import Callable, List, Optional

class Linker(object):
    def __init__(self, model:nn.Module):
        self.model = model

    def quant_modules(self):
        for _, module in self.model.named_modules():
            if isinstance(module, Quantizer):
                yield module

    @property
    def do_quant(self):
        for module in self.quant_modules():
            return module.do_quant
        
    @do_quant.setter
    def do_quant(self, new_value):
        for module in self.quant_modules():
            module.do_quant = new_value

    @property
    def do_collect(self):
        for module in self.quant_modules():
            return module.do_collect
        
    @do_collect.setter
    def do_collect(self, new_value):
        for module in self.quant_modules():
            module.do_collect = new_value

    @property
    def do_export(self):
        for module in self.quant_modules():
            return module.do_export
        
    @do_export.setter
    def do_export(self, new_value):
        for module in self.quant_modules():
            module.do_export = new_value

    def post_compute(self):
        for module in self.quant_modules():
            module.post_compute()
    

def quant(x, scale):
    return torch.round(x / scale).clamp(-127, +127)

def dequant(x, scale):
    return x * scale

def fake_quant(x, scale):
    return torch.round(x / scale).clamp(-127, +127) * scale

class Method(Enum):
    PerTensor  = 0
    PerChannel = 1

import copy
import scipy.stats as stats
from collections import Counter
import numpy as np
from tqdm import tqdm
class CalibrationHistogram(object):
    def __init__(self, method):
        """
        Args:
            method (Method): only support PerTensor
        """
        assert method == Method.PerTensor, f"Unsupport per_channel."
        self.method       = method
        self.collect_data = None


    def _compute_amax_entropy(self, calib_hist, calib_bin_edges, num_bits=8, unsigned=True, stride=1, start_bin=128):
        """Returns amax that minimizes KL-Divergence of the collected histogram"""

        # If calibrator hasn't collected any data, return none
        if calib_bin_edges is None and calib_hist is None:
            return None

        def _normalize_distr(distr):
            summ = np.sum(distr)
            if summ != 0:
                distr = distr / summ

        bins = calib_hist[:]
        bins[0] = bins[1]

        total_data = np.sum(bins)

        divergences = []
        arguments = []

        # we are quantizing to 128 values + sign if num_bits=8
        nbins = 1 << (num_bits - 1 + int(unsigned))

        starting = start_bin
        stop = len(bins)

        new_density_counts = np.zeros(nbins, dtype=np.float64)

        for i in tqdm(range(starting, stop + 1, stride), total=stop-starting):
            new_density_counts.fill(0)
            space = np.linspace(0, i, num=nbins + 1)
            digitized_space = np.digitize(range(i), space) - 1

            digitized_space[bins[:i] == 0] = -1

            for idx, digitized in enumerate(digitized_space):
                if digitized != -1:
                    new_density_counts[digitized] += bins[idx]

            counter = Counter(digitized_space)
            for key, val in counter.items():
                if key != -1:
                    new_density_counts[key] = new_density_counts[key] / val

            new_density = np.zeros(i, dtype=np.float64)
            for idx, digitized in enumerate(digitized_space):
                if digitized != -1:
                    new_density[idx] = new_density_counts[digitized]

            total_counts_new = np.sum(new_density) + np.sum(bins[i:])
            _normalize_distr(new_density)

            reference_density = np.array(bins[:len(digitized_space)])
            reference_density[-1] += np.sum(bins[i:])

            total_counts_old = np.sum(reference_density)
            # if round(total_counts_new) != total_data or round(total_counts_old) != total_data:
            #     raise RuntimeError("Count mismatch! total_counts_new={}, total_counts_old={}, total_data={}".format(
            #         total_counts_new, total_counts_old, total_data))

            _normalize_distr(reference_density)

            ent = stats.entropy(reference_density, new_density)
            divergences.append(ent)
            arguments.append(i)

        divergences = np.array(divergences)
        # print("divergences={}".format(divergences))
        last_argmin = len(divergences) - 1 - np.argmin(divergences[::-1])
        calib_amax = calib_bin_edges[last_argmin * stride + starting]
        calib_amax = torch.tensor(calib_amax.item())  #pylint: disable=not-callable
        print(calib_amax)
        return calib_amax


    def post_compute_amax(self):
        if self.collect_data is None:
            print("=======self.collect_data is None=======")
            return None
        
        hist, bins_width, absmax = self.collect_data
        num_of_bins    = hist.numel()
        #=================mse_start================
        # centers        = torch.linspace(bins_width * 0.5, absmax - bins_width * 0.5, num_of_bins, device=hist.device, dtype=hist.dtype)
        # condidates_start = 128
        # condidates     = centers[condidates_start:]

        # centers        = centers.view(1, -1)
        # condidates     = condidates.view(-1, 1)
        
        # reproject      = fake_quant(centers, condidates / 127)
        # different      = (((centers - reproject) ** 2) * hist).sum(dim=1)
        # select_index  = torch.argmin(different)
        # optimal_center = condidates[select_index, 0]
        #================mse end====================

        #=============entropy start===============

        """Returns amax that minimizes KL-Divergence of the collected histogram"""
        calib_bin_edges = torch.linspace(0, absmax, num_of_bins+1)
        calib_hist = hist.to("cpu").numpy()
        calib_bin_edges = calib_bin_edges.to("cpu").numpy()
        optimal_center = self._compute_amax_entropy(calib_hist, calib_bin_edges)

        #================mse end====================
        # free 
        self.collect_data = None
        return optimal_center

    def collect(self, x):
        if self.collect_data is None:
            xabs        = x.float().abs()
            xabs_max    = xabs.max().item()
            num_bins    = 2048
            bins_width  = xabs_max / num_bins
            hist        = torch.histc(xabs, num_bins, min=0, max=xabs_max)
            self.collect_data = hist, bins_width, xabs_max
        else:
            prev_hist, bins_width, prev_xabs_max = self.collect_data
            current_xabs     = x.float().abs()
            current_xabs_max = current_xabs.max().item()
            current_absmax   = max(current_xabs_max, prev_xabs_max)
            current_num_bins = math.ceil(current_absmax / bins_width)
            hist             = torch.histc(current_xabs, current_num_bins, min=0, max=current_absmax)
            hist[:prev_hist.numel()] += prev_hist
            self.collect_data = hist, bins_width, current_absmax

class CalibrationABSMax(object):
    def __init__(self, method):
        """
        Args:
            method (Method): The method to be used for calibration, either PerTensor or PerChannel.
        """       
        self.method = method
        self.amax   = 0.0
    
    def collect(self, x):
        if self.method == Method.PerTensor:
            self.amax = max(x.float().abs().max().item(), self.amax)
        elif self.method == Method.PerChannel:
            self.amax = x.abs().view(x.size(0), -1).amax(1).view(-1, 1, 1, 1)

    def post_compute_amax(self):
        return self.amax

class Quantizer(nn.Module):
    def __init__(self, calibrator):
        """
        Args:
            calibrator (CalibrationHistogram or CalibrationABSMax)
        """   
        super().__init__()
        self.calibrator     = calibrator
        self.do_collect     = False
        self.do_quant       = False
        self.do_export      = False

    def extra_repr(self):
        repr = super().extra_repr()
        scale = "NoneScale"
        if hasattr(self, "_scale"):
            if self.calibrator.method == Method.PerChannel:
                scale = f"scale_min={self._scale.min().item():.5f}, scale_max={self._scale.max().item():.5f}"
            else:
                scale = f"scale={self._scale.item():.5f}"

        repr += f"{scale}, do_collect={self.do_collect}, do_quant={self.do_quant}, do_export={self.do_export}, method={self.calibrator.method}"
        return repr
    # 若直接 nn.Conv2d = QuantConv2d去替换，就不要写下面的。
    # 当完成了加载模型后，用replace_modules方法替换，再写下方代码
    #   二者区别在于，第一种先替换QuanConv2d，再创建模型，再加载权重。加载权重会调用quantizer的_load_from_state_dict。会报错，找不到_sacle
    #   而第二种方法，先加载模型，加载权重。再用QuanConv2d替换。不会调用quantizer的_load_from_state_dict。只有在加载已经有QDQ节点的模型，才会触发quantizer的_load_from_state_dict
    # def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
    #     model_device = next(self.parameters()).device  # 获取模型所在的设备
    #     self.register_buffer("_scale", state_dict[f"{prefix}_scale"].to(model_device))
    #     super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    
    def post_compute(self):
        amax = self.calibrator.post_compute_amax()
        self.register_buffer("_scale", torch.tensor(amax / 127).clamp(min=1e-7))

    def forward(self, x):
        assert sum([self.do_collect, self.do_quant, self.do_export]) <= 1, f"Invalid configuration: do_collect={self.do_collect}, do_quant={self.do_quant}, do_export={self.do_export}"

        if self.do_collect:
            self.calibrator.collect(x)
        elif self.do_quant:
            return fake_quant(x, self._scale)
        elif self.do_export:
            if self.calibrator.method == Method.PerTensor:
                return torch.fake_quantize_per_tensor_affine(x, self._scale.item(), 0, -128, +127)
            elif self.calibrator.method == Method.PerChannel:
                scale_sequeeze = self._scale.view(-1)
                zero_point     = torch.zeros(self._scale.size(0), dtype=torch.int32, device=self._scale.device)
                return torch.fake_quantize_per_channel_affine(x, scale_sequeeze, zero_point, 0, -128, +127)
        return x

class QTypeInputAndWeight:
    def init_quantizer(self):
        self.input_quantizer_  = Quantizer(CalibrationHistogram(Method.PerTensor))
        # self.input_quantizer_ = Quantizer(CalibrationABSMax(Method.PerTensor))
        self.weight_quantizer_ = Quantizer(CalibrationABSMax(Method.PerChannel))

class QTypeInputOnly:
    def init_quantizer(self):
        self.input_quantizer_  = Quantizer(CalibrationHistogram(Method.PerTensor))

class QuantConv2d(nn.Conv2d, QTypeInputAndWeight):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_quantizer()

    def forward(self, x):
        quant_x = self.input_quantizer_(x)
        quant_w = self.weight_quantizer_(self.weight)
        return super()._conv_forward(quant_x, quant_w, self.bias)

modules_map = [
    (torch.nn, "Conv2d", QuantConv2d),
]

def replace_modules(model: nn.Module, ignore_proxy: [Callable, List[str]]=None):

    preper_modules_map = []
    for m, name, target in modules_map:
        preper_modules_map.append([getattr(m, name), target])

    select_modules = []
    for target, target_module in model.named_modules():
        if ignore_proxy is not None:
            if isinstance(ignore_proxy, callable):
                if ignore_proxy(target):
                    continue
            elif isinstance(ignore_proxy, list) or isinstance(ignore_proxy, tuple) or isinstance(ignore_proxy, set) or isinstance(ignore_proxy, dict):
                if target in ignore_proxy:
                    continue
            else:
                raise NotImplementedError(f"Unsupport ignore proxy {ignore_proxy}")

        for old_cls, new_cls in preper_modules_map:
            if isinstance(target_module, old_cls):
                select_modules.append([target_module, target, new_cls])
                break

    for target_module, target, new_cls in select_modules:
        quant_module = new_cls.__new__(new_cls)
        for k, val in vars(target_module).items():
            setattr(quant_module, k, val)
        
        quant_module.init_quantizer()
        atoms = target.split(".")
        parent = model.get_submodule(".".join(atoms[:-1]))
        item  = atoms[-1]
        setattr(parent, item, quant_module)
    return model
