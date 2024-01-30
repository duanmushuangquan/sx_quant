# 目的1:导出onnx
# 目的2:写linker
import torch
from torchvision.transforms import transforms
from PIL import Image
import torch.nn as nn
from torchvision.models import resnet
import inspect
import os
import math
from enum import Enum


class Logger:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def set_verbose(self, enable=True):
        self.verbose = enable

    def log(self, *msg):
        if self.verbose:
            stack = inspect.stack()[1]
            name = os.path.basename(stack.filename)
            formatted_msg = " ".join(str(m) for m in msg)
            print(f"[{name}:{stack.lineno}]: {formatted_msg}")

logger = Logger(True)

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

device = torch.device("cpu")

def fake_quant(x, scale):
    return torch.round(x / scale.clamp(min=1e-7)).clamp(-127, 127) * scale


class Method(Enum):
    """
    Enumeration to represent the quantization method.
    """
    PerTensor  = 0
    PerChannel = 1

class CalibrationABSMax(object):
    def __init__(self, method):
        """
        Represents a calibration strategy based on histogram data.

        This strategy uses histogram data of the tensors to determine the
        optimal calibration parameters.

        Args:
            method (Method): The method to be used for calibration, either
            PerTensor or PerChannel.
        """
        self.method = method
        self.amax   = 0.0

    def collect(self, x):
        if self.method == Method.PerTensor:
            self.amax = max(self.amax, x.abs().max().item())
        elif self.method == Method.PerChannel:
            self.amax = x.abs().view(x.size(0), -1).amax(1).view(-1, 1, 1, 1)

    def post_compute_amax(self):
        return self.amax

class CalibratorHistogram(object):
    def __init__(self, method):
        """
        Represents a calibration strategy based on histogram data.

        This strategy uses histogram data of the tensors to determine the
        optimal calibration parameters.

        Args:
            method (Method): The method to be used for calibration, either
            PerTensor or PerChannel.
        """
        assert method == Method.PerTensor, f"Unsupport per_channel."
        self.method = method
        self.collect_data   = None 

    def post_compute_amax(self):
        hist, bins_width, absmax = self.collect_data
        device       = hist.device
        num_of_bisn  = hist.numel()
        centers      = torch.linspace(bins_width * 0.5, absmax - bins_width * 0.5, num_of_bisn, device=device, dtype=hist.dtype)
        condidates_start = 128
        condidates   = centers[condidates_start:]

        centers      = centers.view(1, -1)
        condidates   = condidates.view(-1, 1)

        reproject    = fake_quant(centers, condidates / 127)

        different    = ((centers - reproject) ** 2 * hist).sum(dim=1)

        select_index = torch.argmin(different)
        optimal_center = condidates[select_index, 0]

        self.input_absmax_ = optimal_center
        self.collect_data = None
        print(f"min_cost_idx = {select_index + 128}, selected_center = {self.input_absmax_.item()}")
        return optimal_center

    def collect(self, x):
        if self.collect_data is None:
            xabs       = x.abs()
            xabs_max   = xabs.max().item()
            num_bins   = 2048
            bins_width = xabs_max / num_bins
            hist = torch.histc(xabs, num_bins, min=0, max=xabs_max)
            self.collect_data = hist, bins_width, xabs_max
        else:
            prev_hist, bins_width, prev_xabs_max = self.collect_data
            current_xabs        = x.abs()
            current_xabs_max    = current_xabs.max().item()    
            current_absmax      = max(prev_xabs_max, current_xabs_max)
            current_num_bins    = math.ceil(current_absmax / bins_width)

            hist = torch.histc(current_xabs, current_num_bins, min=0, max=current_absmax)
            hist[:prev_hist.numel()]   += prev_hist
            self.collect_data = hist, bins_width, current_absmax

class Quantizer(nn.Module):
    def __init__(self, calibrator):
        """
        Initialize the Quantizer with a given calibration tactics.

        Args:
            calibration (CalibratorHistogram or CalibrationABSMax): The calibration tactics to be used. This must be
            an instance of either CalibratorHistogram or CalibrationABSMax.
        """
        super().__init__()
        self.calibrator     = calibrator
        self.do_collect     = False
        self.do_quant       = False   
        # 1.1 新增属性
        self.do_export      = False

    def post_compute(self):
        amax = self.calibrator.post_compute_amax()
        self.register_buffer("_scale", (amax / 127).clamp(min=1e-7))

    def extra_repr(self):
        scale = "NoScale"
        if hasattr(self, "_scale"):
            if self.calibrator.method == Method.PerChannel:
                scale = f"scale=[min={self._scale.min().item()}, max={self._scale.max().item()}]"
            else:
                scale = f"scale={self._scale.item()}"
        return super().extra_repr() + f"Quantizer({scale}, do_collect={self.do_collect}, do_collect={self.do_collect}, method={self.calibrator.method}, do_quant={self.do_quant})"

    def forward(self, x):
        assert sum([self.do_collect, self.do_quant, self.do_export]) <= 1, f"Invalid configuration: do_collect={self.do_collect}, do_quant={self.do_quant}"

        if self.do_collect:
            self.calibrator.collect(x)
            return x
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
    
class QuantConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_quantizer_  = Quantizer(CalibratorHistogram(Method.PerTensor))
        self.weight_quantizer_ = Quantizer(CalibrationABSMax(Method.PerChannel))
        
    def forward(self, x):
        return super()._conv_forward(self.input_quantizer_(x), self.weight_quantizer_(self.weight), self.bias)

logger.log("replace conv2d to Quantconv2d")
nn.Conv2d = QuantConv2d
model = resnet.resnet18(pretrained=True).eval().to(device)

logger.log("load images")
test_transform = transforms.Compose([
    transforms.Resize(224 + 32, transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])


img_dog = Image.open("dog.jpg")
img_cat = Image.open("cat.jpg")

img_dog_tensor = test_transform(img_dog)[None, ...].to(device)
img_cat_tensor = test_transform(img_cat)[None, ...].to(device)

images_list = [img_dog_tensor, img_cat_tensor]

logger.log("do collect.......")
Linker(model).do_collect = True
with torch.no_grad():          
    for i, img in enumerate(images_list):
        logger.log(f"collect {i}")
        model(img)

Linker(model).post_compute()
Linker(model).do_collect = False

Linker(model).do_quant = True
logger.log("do quant infer.......")
with torch.no_grad():   
    for img in images_list:
        yfp32 = model(img)
        predict = torch.softmax(yfp32, dim=1)
        label = predict.argmax(1).item()
        confidence = predict[0, label].item()
        logger.log(f"This image predict result: label={label}, confidence={confidence}") 

Linker(model).do_quant = False

logger.log("do infer.......")
with torch.no_grad():   
    for img in images_list:
        yfp32 = model(img)
        predict = torch.softmax(yfp32, dim=1)
        label = predict.argmax(1).item()
        confidence = predict[0, label].item()
        logger.log(f"This image predict result: label={label}, confid''ence={confidence}")

Linker(model).do_export = True

torch.onnx.export(model, img_dog_tensor, "./myint8.onnx", opset_version=14)
Linker(model).do_export = False