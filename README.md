# sxquant: A tiny quantization framework for better collect speed.

## 0.0 The core code is in the sx_quant.py file, copy it into the project workspace, and start ptq quantization

## 1.0 replace torch.nn.Conv2d to QuantConv2d

- The first method

```python
from sx_quant import QuantConv2d
from torchvision.models import resnet
import torch.nn as nn

# 1.0 Original model
# model = resnet.resnet18(True)
# print(model)

# 2.0 In a very simple way, replace nn.Conv2d with QuantConv2d
nn.Conv2d = QuantConv2d
model = resnet.resnet18(True)
print(model)


""" The following is the information displayed on the terminal
ResNet(
  (conv1): QuantConv2d(
    3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    (input_quantizer_): Quantizer(NoneScale, do_collect=False, do_quant=False, do_export=False, method=Method.PerTensor)
    (weight_quantizer_): Quantizer(NoneScale, do_collect=False, do_quant=False, do_export=False, method=Method.PerChannel)
  )
...
"""
```

- The second method

```python
# Most models are created first and then loaded with ckpt, so a second method is provided
from sx_quant import replace_modules

model = resnet.resnet18(True)
# Some models may have a statement here to load ckpt
replace_modules(model)
print(model)


```


## 2.0 To do post training quantization calibration.

```python
# Give your model to Linker and turn on the collect switch
Linker(model).do_collect = True      

# The training set data dataloader was used to do the model forward and collect the dynam#ic range
# exp：collect_stats(model, train_dataloader, device, calib_batch_size)

# turn off the collect switch
Linker(model).do_collect = False

# Calculation scale
Linker(model).post_compute()

```


## 3.0 Turn on quantization inference and simulate quantization error

```python
# Set the quantization switch to True
Linker(model).do_quant = True

# evaluate
ap_quant = evaluate(model, valloader)

# Set the quantization switch to False
Linker(model).do_quant = False
```


## 4.0 Export onnx files with qdq nodes

```python
# Set the export switch to True
Linker(model).do_export = True
device = next(model.parameters()).device
model.float()
input_dummy = torch.zeros(1,3,640,640, device=device)
model.eval()

# export
with torch.no_grad():
    torch.onnx.export(model, input_dummy, "your_onnx_name.onnx", opset_version=14)

#  Set the export switch to False
Linker(model).do_export = False
```
