import torchvision.models as models
from mqbench.convert_deploy import convert_deploy
from mqbench.prepare_by_platform import prepare_by_platform, BackendType
from mqbench.utils.state import enable_calibration, enable_quantization, enable_calibration_woquantization
from mqbench.advanced_ptq import ptq_reconstruction

import torch
from torch import nn
import os
from tqdm import tqdm
from imagenet import load_data
from evaluate import *

class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)

    def __getattr__(self, key):
        value = self[key]
        if isinstance(value, dict):
            value = DotDict(value)
        return value

# 加载预训练的resnet50模型
model = torch.load("./resnet50.pth")
model.eval()
model.cuda()


# 数据路径和批处理大小
data_path = "../ILSVRC2012"

def load_calibrate_data(train_loader, cali_batchsize):
    cali_data = []
    for i, batch in enumerate(train_loader):
        cali_data.append(batch[0])
        if i + 1 == cali_batchsize:
            break
    return cali_data

train_loader, val_loader = load_data(path=data_path,batch_size=64)
cali_data = load_calibrate_data(train_loader, cali_batchsize=16)

# with torch.no_grad():
#     evaluate(val_loader,model,device="cuda", print_freq=20)

extra_prepare_dict = {    
    'extra_qconfig_dict':{
        'w_observer': 'MinMaxObserver',
        'a_observer': 'EMAMinMaxObserver',
        'w_fakequantize': 'AdaRoundFakeQuantize',
        'a_fakequantize': 'QDropFakeQuantize',# 后面prob设置为1，因此这里实际用的是BRECQ
    },
    'w_qscheme':{
        'bit': 4,
        'symmetry': False,
        'per_channel': True,
        'pot_scale': False,
    },
    'a_qscheme':{
        'bit': 4,
        'symmetry': False,
        'per_channel': False,
        'pot_scale': False,
    }
}
extra_prepare_dict = DotDict(extra_prepare_dict)


backend_type = BackendType.Tensorrt
model = prepare_by_platform(model,backend_type,extra_prepare_dict)
model.cuda()


ptq_reconstruction_config = {
    'pattern': 'block',                   # 按块量化
    'scale_lr': 4.0e-5,                   # 参数的学习率
    'warm_up': 0.2,                       # 不加正则化直接向下或向上取整的迭代次数占最大迭代次数的比例
    'weight': 0.01,                       # 正则项的损失权重
    'max_count': 20000,                   # 迭代步数
    'b_range': [20,2],                    # beta 衰减范围
    'keep_gpu': True,                     # 校准数据是否保留在 GPU 上
    'round_mode': 'learned_hard_sigmoid', # 重建权重的方式
    'prob': 1.0,                          # QDROP 的丢弃概率，Adaround 和 BRECQ 方法中此值为 1.0
}
ptq_reconstruction_config = DotDict(ptq_reconstruction_config)


with torch.no_grad():
    enable_calibration_woquantization(model, quantizer_type='act_fake_quant')
    # 向前传播收集数据 
    for batch in cali_data:
        model(batch.cuda())
    enable_calibration_woquantization(model, quantizer_type='weight_fake_quant')
    model(cali_data[0].cuda())

# 执行PTQ重建
model = ptq_reconstruction(model, cali_data, ptq_reconstruction_config)

enable_quantization(model)

with torch.no_grad():
    evaluate(val_loader,model,device="cuda", print_freq=20)

torch.save(model.state_dict(), "./mqbench_w4a4_resnet50_params.pth")