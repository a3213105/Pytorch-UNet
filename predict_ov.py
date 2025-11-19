import argparse
import logging
import psutil
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
# from torchvision import transforms
from pathlib import Path

# from utils.data_loading import BasicDataset
# from unet import UNet
# from utils.utils import plot_img_and_mask

import openvino as ov
from openvino.preprocess import PrePostProcessor, ResizeAlgorithm
from openvino import Core, Model, get_version, AsyncInferQueue, InferRequest, Layout, Type, Tensor

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--loop', '-l', type=int, default=10, help='Number of classes')
    parser.add_argument('--warmup', '-w', type=int, default=2, help='Number of classes')
    
    return parser.parse_args()

def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))

def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)

def load_torch_model(args) :
    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)
    print(f"mask_values={mask_values}, net.n_classes={net.n_classes}")
    
    ov_path = Path(args.model).with_suffix('.xml')
    if not ov_path.exists() :
        ov_model = ov.convert_model(net, example_input=torch.rand(1, 3, 1024, 1024))
        ov.save_model(ov_model, ov_path, compress_to_fp16=False)
    
    logging.info('Model loaded!')
    return net, device, mask_values

def predict_img_torch(net, full_img, scale, n_classes, out_threshold=0.5):
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale, is_mask=False))
    # print(f"preprocess done, img.shape={img.shape}")
    # img = torch.from_numpy(BasicDataset.preprocess1(img, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(dtype=torch.float32)
    # img = np.repeat(img, 2, axis=0)
    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold
    return mask.long().squeeze().numpy()

def load_ov_model1(model_path: str, amx: int):
    core = ov.Core()
    model = core.read_model(model=model_path)

    ppp = PrePostProcessor(model)
    ppp.input(0).tensor() \
            .set_element_type(Type.u8) \
            .set_shape([1, 1024, 1024, 3]) \
            .set_layout(Layout('NHWC'))
    ppp.input(0).model().set_layout(Layout('NCHW'))

    ppp.input(0).preprocess() \
            .resize(ResizeAlgorithm.RESIZE_BICUBIC_PILLOW, 512, 512) \
            .convert_element_type(Type.f32) \
            .scale(255.0)
    model = ppp.build()
        
    stream_num=1
    data_type = 'bf16' if amx==1 else 'f16' if amx==2 else 'f32'
    hint = 'THROUGHPUT' if stream_num>1 else 'LATENCY'
    config = {}
    config['NUM_STREAMS'] = str(stream_num)
    config['PERF_COUNT'] = 'NO'
    config['INFERENCE_PRECISION_HINT'] = data_type
    config['PERFORMANCE_HINT'] = hint
    compiled_model = core.compile_model(model, 'CPU', config)
    return compiled_model

def predict_img_ov1(net, full_img, scale, n_classes, out_threshold=0.5):
    img = np.asarray(full_img)
    img = np.expand_dims(img, axis=0)
    output = torch.from_numpy(net(img)[0])
    output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
    if n_classes > 1:
        mask = output.argmax(dim=1)
    else:
        mask = torch.sigmoid(output) > out_threshold
    return mask[0].long().squeeze().numpy()

def load_ov_model(model_path: str, amx: int):
    core = ov.Core()
    model = core.read_model(model=model_path)

    ppp = PrePostProcessor(model)
    ppp.input(0).tensor() \
            .set_element_type(Type.u8) \
            .set_layout(Layout('NHWC'))
    ppp.input(0).model().set_layout(Layout('NCHW'))

    ppp.input(0).preprocess() \
            .convert_element_type(Type.f32) \
            .scale(255.0)
    model = ppp.build()
        
    stream_num=1
    data_type = 'bf16' if amx==1 else 'f16' if amx==2 else 'f32'
    hint = 'THROUGHPUT' if stream_num>1 else 'LATENCY'
    config = {}
    config['NUM_STREAMS'] = str(stream_num)
    config['PERF_COUNT'] = 'NO'
    config['INFERENCE_PRECISION_HINT'] = data_type
    config['PERFORMANCE_HINT'] = hint
    compiled_model = core.compile_model(model, 'CPU', config)
    return compiled_model

def predict_img_ov(net, full_img, scale, n_classes, out_threshold=0.5):
    w, h = full_img.size
    newW, newH = int(scale * w), int(scale * h)
    img = full_img.resize((newW, newH), resample=Image.BICUBIC)
    img = np.asarray(img)
    img = np.expand_dims(img, axis=0)
    # img = np.repeat(img, 2, axis=0)
    output = torch.from_numpy(net(img)[0])
    output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
    if n_classes > 1:
        mask = output.argmax(dim=1)
    else:
        mask = torch.sigmoid(output) > out_threshold
    return mask.long().squeeze().numpy()

def torch_infer(in_files, out_files, net):

    class unet_wrapper(torch.nn.Module):
        def __init__(self, net):
            super(unet_wrapper, self).__init__()
            self.net = net

        def forward(self, img, full_img_shape):
            with torch.no_grad():
                output = net(img).cpu()
                # output = F.interpolate(output, (img.shape[2] * 2, img.shape[3] * 2), mode='bilinear')
                output = F.interpolate(output, (full_img_shape[0], full_img_shape[1]), mode='bilinear')
                return output
    
    unet_ov = unet_wrapper(net)
        
    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')
        full_img = Image.open(filename)

        img = torch.from_numpy(BasicDataset.preprocess(None, full_img, args.scale, is_mask=False))
        print(f"preprocess done, img.shape={img.shape}")
        img = img.unsqueeze(0)
        img = img.to(device=device, dtype=torch.float32)
        full_img_shape = torch.tensor([full_img.size[1], full_img.size[0]], dtype=torch.int32).to(device=device)
        print(f"full_img.size={full_img.size}, img.shape={img.shape}, full_img_shape={full_img_shape}")
        ov_model = ov.convert_model(unet_ov, example_input={"img":img, "full_img_shape":full_img_shape})
        # ov_model = ov.convert_model(unet_ov, example_input=img)
        ov.save_model(ov_model, '../UNet-models/unet-all.xml', compress_to_fp16=False)

        warmup = 3
        for j in range(warmup) :
            mask = predict_img(net=net, full_img=full_img, img=img,
                            out_threshold=args.mask_threshold, device=device)
        loop = 50
        st = time.perf_counter()
        for j in range(loop) :
            mask = predict_img(net=net, full_img=full_img, img=img,
                            out_threshold=args.mask_threshold, device=device)
        et = time.perf_counter()
        print(f'Average inference time over loop runs: {(et - st)/loop:.4f} seconds')
    
def infer(in_files, out_files, scale, mask_threshold, warmup, loop,n_classes,  net, predict_img):
    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')
        full_img = Image.open(filename)

        for j in range(warmup) :
            mask = predict_img(net=net, full_img=full_img, scale=scale,
                               n_classes=n_classes, out_threshold=mask_threshold)
        st = time.perf_counter()
        for j in range(loop) :
            mask = predict_img(net=net, full_img=full_img, scale=scale,
                               n_classes=n_classes, out_threshold=mask_threshold)
        et = time.perf_counter()
        print(f'Average inference time over {loop} runs: {(et - st)/loop:.4f} seconds')
    return mask

if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_files = args.input
    out_files = get_output_filenames(args)
    print(f"args={args}")
    
    # net, device, mask_values = load_torch_model(args)
    # net.eval()
    
    # mask_torch = infer(in_files, out_files, args.scale, args.mask_threshold, args.warmup, args.loop, net.n_classes, 
    #       net, predict_img_torch)

    ov_path = Path(args.model).with_suffix('.xml')
    mode_name = ['F32', 'BF16', 'F16']
    for i,name in enumerate(mode_name):            
        ov_net = load_ov_model(ov_path, amx=i)
        mask_ov = infer(in_files, out_files, args.scale, args.mask_threshold, args.warmup, args.loop, args.classes, 
            ov_net, predict_img_ov)
        # exact_equal = np.array_equal(mask_torch, mask_ov)
        # ne_count = np.sum(mask_torch != mask_ov)
        # print(f"{name}是否完全相等:{exact_equal}, 差异数量:{ne_count}")
        process = psutil.Process(os.getpid())        
        # 内存使用（单位：字节）
        mem_info = process.memory_info()
        print(f"RSS: {mem_info.rss / 1024 ** 3:.2f} GB")  # 常驻内存
        print(f"VMS: {mem_info.vms / 1024 ** 3:.2f} GB")  # 虚拟内存
        del ov_net
    # print(f"mask_torch={mask_torch.shape}, mask_ov={mask_ov.shape}")

    # for i,name in enumerate(mode_name):            
    #     ov_net = load_ov_model1(ov_path, amx=i)
    #     mask_ov = infer(in_files, out_files, args.scale, args.mask_threshold, args.warmup, args.loop, args.classes, 
    #         ov_net, predict_img_ov1)
    #     exact_equal = np.array_equal(mask_torch, mask_ov)
    #     ne_count = np.sum(mask_torch != mask_ov)
    #     print(f"{name}是否完全相等:{exact_equal}, 差异数量:{ne_count}")
    #     del ov_net