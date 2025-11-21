import argparse
import logging
import psutil
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
import copy

from utils.data_loading import BasicDataset
from unet import UNet
from concurrent.futures import ThreadPoolExecutor

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
    parser.add_argument('--nstreams', '-n', type=int, default=2, help='Number of ov streams')
    
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

def predict_img_torch(net, full_img, scale, n_classes, out_threshold=0.5, use_amp=False):
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale, is_mask=False))
    # print(f"preprocess done, img.shape={img.shape}")
    # img = torch.from_numpy(BasicDataset.preprocess1(img, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(dtype=torch.float32)
    # img = np.repeat(img, 2, axis=0)
    with torch.no_grad():
        if use_amp:
            with torch.amp.autocast('cpu', dtype=torch.bfloat16) :
                output = net(img).float().cpu()
        else :
            output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold
    return mask.long().squeeze().numpy()

def prepare_input(full_img, scale):
    w, h = full_img.size
    newW, newH = int(scale * w), int(scale * h)
    img = full_img.resize((newW, newH), resample=Image.BICUBIC)
    img = np.asarray(img)
    img = np.expand_dims(img, axis=0)
    return img

def load_ov_model_all(model_path: str, nstreams: int, amx: int):
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
    
    data_type = 'bf16' if amx==1 else 'f16' if amx==2 else 'f32'
    hint = 'THROUGHPUT' if nstreams>1 else 'LATENCY'
    config = {}
    config['NUM_STREAMS'] = str(nstreams)
    config['PERF_COUNT'] = 'NO'
    config['INFERENCE_PRECISION_HINT'] = data_type
    config['PERFORMANCE_HINT'] = hint
    compiled_model = core.compile_model(model, 'CPU', config)
    num_requests = compiled_model.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")
    infer_queue = AsyncInferQueue(compiled_model, num_requests)
    return compiled_model, infer_queue

def predict_img_ov_all(net, full_img, scale, n_classes, out_threshold=0.5):
    img = prepare_input(full_img, scale)
    full_img_size = np.array(full_img.size, dtype=np.int32)
    output = net({0:img, 1:full_img_size})[0]
    output = np.squeeze(output)
    return output

def infer_ov_all(name, in_files, scale, mask_threshold, warmup, n_classes, infer_queue):
    ov_results = {}

    def completion_callback(infer_request: InferRequest, index: any) :
        output = np.squeeze(infer_request.get_output_tensor(0).data)
        ov_results[index] = copy.deepcopy(output)

    infer_queue.set_callback(completion_callback)

    for i in range(warmup):
        img = prepare_input(in_files[i], scale)
        full_img_size = np.array(in_files[i].size, dtype=np.int32)
        infer_queue.start_async({0:img, 1:full_img_size}, userdata=i)#, share_inputs=True)
    infer_queue.wait_all()

    ov_results = {}
    st = time.perf_counter()
    for i, in_file in enumerate(in_files):
        img = prepare_input(in_file, scale)
        full_img_size = np.array(in_file.size, dtype=np.int32)
        infer_queue.start_async({0:img, 1:full_img_size}, userdata=i)#, share_inputs=True)
    infer_queue.wait_all()
    et = time.perf_counter()
    loop = len(in_files)
    total_latency = et - st
    print(f'{name} {loop} runs use {total_latency:.4f}, Average inference time: {total_latency/loop:.4f} seconds, {loop/total_latency:.4f} FPS')
    return ov_results[0]

def convert_ov_all(full_img, net, ov_path):
    ov_path = Path(str(ov_path).replace('.xml', '_all.xml'))
    if ov_path.exists() :
        return ov_path

    class unet_wrapper(torch.nn.Module):
        def __init__(self, net):
            super(unet_wrapper, self).__init__()
            self.net = net

        def forward(self, img, full_img_shape):
            with torch.no_grad():
                output = net(img).cpu()
                output = F.interpolate(output, (full_img_shape[0], full_img_shape[1]), mode='bilinear')
                mask = output.argmax(dim=1)
                return mask.long()
   
    unet_ov = unet_wrapper(net)
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, args.scale, is_mask=False))
    print(f"preprocess done, img.shape={img.shape}")
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    full_img_shape = torch.tensor([full_img.size[0], full_img.size[1]], dtype=torch.int32).to(device=device)
    print(f"full_img.size={full_img.size}, img.shape={img.shape}, full_img_shape={full_img_shape}")
    ov_model = ov.convert_model(unet_ov, example_input={"img":img, "full_img_shape":full_img_shape})
    ov.save_model(ov_model, ov_path, compress_to_fp16=False)
    return ov_path

def load_ov_model(model_path: str, nstreams: int, amx: int):
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
        
    data_type = 'bf16' if amx==1 else 'f16' if amx==2 else 'f32'
    hint = 'THROUGHPUT' if nstreams>1 else 'LATENCY'
    config = {}
    config['NUM_STREAMS'] = str(nstreams)
    config['PERF_COUNT'] = 'NO'
    config['INFERENCE_PRECISION_HINT'] = data_type
    config['PERFORMANCE_HINT'] = hint
    compiled_model = core.compile_model(model, 'CPU', config)
    num_requests = compiled_model.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")
    infer_queue = AsyncInferQueue(compiled_model, num_requests)
    return compiled_model, infer_queue

def predict_img_ov(net, full_img, scale, n_classes, out_threshold=0.5, use_amp=False):
    w, h = full_img.size
    newW, newH = int(scale * w), int(scale * h)
    img = full_img.resize((newW, newH), resample=Image.BICUBIC)
    img = np.asarray(img)
    img = np.expand_dims(img, axis=0)
    output = torch.from_numpy(net(img)[0])
    output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
    if n_classes > 1:
        mask = output.argmax(dim=1)
    else:
        mask = torch.sigmoid(output) > out_threshold
    return mask.long().squeeze().numpy()

def infer_ov(name, in_files, scale, mask_threshold, warmup, n_classes, infer_queue):
    results = {}
    def completion_callback(infer_request: InferRequest, index: any) :
        nid = index[0]
        full_img = index[1]
        output = torch.from_numpy(infer_request.get_output_tensor(0).data)
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold
        results[nid] = copy.deepcopy(mask.long().squeeze().numpy())
        return 

    infer_queue.set_callback(completion_callback)
    
    for i in range(warmup):
        img = prepare_input(in_files[i], scale)
        infer_queue.start_async({0: img}, userdata=[i,in_files[i]])
    infer_queue.wait_all()

    results = {}
    st = time.perf_counter()
    for i, in_file in enumerate(in_files):
        img = prepare_input(in_file, scale)
        infer_queue.start_async({0: img}, userdata=[i,in_file])
    infer_queue.wait_all()
    et = time.perf_counter()
    loop = len(in_files)
    total_latency = et - st
    print(f'{name} {loop} runs use {total_latency:.4f}, Average inference time: {total_latency/loop:.4f} seconds, {loop/total_latency:.4f} FPS')
    return results[0]

def process_ov_output(args):
    output, img_size, n_classes, out_threshold = args
    output = torch.from_numpy(output)
    # print(f"output={output.shape}, img_size={img_size}, n_classes={n_classes}, out_threshold={out_threshold}")
    output = F.interpolate(output, (img_size[1], img_size[0]), mode='bilinear')
    if n_classes > 1:
        mask = output.argmax(dim=1)
    else:
        mask = torch.sigmoid(output) > out_threshold
    return mask.long().squeeze().numpy()

def infer_ov_multithread(name, in_files, scale, mask_threshold, warmup, n_classes, infer_queue):
    ov_results = {}

    def completion_callback(infer_request: InferRequest, index: any) :
        ov_results[index] = copy.deepcopy(infer_request.get_output_tensor(0).data)

    infer_queue.set_callback(completion_callback)

    for i in range(warmup):
        img = prepare_input(in_files[i], scale)
        infer_queue.start_async({0: img}, userdata=i)#, share_inputs=True)
    infer_queue.wait_all()

    ov_results = {}
    st = time.perf_counter()
    for i, in_file in enumerate(in_files):
        img = prepare_input(in_file, scale)
        infer_queue.start_async({0: img}, userdata=i)#, share_inputs=True)
    infer_queue.wait_all()
    
    max_workers = len(infer_queue)
    args_list = [(ov_results[i], in_file.size, n_classes, mask_threshold) for i, in_file in enumerate(in_files)]
    results = []
    if max_workers < 2:
        for args in args_list:
            results.append(process_ov_output(args))
    else :
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_ov_output, args_list))
    et = time.perf_counter()
    loop = len(in_files)
    total_latency = et - st
    print(f'{name} {loop} runs use {total_latency:.4f}, Average inference time: {total_latency/loop:.4f} seconds, {loop/total_latency:.4f} FPS')
    return results[0]

def infer(name, in_files, scale, mask_threshold, warmup, loop, n_classes, net, predict_img, use_amp=False):
    for i in range(warmup) :
        mask = predict_img(net=net, full_img=in_files[i], scale=scale,
                           n_classes=n_classes, out_threshold=mask_threshold, use_amp=use_amp)

    st = time.perf_counter()
    for i in range(loop) :
        mask = predict_img(net=net, full_img=in_files[i], scale=scale,
                           n_classes=n_classes, out_threshold=mask_threshold, use_amp=use_amp)
    et = time.perf_counter()
    # loop = len(in_files)
    total_latency = et - st
    print(f'{name} {loop} runs use {total_latency:.4f}, Average inference time: {total_latency/loop:.4f} seconds, {loop/total_latency:.4f} FPS')
    return mask

def prepare_images(in_files, ncount) :
    image_list = []
    for i in range(ncount) :
        for filename in in_files:
            full_img = Image.open(filename)
            image_list.append(full_img)
            if len(image_list) >= ncount:
                return image_list
    return image_list
    
if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    if args.warmup < args.nstreams:
        args.warmup =  args.nstreams
    if args.loop < args.nstreams * 3:
        args.loop =  args.nstreams * 3

    print(f"args={args}")
    
    in_files = prepare_images(args.input, args.loop if args.loop > args.warmup else args.warmup)
    
    net, device, mask_values = load_torch_model(args)
    net.eval()
    
    mask_torch = infer("Torch_F32", in_files, args.scale, args.mask_threshold, 1, 1, net.n_classes, 
          net, predict_img_torch, False)

    mask_torch_BF16 = infer("Torch_BF16", in_files, args.scale, args.mask_threshold, 1, 1, net.n_classes, 
          net, predict_img_torch, True)
    
    exact_equal = np.array_equal(mask_torch, mask_torch_BF16)
    ne_count = np.sum(mask_torch != mask_torch_BF16)
    print(f"TorchBF16是否完全相等:{exact_equal}, 差异数量:{ne_count}")

    ov_path = Path(args.model).with_suffix('.xml')
    mode_name = ['F32', 'BF16', 'F16']
    for i,name in enumerate(mode_name):            
        ov_net, ov_infer_q = load_ov_model(ov_path, args.nstreams, amx=i)

        mask_ov = infer(f"OV_sync_{name}", in_files, args.scale, args.mask_threshold, args.warmup, args.loop, args.classes, 
            ov_net, predict_img_ov)
        exact_equal = np.array_equal(mask_torch, mask_ov)
        ne_count = np.sum(mask_torch != mask_ov)
        print(f"{name}是否完全相等:{exact_equal}, 差异数量:{ne_count}")

        # mask_ov = infer_ov(in_files, args.scale, args.mask_threshold, args.warmup, args.classes, ov_infer_q)
        # exact_equal = np.array_equal(mask_torch, mask_ov)
        # ne_count = np.sum(mask_torch != mask_ov)
        # print(f"{name}是否完全相等:{exact_equal}, 差异数量:{ne_count}")
        
        mask_ov = infer_ov_multithread(f"OV_async_{name}", in_files, args.scale, args.mask_threshold, args.warmup, args.classes, ov_infer_q)
        exact_equal = np.array_equal(mask_torch, mask_ov)
        ne_count = np.sum(mask_torch != mask_ov)
        print(f"{name}是否完全相等:{exact_equal}, 差异数量:{ne_count}")

        process = psutil.Process(os.getpid())        
        # 内存使用（单位：字节）
        mem_info = process.memory_info()
        print(f"RSS: {mem_info.rss / 1024 ** 3:.2f} GB")  # 常驻内存
        print(f"VMS: {mem_info.vms / 1024 ** 3:.2f} GB")  # 虚拟内存
        del ov_net

    ov_path = convert_ov_all(in_files[0], net, ov_path)
    print(f"ov_path={ov_path}")
    mode_name = ['F32', 'BF16', 'F16']
    for i,name in enumerate(mode_name):            
        ov_net, ov_infer_q = load_ov_model_all(ov_path, args.nstreams, amx=i)
       
        mask_ov = infer_ov_all(f"OV_ALL_async_{name}", in_files, args.scale, args.mask_threshold, args.warmup, args.classes, ov_infer_q)
        exact_equal = np.array_equal(mask_torch, mask_ov)
        ne_count = np.sum(mask_torch != mask_ov)
        print(f"{name}是否完全相等:{exact_equal}, 差异数量:{ne_count}")

        process = psutil.Process(os.getpid())        
        # 内存使用（单位：字节）
        mem_info = process.memory_info()
        print(f"RSS: {mem_info.rss / 1024 ** 3:.2f} GB")  # 常驻内存
        print(f"VMS: {mem_info.vms / 1024 ** 3:.2f} GB")  # 虚拟内存
        del ov_net