

import os
import json
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from controlnet_aux import HEDdetector
from diffusers import StableDiffusionDepth2ImgPipeline

from diffusers.utils import load_image

def resize_image(input_image, res=512):
    W, H = input_image.size
    H, W = float(H), float(W)
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    if res is None:
        size_ = (W, H)
    else:
        k = res / min(W, H)
        size_ = (int(W * k), int(H * k))

    img_rsz = input_image.resize(size_)
    return img_rsz



def main(args):

    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')


    d2i = StableDiffusionDepth2ImgPipeline.from_pretrained(
        pretrained_model_name_or_path='/home/fkl/workspace/wordart/models/depth2image/models--stabilityai--stable-diffusion-2-depth/snapshots/d49bafe6f381b0fe37ccfc4c8f6a23424b09d6ef',
        torch_dtype=torch.float16,
    ).to(device=device)

    scribble_preprocess = HEDdetector.from_pretrained(
        pretrained_model_or_path='/home/fkl/workspace/wordart/models/preprocess/hed/models--lllyasviel--Annotators/snapshots/982e7edaec38759d914a963c48c4726685de7d96',
        filename='ControlNetHED.pth', 
    ).to(device=device)

    seeds = random.sample(range(1000), 5)

    for folder in tqdm(os.listdir(args.input)):

        output_folder = os.path.join(args.output, folder)
        os.makedirs(output_folder, exist_ok=True)

        # load prompts
        with open(os.path.join(os.path.dirname(args.output), 'LLM', f'{folder}.json'), 'r') as file:
            sem = json.load(file)

        prompt = sem['sub_prompt'] + args.positive_prompt

        sub_image = load_image(os.path.join(args.input, folder, 'sub.jpg'))
        sub_image = resize_image(sub_image, args.resolution)
        surr_image = load_image(os.path.join(args.input, folder, 'surr.jpg'))
        surr_image = resize_image(surr_image, args.resolution)
        W, H = sub_image.size

        for idx, seed in enumerate(seeds):
            sub_depth = d2i(
                prompt="a black and white drawing of " + prompt, 
                image=sub_image, 
                guidance_scale=10,
                height=H,
                width=W,
                negative_prompt=args.negative_prompt, 
                strength=0.76, 
                generator=torch.manual_seed(seed)
            ).images[0]

            sub_depth.save(os.path.join(output_folder, f'sub_{idx}.jpg'))

        surr_scribble = scribble_preprocess(surr_image, detect_resolution=args.resolution, image_resolution=args.resolution)  # shortest side 1024
        surr_scribble.save(os.path.join(output_folder, f'surr.jpg'))

        os.system(f'cp {os.path.join(args.input, folder, "mask.jpg")} {output_folder}/mask.jpg')



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0, help='choose the device')
    parser.add_argument('--resolution', type=int, default=512, help='define the output image resolution')
    parser.add_argument('--input', type=str, default='./preds', help='Input file')
    parser.add_argument('--output', type=str, default='./outs/CCG', help='Output file')
    parser.add_argument('--positive_prompt', type=str, default='complete, beautiful, elegant, artistic, easy background, plain background, simple background, clean background, easy layout, plain layout, simple layout, one, individual, sole, isolated, solitary, detached, alone, subjectival, planar, orderly, negative space')
    parser.add_argument('--negative_prompt', type=str, default='vignetting, Camera dark angle, bad, deformed, ugly, lousy anatomy, worst quality, low quality, jpeg artifacts, uplicate, morbid, mutilated, extra things, cut off, deformities, bad anatomy, bad proportions, deformed,blurry, stereoscopic, cluttered background, complex background, cluttered layout, complex layout, multiple, poly')
    args = parser.parse_args()

    main(args)