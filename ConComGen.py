

import os
import json
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from diffusers.utils import load_image
from modules.scheduling_ddim_L import DDIMScheduler_L
from modules.pipeline_controlnets_masks import StableDiffusionControlNetsPipeline


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

    controlnet_scibble = ControlNetModel.from_pretrained(
        "models/controlnet/models--lllyasviel--sd-controlnet-scribble/snapshots/864edcd5ccc6ee2695eeebea5b4512100c83e7b3", 
        torch_dtype=torch.float16
    )
    controlnet_seg = ControlNetModel.from_pretrained(
        'models/controlnet/models--lllyasviel--sd-controlnet-seg/snapshots/ecdcb5645b5099c9a7500a504fb9ab3f743c4d96',
        torch_dtype=torch.float16
    )
    
    controlnet_sub = [controlnet_scibble, controlnet_seg]
    controlnet_surr = [controlnet_scibble]

    sd_sub = StableDiffusionControlNetPipeline.from_pretrained(
        '../train_lora/stablediffusion/models--runwayml--stable-diffusion-v1-5/snapshots/1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9',
        controlnet= controlnet_sub,
        torch_dtype=torch.float16,
    ).to(device=args.device)
    sd_pipe = StableDiffusionControlNetsPipeline.from_pretrained(
        '../train_lora/stablediffusion/models--runwayml--stable-diffusion-v1-5/snapshots/1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9',
        controlnet= controlnet_surr,
        torch_dtype=torch.float16,
    ).to(device=args.device)
    
    sd_pipe.register_addl_models(sd_sub)
    sd_pipe.schedule = DDIMScheduler_L
    sd_pipe.load_lora_weights('loras', weight_name='vitaglyphy_1.safetensors')
    sd_pipe.enable_xformers_memory_efficient_attention()

    # f
    for folder in tqdm(os.path.join(args.input_folder, 'CCG')):
        output_folder = os.path.join(args.output, folder)
        os.makedirs(output_folder, exist_ok=True)

        # load prompts
        with open(f'./{args.input}/LLM/{folder}.json', 'r') as file:
            sem = json.load(file)
            
        sub_prompt = sem['sub_prompt'] + args.positive_prompt
        surr_prompt = sem['surr_prompt'] + args.positive_prompt

        # load images and mask
        surr_image = load_image(os.path.join(args.input, 'CCG', folder, 'surr.jpg'))
        surr_image = resize_image(surr_image, args.resulation)

        sub_image = load_image(os.path.join(args.input, 'CCG', folder, 'sub.jpg'))
        sub_image = resize_image(sub_image, args.resulation)

        mask = load_image(os.path.join(args.input, 'CCG', folder, 'mask.jpg')).convert('L')
        mask = resize_image(mask, args.resulation)
        mask = transforms.ToTensor()(mask).squeeze().to(device=device)

        # generate
        W, H = surr_image.size
        seed = random.sample(range(100000), 1)

        img = sd_pipe(
            prompt = [surr_prompt],
            image = [surr_image],
            height = H, 
            width = W, 
            num_inference_steps = 50,
            guidance_scale = 15, 
            negative_prompt = args.negative_prompt,
            generator = torch.manual_seed(seed),
            controlnet_conditioning_scale = [0.7],

            addl_prompts = [sub_prompt],
            addl_images = [[sub_image] * len(controlnet_sub)],
            weights = [0.85],
            masks = [mask],
            addl_ctrlnet_conditioning_scale = [1.1],
        ).images[0]

        img.save(os.path.join(output_folder, f'{seed}.jpg'))




if __name__ == 'main':
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('device', type=int, default=0, help='choose the device')
    parser.add_argument('resulation', type=int, default=1024, help='define the output image resulation')
    parser.add_argument('input', type=str, default='./outs', help='Input file')
    parser.add_argument('output', type=str, default='./results', fhelp='Output file')
    parser.add_argument('positive_prompt', type=str, default='complete, beautiful, elegant, artistic, easy background, plain background, simple background, clean background, easy layout, plain layout, simple layout, one, individual, sole, isolated, solitary, detached, alone, subjectival, planar, orderly, negative space')
    parser.add_argument('negative_prompt', type=str, default='vignetting, Camera dark angle, bad, deformed, ugly, lousy anatomy, worst quality, low quality, jpeg artifacts, uplicate, morbid, mutilated, extra things, cut off, deformities, bad anatomy, bad proportions, deformed,blurry, stereoscopic, cluttered background, complex background, cluttered layout, complex layout, multiple, poly')
    args = parser.parse_args()

    main(args)