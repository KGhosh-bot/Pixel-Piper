import pprint
from typing import Union, Tuple,List
#import matplotlib.pyplot as plt

from pathlib import Path
import torch
from PIL import Image

from masked_logic import PredicatedDiffPipeline
from spatial_processor import cal_attn_mask_xl
import utils
from utils import AttentionStore
from torch.nn import functional as F
import argparse
import numpy as np

import torch.nn as nn
from torch.cuda.amp import autocast
import copy
import random
import wandb
from config_parser import parse_arguments

import warnings
warnings.filterwarnings("ignore", category=UserWarning)



# Models dictionary
models_dict = {
    "Juggernaut": "RunDiffusion/Juggernaut-XL-v8",
    "RealVision": "SG161222/RealVisXL_V4.0",
    "SDXL": "stabilityai/stable-diffusion-xl-base-1.0",
    "Unstable": "stablediffusionapi/sdxl-unstable-diffusers-y"
}

# Styles dictionary
styles = {
    "(No_style)": ("{prompt}", ""),
    "Japanese_Anime": ("anime artwork illustrating {prompt}. created by japanese anime studio. highly emotional. best quality, high resolution, (Anime Style, Manga Style:1.3), Low detail, sketch, concept art, line art, webtoon, manhua, hand drawn, defined lines, simple shades, minimalistic, High contrast, Linear compositions, Scalable artwork, Digital art, High Contrast Shadows", "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"),
    "Digital/Oil_Painting": ("{prompt} . (Extremely Detailed Oil Painting:1.2), glow effects, godrays, Hand drawn, render, 8k, octane render, cinema 4d, blender, dark, atmospheric 4k ultra detailed, cinematic sensual, Sharp focus, humorous illustration, big depth of field", "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"),
    "Pixar/Disney_Character": ("Create a Disney Pixar 3D style illustration on {prompt} . The scene is vibrant, motivational, filled with vivid colors and a sense of wonder.", "lowres, bad anatomy, bad hands, text, bad eyes, bad arms, bad legs, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, blurry, grayscale, noisy, sloppy, messy, grainy, highly detailed, ultra textured, photo"),
    "Photographic": ("cinematic photo {prompt} . Hyperrealistic, Hyperdetailed, detailed skin, matte skin, soft lighting, realistic, best quality, ultra realistic, 8k, golden ratio, Intricate, High Detail, film photography, soft focus", "drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"),
    "Comic_book": ("comic {prompt} . graphic illustration, comic art, graphic novel art, vibrant, highly detailed", "photograph, deformed, glitch, noisy, realistic, stock photo, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"),
    "Line_art": ("line art drawing {prompt} . professional, sleek, modern, minimalist, graphic, line art, vector graphics", "anime, photorealistic, 35mm film, deformed, glitch, blurry, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, mutated, realism, realistic, impressionism, expressionism, oil, acrylic, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"),
    "Black_and_White_Film_Noir": ("{prompt} . (b&w, Monochromatic, Film Photography:1.3), film noir, analog style, soft lighting, subsurface scattering, realistic, heavy shadow, masterpiece, best quality, ultra realistic, 8k", "anime, photorealistic, 35mm film, deformed, glitch, blurry, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, mutated, realism, realistic, impressionism, expressionism, oil, acrylic, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"),
    "Isometric_Rooms": ("Tiny cute isometric {prompt} . in a cutaway box, soft smooth lighting, soft colors, 100mm lens, 3d blender render", "anime, photorealistic, 35mm film, deformed, glitch, blurry, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, mutated, realism, realistic, impressionism, expressionism, oil, acrylic, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"),
}

def load_model(config):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    if config.sd_2_1:
        stable_diffusion_version = "stabilityai/stable-diffusion-2-1-base"
    else:
        stable_diffusion_version = "CompVis/stable-diffusion-v1-4"
    stable = PredicatedDiffPipeline.from_pretrained(stable_diffusion_version).to(device)
    stable.safety_checker = lambda images, **kwargs: (images, False)
    return stable


def get_indices_to_alter(stable, prompt: str) -> List[int]:
    token_idx_to_word = {idx: stable.tokenizer.decode(t)
                         for idx, t in enumerate(stable.tokenizer(prompt)['input_ids'])
                         if 0 < idx < len(stable.tokenizer(prompt)['input_ids']) - 1}
    pprint.pprint(token_idx_to_word)
    token_indices = input("Please enter the a comma-separated list indices of the tokens you wish to "
                          "alter (e.g., 2,5): ")
    token_indices = [int(i) for i in token_indices.split(",")]
    print(f"Altering tokens: {[token_idx_to_word[i] for i in token_indices]}")
    return token_indices


def run_on_prompt(prompt: List[str],
                  model: PredicatedDiffPipeline,
                  negative_prompt:'str',
                  controller: AttentionStore,
                  token_indices: List[int],
                  seed: torch.Generator,
                  config,
                  ) -> Image.Image:
    if controller is not None:
        utils.register_attention_control(model, controller)#, config.id_length,config.sa32,config.sa64,config.height,config.width)
    outputs,loss_value_per_step,attention_for_obj_t = model(prompt=prompt,
                    neg_prompt = config.neg_prompt,
                    attention_store=controller,
                    indices_to_alter=token_indices,
                    attention_res=config.attention_res,
                    guidance_scale=config.guidance_scale,
                    negative_prompt=negative_prompt,
                    generator=seed,
                    num_inference_steps=config.n_inference_steps,
                    max_iter_to_alter=config.max_iter_to_alter,
                    run_standard_sd=config.run_standard_sd,
                    run_attention_sd = config.run_attention_sd,
                    thresholds=config.thresholds,
                    scale_factor=config.scale_factor,
                    scale_range=config.scale_range,
                    smooth_attentions=config.smooth_attentions,
                    sigma=config.sigma,
                    kernel_size=config.kernel_size,
                    sd_2_1=config.sd_2_1,
                    attention_corr_indices=config.attention_corr_indices,
                    attention_leak_indices=config.attention_leak_indices,
                    attention_exist_indices=config.attention_exist_indices,
                    attention_possession_indices=config.attention_possession_indices,
                    attention_save_t=config.attention_save_t,
                    loss_function=config.loss_function,
                    )
    print("IMAGE",type(outputs.images))
    #print("att",attention_maps.shape)

    image = outputs.images[0]
    return image,loss_value_per_step,attention_for_obj_t



def apply_style(style_name: str, positives: list, negative: str = ""):
    p, n = styles.get(style_name, styles["(No_style)"])
    return [p.replace("{prompt}", positive) for positive in positives], n + ' ' + negative

# Main function
def main(args):
    # setup_seed(args.seed)
    # generator = torch.Generator(device="cuda").manual_seed(args.seed)
    wandb_key = 'f8c905c1d6034e22e3e595ee05be463091170eb8'
    wandb.login(key = wandb_key)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    # Convert paths
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize Wandb
    wandb.init(project="Masked Logic", config={
        "model_name": args.model_name,
        "style_name": args.style_name,
        "seeds": args.seeds,
        "height": args.height,
        "width": args.width,
        "n_inference_steps": args.n_inference_steps,
        "guidance_scale": args.guidance_scale,
        "general_prompt": args.general_prompt,
        "negative_prompt": args.negative_prompt,
        "prompt_array": args.prompt_array,
        "id_length": args.id_length,
        "sa32": args.sa32,
        "sa64": args.sa64,
        "attention_corr_indices": args.attention_corr_indices,
        "attention_leak_indices": args.attention_leak_indices,
        "attention_exist_indices": args.attention_exist_indices,
        "attention_possession_indices": args.attention_possession_indices,
    })
    print(args.attention_corr_indices)
    # Load the model
    sd_model_path = models_dict[args.model_name]
    # pipe = AutoPipelineForText2Image.from_pretrained(sd_model_path, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    stable = PredicatedDiffPipeline.from_pretrained(sd_model_path)#.to(device)
    stable.safety_checker = lambda images, **kwargs: (images, False)
    stable.enable_model_cpu_offload()
    stable.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
    # pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    # pipe.scheduler.set_timesteps(args.num_steps)
    unet = stable.unet

    global mask1024,mask4096
    mask1024, mask4096 = cal_attn_mask_xl(5,args.id_length,args.sa32,args.sa64,args.height,args.width,device=device,dtype= torch.float16)

    # Generate images
    prompts = [args.general_prompt + " " + prompt for prompt in args.prompt_array]
    # id_prompts = prompts[:args.id_length]
    # real_prompts = prompts[args.id_length:]
    
    id_prompts, negative_prompt = apply_style(args.style_name, prompts, args.negative_prompt)
    # images = pipe(id_prompts, num_inference_steps=args.num_steps, guidance_scale=args.guidance_scale, height=args.height, width=args.width, negative_prompt=negative_prompt, generator=generator).images

    token_indices = get_indices_to_alter(stable, id_prompts) if args.token_indices is None else args.token_indices
    tokenizer = stable.tokenizer
    tokens = tokenizer.encode(id_prompts)

    if args.run_attention_sd:
        att_output_path = args.output_path /'proposed'/args.general_prompt
        att_output_path.mkdir(exist_ok=True, parents=True)
    else:
        att_output_path = args.output_path /'vanilla'/args.general_prompt
        att_output_path.mkdir(exist_ok=True, parents=True)

    images = []
    attend_maps_for_tokens = []
    for seed in args.seeds:
        print(f"Seed: {seed}")
        #seed += 25
        g = torch.Generator('cuda').manual_seed(seed)
        controller = AttentionStore()
        image,loss_value_per_step,attention_maps = run_on_prompt(prompt=id_prompts,
                                                                                    model=stable,
                                                                                    negative_prompt=negative_prompt,
                                                                                    controller=controller,
                                                                                    token_indices=token_indices,
                                                                                    seed=g,
                                                                                    config=args,
                                                                                    )
        for i, img in enumerate(image):
            img.save(att_output_path/f'{seed}_{args.prompt_array[i]}.jpeg')
            wandb.log({f"image_{args.prompt_array[i]}": wandb.Image(img)})
        images.append(image)

    wandb.finish()

if __name__ == "__main__":
    args = parse_arguments()  # Get arguments from config_parser.py
    main(args)
