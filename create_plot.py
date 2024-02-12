import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image

from transformers import CLIPTextModel, CLIPTokenizer
from modified_diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel
from torch import autocast

from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import os
import time
import pandas as pd
import pickle
from collections import OrderedDict


# Set device
torch_device = "cuda" if torch.cuda.is_available() else "cpu"
tic, toc = (time.time, time.time)

import copy

from probe_src.probe_depth_datasets import ProbeOSDataset, threshold_target
from probe_src.probe_utils import dice_coeff, weighted_f1, plt_test_results, train, test, ModuleHook
from probe_src.vis_partially_denoised_latents import generate_image, _init_models, get_gpu_memory
from probe_src.probe_models import probeLinearDense

# Reproducibility
import random
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# libraries for Intrinsic (shading and illumination)
from chrislib.general import show, view, uninvert
from chrislib.data_util import load_image
from intrinsic.pipeline import run_pipeline
from intrinsic.model_util import load_models

import warnings
warnings.filterwarnings('ignore') 

from plot_config import getConfig

args = getConfig()


def generate_image_modified(prompt, seed_num, tokenizer, text_encoder, net, vae, scheduler,
                   batch_size=1, 
                   height=512, width=512, 
                   num_inference_steps=15, 
                   guidance_scale=7.5,
                   modified_unet=None,
                   at_step=None,
                   return_latents=False,
                   stop_at_step=None,
                   ):
    torch.manual_seed(seed_num)
    
    # Prep text 
    text_input = tokenizer(prompt, padding="max_length", 
                           max_length=tokenizer.model_max_length, 
                           truncation=True, return_tensors="pt")

    max_length = text_input.input_ids.shape[-1]

    uncond_input = tokenizer("", padding="max_length", 
                             max_length=max_length, return_tensors="pt")

    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0] 

    text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    scheduler.set_timesteps(num_inference_steps)

    # Prep latents
    latents = torch.randn(
        (batch_size, net.in_channels, height // 8, width // 8)
    )

    latents = latents.to(torch_device)
    latents = latents * scheduler.sigmas[0] # Need to scale to match k

    # Loop
    output_dict = {}
    with autocast("cuda"):
        for j, t in enumerate(scheduler.timesteps):
            sigma = scheduler.sigmas[j]
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

            with torch.no_grad():
                output = net(latent_model_input, t, encoder_hidden_states=text_embeddings)
                noise_pred = output["sample"]
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

            # perform guidance
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, j, latents)["prev_sample"]
            
            transformed_latents = 1 / 0.18215 * latents
            
            with torch.no_grad():
                decoded = vae.decode(transformed_latents)
                try:
                    image = decoded["sample"]
                except:
                    image = decoded
                    
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
                image = (image * 255).round().astype("uint8")[0]
            output_dict[j] = image  
        
    return output_dict


def plot_image_steps(image_index, compact=False, directory_path='datasets/steps'):
    dots = Image.open('resources/dot dot dot.png')

    def get_images(prefix): # prefix: 'image', 'mask', 'TRACER', 'depth', 'MiDaS', 'shading', 'Intrinsic'
        directory = os.path.join(directory_path, f'{prefix}_steps')
        files = [f for f in os.listdir(directory) if image_index in f and f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        files = sorted(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        images = [Image.open(os.path.join(directory, file)) for file in files]
        return images

    images = get_images('image')
    masks = get_images('mask')
    TRACER = get_images('TRACER')
    depths = get_images('depth')
    MiDaS = get_images('MiDaS')
    shadings = get_images('shading')
    Intrinsic = get_images('Intrinsic')

    n_rows = 7
    n_cols = 7 if compact else len(images)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.2, n_rows * 1.5))
    subtitle_size = 16
    axes[1][0].set_title('Probe Mask', fontsize=subtitle_size)
    axes[2][0].set_title('Image Mask', fontsize=subtitle_size)
    axes[3][0].set_title('Probe Depth', fontsize=subtitle_size)
    axes[4][0].set_title('Image Depth', fontsize=subtitle_size)
    axes[5][0].set_title('Probe Shading', fontsize=subtitle_size)
    axes[6][0].set_title('Image Shading', fontsize=subtitle_size)

    for i, (image, mask, mask_syn, depth, depth_syn, shading, shading_syn) in enumerate(zip(images, masks, TRACER, depths, MiDaS, shadings, Intrinsic)):
        if compact & (i >= 5) & (i <= 12):
            continue

        if compact & (i == 13):
            for row in np.arange(n_rows):
                axes[row][5].imshow(dots)
                axes[row][5].axis('off')
            continue

        if compact & (i == 14):
            i = 6
            axes[0][i].set_title(f'step {15}', fontsize=15)
            image, mask, depth, depth_syn, shading, shading_syn = images[-1], masks[-1], depths[-1], MiDaS[-1], shadings[-1], Intrinsic[-1]
        else:
            axes[0][i].set_title(f'step {i+1}', fontsize=15)

        # Plot original images
        axes[0][i].imshow(image)
        axes[0][i].axis('off')

        # Plot masks
        axes[1][i].imshow(mask)
        axes[1][i].axis('off')

        axes[2][i].imshow(mask_syn, cmap='gray')
        axes[2][i].axis('off')

        # Plot depths
        axes[3][i].imshow(depth)
        axes[3][i].axis('off')

        axes[4][i].imshow(depth_syn)
        axes[4][i].axis('off')

        # Plot shadings
        axes[5][i].imshow(shading) # vmin and vmax do not make any difference
        axes[5][i].axis('off')

        axes[6][i].imshow(shading_syn)
        axes[6][i].axis('off')


    plt.tight_layout()

    folder_dir = "plots"
    if not os.path.exists(folder_dir):
        os.makedirs(folder_dir)

    if compact:
        plot_name = f'{image_index} COMPACT.png'
    else:
        plot_name = f'{image_index}.png'

    plt.savefig(os.path.join(folder_dir, plot_name), bbox_inches='tight', pad_inches=0.1)
    plt.show()

        

def main(args):
    input_dims_dict = {"down_0": 320,
                       "down_1": 640,
                       "down_2": 1280,
                       "up_1": 1280,
                       "up_2": 640,
                       "up_3": 320,
                       "mid_0": 1280}

    scale_dict = {"down_0": 8,
                  "down_1": 16,
                  "down_2": 32,
                  "up_1": 32,
                  "up_2": 16,
                  "up_3": 8,
                  "mid_0": 64}

    # Preset probing arguments
    output_dir_name = "attn1_out"
    layer_name = "transformer_blocks.0.attn1.to_out.0"
    block_type = "attentions"
    postfix = "self_attn_out"

    # Alter these arguments as needed. Example: segmentation_probe_down_0_attn1_out_0_final.pth
    steps = np.arange(15)
    block = "up"
    block_ind = 3
    layer_ind = 0 # or is 1 better?

    vae_pretrained="CompVis/stable-diffusion-v1-4"
    CLIPtokenizer_pretrained="openai/clip-vit-large-patch14"
    CLIPtext_encoder_pretrained="openai/clip-vit-large-patch14"
    denoise_unet_pretrained="CompVis/stable-diffusion-v1-4"

    vae, tokenizer, text_encoder, unet, scheduler = _init_models(vae_pretrained=vae_pretrained,
                                                                 CLIPtokenizer_pretrained=CLIPtokenizer_pretrained,
                                                                 CLIPtext_encoder_pretrained=CLIPtext_encoder_pretrained,
                                                                 denoise_unet_pretrained=denoise_unet_pretrained)

    # Load the MiDaS model for depth
    midas_model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    midas = torch.hub.load("intel-isl/MiDaS", midas_model_type).to(torch_device)
    midas.eval();
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.dpt_transform

    print("Successfully loaded the MiDaS model for depth")

    # Load the Intrinsic model for shading/illumination
    intrinsic_model = load_models('paper_weights')

    print("Successfully loaded the Intrinsic model for shading/illumination")
    
    test_df = pd.read_csv("test_split_prompts_seeds.csv", encoding = "ISO-8859-1")
    prompt_ind = args.prompt_ind
    prompt, seed_num = test_df.loc[test_df['prompt_inds'] == prompt_ind].iloc[0][["prompts", "seeds"]]
    image_index = f'prompt_{prompt_ind}_seed_{seed_num}'
    
    
    # Create the necessary directories to store resulting images
    data_path = "datasets/steps/"
    for folder in ["image_steps", "mask_steps", "depth_steps", "shading_steps",
                   "TRACER_steps", "MiDaS_steps", "Intrinsic_steps"]:
        folder_dir = os.path.join(data_path, folder)
        if not os.path.exists(folder_dir):
            os.makedirs(folder_dir)


    # Save the intermediate output of LDM self-attention layers
    # They are used as the input to the probing classifiers
    features = OrderedDict()

    # recursive hooking function
    for name, module in unet.named_modules():
        if isinstance(module, torch.nn.Linear):
            features[name] = ModuleHook(module)

    # Regenerate the images in probing dataset
    print(f'Began generating the images for the prompt "{prompt}" using Stable Diffusion')
    image_dict = generate_image_modified(prompt, seed_num, num_inference_steps=15,
                           net=unet, tokenizer=tokenizer, text_encoder=text_encoder, scheduler=scheduler, vae=vae,
                           stop_at_step=15)



    for j, image in image_dict.items():
        image_name = f'{image_index}_step_{j}.png'

        # Save each intermediate image as png
        plt.imsave(os.path.join(data_path, 'image_steps', image_name), image)

        # Apply TRACER model to get mask


        # Apply MiDaS model to get depth
        image_for_MiDaS = plt.imread(os.path.join(data_path, 'image_steps', image_name))[...,:3]
        if image_for_MiDaS.max() <= 1:
            image_for_MiDaS *= 255
            image_for_MiDaS = image_for_MiDaS.astype("uint8")

        input_batch = transform(image_for_MiDaS).to(torch_device)

        with torch.no_grad():
            MiDaS_result = midas(input_batch)
            MiDaS_result = torch.nn.functional.interpolate(
                MiDaS_result.unsqueeze(1),
                size=image_for_MiDaS.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze().cpu().detach()

        plt.imsave(os.path.join(data_path, 'MiDaS_steps', image_name), MiDaS_result, cmap='turbo')

        # Apply Intrinsic model to get shading/illumination
        image_for_Intrinsic = load_image(os.path.join(data_path, 'image_steps', image_name))[:,:,:3]

        Intrinsic_result = run_pipeline(
            intrinsic_model,
            image_for_Intrinsic,
            resize_conf=0.0,
            maintain_size=True,
            linear=False,
            device=torch_device
        )

        shd = uninvert(Intrinsic_result['inv_shading'])
        plt.imsave(os.path.join(data_path, 'Intrinsic_steps', image_name), view(shd))


    print("Finished saving the images at every denoising step")    

    
    
    
    # Using the probe checkpoints to produce intermediate images for mask, depth, shading
    for at_step in range(15):
        input_dim = input_dims_dict[f"{block}_{block_ind}"]
        scale = scale_dict[f"{block}_{block_ind}"]
        layer = f"{block}_{block_ind}_{output_dir_name}_{layer_ind}"

        if block == "mid":
            chosen_layer_name = f"mid_block.{block_type}.{layer_ind}.{layer_name}"
        else:
            chosen_layer_name = f"{block}_blocks.{block_ind}.{block_type}.{layer_ind}.{layer_name}"

        internal_repres = features[chosen_layer_name].features[at_step].to(torch.float)

        # Salient object detection (mask)
        probe = probeLinearDense(input_dim, 2, scale, use_bias=False).to(torch_device)
        full_probe_name = f"segmentation_probe_{layer}_final.pth" # segmentation_probe_up_3_attn1_out_0_final.pth
        weights_path = f"probe_checkpoints/large_syn_dataset/at_step_{at_step}/{full_probe_name}"
        probe.load_state_dict(torch.load(weights_path))

        # Save the probe mask result
        with torch.no_grad():
            output = torch.argmax(probe(internal_repres).cpu().detach()[0], dim=0)
        plt.imsave(os.path.join(data_path, 'mask_steps', f'{image_index}_step_{at_step}.png'), output, cmap='gray')


        # Depth
        probe = probeLinearDense(input_dim, 1, scale, use_bias=False).to(torch_device)
        full_probe_name = f"regression_probe_{layer}_final_linear_no_bias_unsmoothed.pth" #regression_probe_up_3_attn1_out_0_final_linear_no_bias_unsmoothed.pth
        weights_path = f"probe_checkpoints/large_syn_dataset_continuous/at_step_{at_step}/{full_probe_name}"
        probe.load_state_dict(torch.load(weights_path))

        # Save the probe depth result
        with torch.no_grad():
            output = probe(internal_repres).cpu().detach()[0].squeeze(0)
        plt.imsave(os.path.join(data_path, 'depth_steps', f'{image_index}_step_{at_step}.png'), output, cmap='turbo')    

        torch.cuda.empty_cache() # Does not do anything
        print(f"End of step {at_step}: {get_gpu_memory()[0]} MiB CUDA memory left")


        # Shading
        probe = probeLinearDense(input_dim, 1, scale, use_bias=False).to(torch_device)
        full_probe_name = f"regression_probe_{layer}_final_linear_no_bias_unsmoothed_lr0.01_epochs30_500_training_images.pth"
        weights_path = f"probe_checkpoints/large_syn_dataset_shading/at_step_{at_step}/{full_probe_name}"
        probe.load_state_dict(torch.load(weights_path))

        # Save the probe depth result
        with torch.no_grad():
            output = probe(internal_repres).cpu().detach()[0].squeeze(0)
        plt.imsave(os.path.join(data_path, 'shading_steps', f'{image_index}_step_{at_step}.png'), output) #, cmap='gray'    

        torch.cuda.empty_cache() # Does not do anything
        print(f"End of step {at_step}: {get_gpu_memory()[0]} MiB CUDA memory left")

    print(f'Finished for the prompt "{prompt}"')
    
    
    
    plot_image_steps(image_index, compact=False)
    plot_image_steps(image_index, compact=True)
    
    
    
    
    
    
    
    
if __name__ == '__main__':
    main(args)  
    