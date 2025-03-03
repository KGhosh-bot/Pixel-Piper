# config_parser.py
import argparse
from pathlib import Path

def parse_arguments():
    """ Parses command-line arguments for the Stable Diffusion script. """
    parser = argparse.ArgumentParser(description="Run Stable Diffusion with AttendExcite and Spatial Attention Processors.")

    # Required arguments
    # parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation")

    # arguments with defaults for predicate logic
    parser.add_argument("--neg_prompt", type=str, default=None, help="The negative prompt for the image generation")
    parser.add_argument("--sd_2_1", type=bool, default=False, help="Whether to use Stable Diffusion v2.1")
    parser.add_argument("--token_indices", nargs='+', type=int, default=[1, 2, 3, 4], help="Which token indices to alter with attend-and-excite")
    parser.add_argument("--seeds", nargs='+', type=int, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], help="Which random seeds to use when generating")
    parser.add_argument("--output_path", type=Path, default=Path('./outputs'), help="Path to save all outputs to")
    parser.add_argument("--n_inference_steps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Text guidance scale")
    parser.add_argument("--max_iter_to_alter", type=int, default=25, help="Number of denoising steps to apply attend-and-excite")
    parser.add_argument("--attention_res", type=int, default=32, help="Resolution of UNet to compute attention maps over")
    parser.add_argument("--run_standard_sd", type=bool, default=False, help="Whether to run standard SD or attend-and-excite")
    parser.add_argument("--run_attention_sd", type=bool, default=True, help="Whether to run Attention SD or Vanilla SD")
    parser.add_argument("--thresholds", type=lambda kv: dict(map(lambda x: (int(x.split(":")[0]), float(x.split(":")[1])), kv.split(","))), default={0: 0}, help="Dictionary defining the iterations and desired thresholds to apply iterative latent refinement in")
    parser.add_argument("--scale_factor", type=int, default=20, help="Scale factor for updating the denoised latent z_t")
    parser.add_argument("--scale_range", type=lambda x: tuple(map(float, x.split(","))), default=(1.0, 0.5), help="Start and end values used for scaling the scale factor - decays linearly with the denoising timestep")
    parser.add_argument("--smooth_attentions", type=bool, default=True, help="Whether to apply the Gaussian smoothing before computing the maximum attention value for each subject token")
    parser.add_argument("--sigma", type=float, default=5, help="Standard deviation for the Gaussian smoothing")
    parser.add_argument("--kernel_size", type=int, default=3, help="Kernel size for the Gaussian smoothing")
    parser.add_argument("--save_cross_attention_maps", type=bool, default=False, help="Whether to save cross attention maps for the final results")
    parser.add_argument("--attention_corr_indices", nargs='+', type=list, default=[
    [[6,7,8,9], [14,15,16,17]],  
    [[6,7,8,9]],               
    [[6,7,8,9]],                
    [[6,7,8,9]],               
    [[6,7,8,9], [16,17,18,19]]], help="Index of corr inclusion pair")
    parser.add_argument("--attention_exist_indices", nargs='+', type=list, default=[
    [8,9,16,17], 
    [8,9,12,13],  
    [8,9,14,15],  
    [8,9,14,15], 
    [8,9,18,19] ], help="Index of exist inclusion pair")
    parser.add_argument("--attention_leak_indices", nargs='+', type=list, default=[
    [[6,7,16,17], [14,15,8,9]], 
    [],                
    [],                
    [],                
    [[6,7,18,19], [16,17,8,9]]], help="Index of leak inclusion pair")
    parser.add_argument("--attention_possession_indices", nargs='+', type=list, default=[
    [],      
    [12,16,8,9],       
    [8,9,14,15],       
    [8,9,14,15],   
    []], help="Index of semantic inclusion pair")
    parser.add_argument("--attention_save_t", nargs='+', type=int, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50], help="Index of attention maps for save")
    parser.add_argument("--loss_function", type=str, default="attention_product_prob", help="Loss function to use")
    parser.add_argument("--mode", type=str, default="practice", help="Mode of operation")

    # Define command line arguments for masked attention
    # parser = argparse.ArgumentParser(description="Generate Consistent Images")
    parser.add_argument("--model_name", type=str, default="RealVision", help="Model name from models_dict")
    parser.add_argument("--style_name", type=str, default="Comic_book", help="Style name from styles")
    # parser.add_argument("--seed", type=int, default=2047, help="Random seed")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    # parser.add_argument("--num_steps", type=int, default=50, help="Number of inference steps")
    # parser.add_argument("--guidance_scale", type=float, default=5.0, help="Guidance scale")
    parser.add_argument("--general_prompt", type=str, default="a man with a black suit", help="General prompt")
    parser.add_argument("--negative_prompt", type=str, default="naked, deformed, bad anatomy, disfigured, poorly drawn face, mutation, extra limb, ugly, disgusting, poorly drawn hands, missing limb, floating limbs, disconnected limbs, blurry, watermarks, oversaturated, distorted hands, amputation", help="Negative prompt")
    parser.add_argument("--prompt_array", nargs='+', default=["wake up in the bed", "have breakfast", "is on the road, go to the company", "work in the company", "running in the playground", "reading book in the home"], help="Array of prompts")
    parser.add_argument("--id_length", type=int, default=4, help="Length of identity prompts")
    parser.add_argument("--sa32", type=float, default=0.5, help="Strength of consistent self-attention for 32x32")
    parser.add_argument("--sa64", type=float, default=0.5, help="Strength of consistent self-attention for 64x64")

    return parser.parse_args()
