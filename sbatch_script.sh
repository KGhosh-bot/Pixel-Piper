#!/bin/bash
# sbatch_script.sh
# Add debug logging
set -x  # Print commands and their arguments as they are executed

# Example parameter sets
style_name="Comic_book"
general_prompt="A brown dog"
negative_prompt="naked, deformed, bad anatomy, disfigured, poorly drawn face, mutation, extra limb, ugly, disgusting, poorly drawn hands, missing limb, floating limbs, disconnected limbs, blurry, watermarks, oversaturated, distorted hands, amputation"
prompt_array=("and a yellow bowl" "on bed" "with a ball" "wearing a hat" "playing with a white cat")

# Print parameters before passing
echo "Parameters being passed:"
echo "style_name: $style_name"
echo "general_prompt: $general_prompt"
echo "negative_prompt: $negative_prompt"
echo "prompt_array: ${prompt_array[@]}"


# Convert prompt_array to a comma-separated string
# prompt_array_str=$(IFS=,; echo "${prompt_array[*]}")
prompt_array_str=$(printf "%s|" "${prompt_array[@]}")
prompt_array_str="${prompt_array_str%|}"  # Remove trailing "|"

# Run with parameter set
sbatch -w cloudifaicdlw001-System-Product-Name -N 1 --gpus=nvidia_geforce_rtx_3090:1 run_docker.sh "$style_name" "$general_prompt" "$negative_prompt" "$prompt_array_str"