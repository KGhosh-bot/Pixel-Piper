#!/bin/bash

set -x  # Enable debug mode

# Print received parameters
echo "train.sh received parameters:"
echo "1: $1"
echo "2: $2"
echo "3: $3"
echo "5: ${4}"


# flags="--style_name $1 --seed $2 --general_prompt $3 --negative_prompt $4 --prompt_array $5"
# flags="--style_name \"$1\" --seed $2 --general_prompt \"$3\" --negative_prompt \"$4\""

# for prompt in "${@:5}"; do
#     flags="$flags --prompt_array \"$prompt\""
# done
# echo "Constructed flags: $flags"

# python3 script.py $flags
flags=(--style_name "$1" --general_prompt "$2" --negative_prompt "$3")

prompt_args=()
for prompt in "${@:4}"; do
    prompt_args+=("$prompt")
done

echo "Constructed flags: ${flags[*]} --prompt_array ${prompt_args[*]}"

python3 run.py "${flags[@]}" --prompt_array "${prompt_args[@]}"
