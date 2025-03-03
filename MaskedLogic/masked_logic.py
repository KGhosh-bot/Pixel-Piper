import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.nn import functional as F
import wandb


from diffusers.utils import (
    logging,
)

from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from diffusers import StableDiffusionXLPipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import retrieve_timesteps
from utils import AttentionStore, aggregate_attention




logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class PredicatedDiffPipeline(StableDiffusionXLPipeline):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion.
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """
    _optional_components = ["safety_checker", "feature_extractor"]


    def _encode_prompt(
            self,
            prompt: str,
            prompt_2: Optional[str] = None,
            device: Optional[torch.device] = None,
            num_images_per_prompt: int = 1,
            do_classifier_free_guidance: bool = True,
            negative_prompt: Optional[str] = None,
            negative_prompt_2: Optional[str] = None,
            prompt_embeds: Optional[torch.Tensor] = None,
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            pooled_prompt_embeds: Optional[torch.Tensor] = None,
            negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
            original_size: Optional[Tuple[int, int]] = (1024, 1024),
            crops_coords_top_left: Tuple[int, int] = (0, 0),
            target_size: Optional[Tuple[int, int]] = (1024, 1024),
            negative_original_size: Optional[Tuple[int, int]] = (1024, 1024),
            negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
            negative_target_size: Optional[Tuple[int, int]] = (1024, 1024),
        ):
            r"""
            Encodes the prompt into text encoder hidden states.

            Args:
                prompt (`str` or `List[str]`, *optional*):
                    prompt to be encoded
                prompt_2 (`str` or `List[str]`, *optional*):
                    The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                    used in both text-encoders
                device: (`torch.device`):
                    torch device
                num_images_per_prompt (`int`):
                    number of images that should be generated per prompt
                do_classifier_free_guidance (`bool`):
                    whether to use classifier free guidance or not
                negative_prompt (`str` or `List[str]`, *optional*):
                    The prompt or prompts not to guide the image generation. If not defined, one has to pass
                    `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                    less than `1`).
                negative_prompt_2 (`str` or `List[str]`, *optional*):
                    The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                    `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
                prompt_embeds (`torch.Tensor`, *optional*):
                    Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                    provided, text embeddings will be generated from `prompt` input argument.
                negative_prompt_embeds (`torch.Tensor`, *optional*):
                    Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                    weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                    argument.
                pooled_prompt_embeds (`torch.Tensor`, *optional*):
                    Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                    If not provided, pooled text embeddings will be generated from `prompt` input argument.
                negative_pooled_prompt_embeds (`torch.Tensor`, *optional*):
                    Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                    weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                    input argument.
                lora_scale (`float`, *optional*):
                    A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
                clip_skip (`int`, *optional*):
                    Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                    the output of the pre-final layer will be used for computing the prompt embeddings.
            """
            device = device or self._execution_device

            
            prompt = [prompt] if isinstance(prompt, str) else prompt

            if prompt is not None:
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]

            # Define tokenizers and text encoders
            tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
            text_encoders = (
                [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
            )

            if prompt_embeds is None:
                prompt_2 = prompt_2 or prompt
                prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

                # textual inversion: process multi-vector tokens if necessary
                prompt_embeds_list = []
                prompts = [prompt, prompt_2]
                for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
                    
                    text_inputs = tokenizer(
                        prompt,
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    )

                    text_input_ids = text_inputs.input_ids
                    untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

                    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                        text_input_ids, untruncated_ids
                    ):
                        removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
                        logger.warning(
                            "The following part of your input was truncated because CLIP can only handle sequences up to"
                            f" {tokenizer.model_max_length} tokens: {removed_text}"
                        )

                    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

                    # We are only ALWAYS interested in the pooled output of the final text encoder
                    pooled_prompt_embeds = prompt_embeds[0]

                    prompt_embeds = prompt_embeds.hidden_states[-2]
                    
                    prompt_embeds_list.append(prompt_embeds)

                prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

            # get unconditional embeddings for classifier free guidance
            zero_out_negative_prompt = negative_prompt is None and self.config.force_zeros_for_empty_prompt
            if do_classifier_free_guidance and negative_prompt_embeds is None and zero_out_negative_prompt:
                negative_prompt_embeds = torch.zeros_like(prompt_embeds)
                negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
            elif do_classifier_free_guidance and negative_prompt_embeds is None:
                negative_prompt = negative_prompt or ""
                negative_prompt_2 = negative_prompt_2 or negative_prompt

                # normalize str to list
                negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
                negative_prompt_2 = (
                    batch_size * [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2
                )

                uncond_tokens: List[str]
                if prompt is not None and type(prompt) is not type(negative_prompt):
                    raise TypeError(
                        f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                        f" {type(prompt)}."
                    )
                elif batch_size != len(negative_prompt):
                    raise ValueError(
                        f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                        f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                        " the batch size of `prompt`."
                    )
                else:
                    uncond_tokens = [negative_prompt, negative_prompt_2]

                negative_prompt_embeds_list = []
                for negative_prompt, tokenizer, text_encoder in zip(uncond_tokens, tokenizers, text_encoders):
                    
                    max_length = prompt_embeds.shape[1]
                    uncond_input = tokenizer(
                        negative_prompt,
                        padding="max_length",
                        max_length=max_length,
                        truncation=True,
                        return_tensors="pt",
                    )

                    negative_prompt_embeds = text_encoder(
                        uncond_input.input_ids.to(device),
                        output_hidden_states=True,
                    )
                    # We are only ALWAYS interested in the pooled output of the final text encoder
                    negative_pooled_prompt_embeds = negative_prompt_embeds[0]
                    negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]

                    negative_prompt_embeds_list.append(negative_prompt_embeds)

                negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)

            if self.text_encoder_2 is not None:
                prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
            else:
                prompt_embeds = prompt_embeds.to(dtype=self.unet.dtype, device=device)

            bs_embed, seq_len, _ = prompt_embeds.shape
            # duplicate text embeddings for each generation per prompt, using mps friendly method
            prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

            if do_classifier_free_guidance:
                # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
                seq_len = negative_prompt_embeds.shape[1]

                if self.text_encoder_2 is not None:
                    negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
                else:
                    negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.unet.dtype, device=device)

                negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
                negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
                bs_embed * num_images_per_prompt, -1
            )
            if do_classifier_free_guidance:
                negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
                    bs_embed * num_images_per_prompt, -1
                )

            # Prepare added time ids & embeddings
            add_text_embeds = pooled_prompt_embeds
            if self.text_encoder_2 is None:
                text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
            else:
                text_encoder_projection_dim = self.text_encoder_2.config.projection_dim
                print("text_encoder_projection_dim:", text_encoder_projection_dim)
            add_time_ids = self._get_add_time_ids(
                original_size,
                crops_coords_top_left,
                target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
            print("1.add_time_ids:", add_time_ids.shape)
            if negative_original_size is not None and negative_target_size is not None:
                negative_add_time_ids = self._get_add_time_ids(
                    negative_original_size,
                    negative_crops_coords_top_left,
                    negative_target_size,
                    dtype=prompt_embeds.dtype,
                    text_encoder_projection_dim=text_encoder_projection_dim,
                )
            else:
                negative_add_time_ids = add_time_ids

            if do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
                add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

            print("2.add_time_ids:", add_time_ids.shape)
            
            return prompt_embeds, add_text_embeds, add_time_ids

    
    def _aggregate_attention_per_token(self, attention_store: AttentionStore, attention_res: int = 32):
        """
        Aggregates attention across multiple layers for SDXL.
        SDXL has a deeper U-Net and requires aggregation from more attention heads.

        :param attention_store: Stores attention maps.
        :param attention_res: Target attention resolution (32x32 or 64x64 for SDXL).
        :return: Aggregated attention maps.
        """
        attention_maps = aggregate_attention(
            attention_store=attention_store,
            res=attention_res,
            from_where=("up", "down", "mid"),  # Keep structure but adapt to SDXL layers
            is_cross=True,
            select=0
        )
        return attention_maps

    @staticmethod
    def _update_latent(latents: torch.Tensor, loss: torch.Tensor, step_size: float, scaling_factor=1.0) -> torch.Tensor:
        """
        Updates the latent space based on computed gradients.
        SDXL requires a larger scaling factor due to its different latent space.

        :param latents: Latent tensor.
        :param loss: Computed loss.
        :param step_size: Step size for updating latents.
        :param scaling_factor: Adjusted scaling factor for SDXL.
        :return: Updated latents.
        """
        grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents], retain_graph=True)[0]

        # Adjust step size scaling for SDXL
        latents = latents - (scaling_factor * step_size * grad_cond)
        return latents

    @staticmethod
    def _compute_loss_attention(attention_maps: torch.Tensor, attention_corr_indices: List[List[int]], size: int) -> torch.Tensor:
        """
        Computes the attention-based loss, ensuring the model focuses on correct tokens.

        :param attention_maps: Attention tensor from the model.
        :param attention_corr_indices: List of index pairs indicating semantic relationships.
        :param size: Normalization size.
        :return: Computed loss.
        """
        loss = 0

        for ind in attention_corr_indices:
            attenB = attention_maps[:, :, ind[0]]  # Modifier token
            attenA = attention_maps[:, :, ind[1]]  # Main token

            # SDXL has deeper attention layers, requiring loss adjustments
            loss_value = (1 - attenA * (1 - attenB))  
            loss_value = torch.sum(loss_value) / size

            loss += loss_value
            print(f"Loss value: {loss_value}")

        return -loss  # Negative sign ensures optimization improves focus
    
    @staticmethod
    def _compute_loss_attention_product_prob(
        attention_maps: torch.Tensor,
        attention_corr_indices: List[List[int]],
        attention_leak_indices: List[List[int]],
        attention_exist_indices: List[int],
        attention_possession_indices: List[List[int]],
        size: int,
        is_sdxl: bool = True
    ) -> torch.Tensor:
        """
        Computes the attend-and-excite loss using the maximum attention value for each token.
        This implementation is adapted for SDXL (128×128 attention maps, multi-scale aggregation, dual text encoders).
        
        :param attention_maps: The computed attention maps from SDXL.
        :param attention_corr_indices: List of attention indices for correspondence loss.
        :param attention_leak_indices: List of indices for leak loss.
        :param attention_exist_indices: List of indices for exist loss.
        :param attention_possession_indices: List of indices for possession loss.
        :param size: Normalization factor for loss computation.
        :param is_sdxl: Boolean flag for SDXL-specific modifications.
        :return: Total loss value.
        """

        loss = 0
        eps = 1e-20  # Small epsilon for numerical stability

        ###########################
        # 1️ Normalize Attention Maps
        ###########################
        last_idx = -1 if not is_sdxl else -2  # Adjust token index range for SDXL
        attention_normalize = attention_maps[:, :, 1:last_idx].clone()
        attention_normalize *= 100  # Scaling factor for softmax stability
        attention_normalize = torch.nn.functional.softmax(attention_normalize, dim=-1)

        ###########################
        # 2️ Compute Existence Loss
        ###########################
        for exist_ind in attention_exist_indices:
            exist_ind = exist_ind * 2 if is_sdxl else exist_ind  # Adjust token index if using SDXL dual encoders
            atten_obj_exist = attention_normalize[:, :, exist_ind - 1]

            loss_attend_value = torch.sum(torch.log(1 - atten_obj_exist + eps)) / size
            loss_attend_value = -torch.log(1 - torch.exp(loss_attend_value))

            loss += 0.1 * loss_attend_value / len(attention_exist_indices)

        ###########################
        # 3️ Compute Correspondence Loss
        ###########################
        for corr_ind in attention_corr_indices:
            atten_adj = attention_maps[:, :, corr_ind[0]]
            atten_obj = attention_maps[:, :, corr_ind[1]]

            atten_adj = (atten_adj - atten_adj.min()) / (atten_adj.max() - atten_adj.min() + eps)
            atten_obj = (atten_obj - atten_obj.min()) / (atten_obj.max() - atten_obj.min() + eps)

            loss_corr_value = -torch.log(1 - (atten_obj * (1 - atten_adj)) + eps) - torch.log(1 - (atten_adj * (1 - atten_obj)) + eps)
            loss_corr_value = torch.sum(loss_corr_value) / size

            loss += loss_corr_value / len(attention_corr_indices)

        ###########################
        # 4️ Compute Leak Loss
        ###########################
        for leak_ind in attention_leak_indices:
            atten_adj = attention_maps[:, :, leak_ind[0]]
            atten_obj = attention_maps[:, :, leak_ind[1]]

            atten_adj = (atten_adj - atten_adj.min()) / (atten_adj.max() - atten_adj.min() + eps)
            atten_obj = (atten_obj - atten_obj.min()) / (atten_obj.max() - atten_obj.min() + eps)

            loss_leak_value = -torch.log(1 - (atten_obj * atten_adj) + eps)
            loss_leak_value = torch.sum(loss_leak_value) / size

            loss += 0.3 * loss_leak_value / len(attention_leak_indices)

        ###########################
        # 5️ Compute Possession Loss
        ###########################
        for possession_ind in attention_possession_indices:
            atten_holder = attention_maps[:, :, possession_ind[0]]
            atten_possession = attention_maps[:, :, possession_ind[1]]

            atten_possession = (atten_possession - atten_possession.min()) / (atten_possession.max() - atten_possession.min() + eps)
            atten_holder = (atten_holder - atten_holder.min()) / (atten_holder.max() - atten_holder.min() + eps)

            loss_possession_value = -torch.log(1 - (atten_possession * (1 - atten_holder)) + eps)
            loss_possession_value = torch.sum(loss_possession_value) / size

            loss += loss_possession_value / len(attention_possession_indices)

        return loss
    
    @staticmethod
    def _compute_loss_attention_exist(
        attention_maps: torch.Tensor,
        attention_corr_indices: List[List[int]],
        size: int,
        is_sdxl: bool = True
    ) -> torch.Tensor:
        """
        Computes the Attend-and-Excite loss using the maximum attention value for each token.
        This version is modified for SDXL by adjusting token indices and handling multi-scale attention.

        :param attention_maps: The computed attention maps.
        :param attention_corr_indices: List of attention indices for correspondence loss.
        :param size: Normalization factor for loss computation.
        :param is_sdxl: Boolean flag for SDXL adjustments.
        :return: Computed loss and relevant attention maps.
        """
        loss = 0
        eps = 1e-8  # Small epsilon to prevent log(0)

        for ind in attention_corr_indices:
            last_idx = -1 if not is_sdxl else -2  # Adjust indexing for SDXL

            # Normalize attention maps across token indices
            attention_normalize = attention_maps[:, :, 1:last_idx].clone()
            attention_normalize *= 100  # Scaling factor for softmax stability
            attention_maps[:, :, 1:last_idx] = torch.nn.functional.softmax(attention_normalize, dim=-1)

            # Adjust token indices for SDXL (2*i, 2*i+1)
            idxB = [ind[1] * 2, ind[1] * 2 + 1] if is_sdxl else [ind[1]]
            idxD = [ind[3] * 2, ind[3] * 2 + 1] if is_sdxl else [ind[3]]

            attenB = attention_maps[:, :, idxB].mean(dim=-1)  # Aggregate SDXL tokens
            attenD = attention_maps[:, :, idxD].mean(dim=-1)  

            print(f"Max attenB: {attenB.max()}, Min attenB: {attenB.min()}")
            print(f"Max attenD: {attenD.max()}, Min attenD: {attenD.min()}")

            # Compute loss using negative log-exponential summation
            loss_value_noun = -torch.log(1 - torch.exp(torch.sum(torch.log(1 - attenB + eps)) / size)) - \
                            torch.log(1 - torch.exp(torch.sum(torch.log(1 - attenD + eps)) / size))

            loss += loss_value_noun
            print(f"Loss Value (Noun): {loss_value_noun}")

        print(f"Final Loss: {loss}")
        attention_for_obj = [attenB, attenD]
        return loss, attention_for_obj

    @staticmethod
    def _compute_loss_attention_negation(
        attention_maps: torch.Tensor,
        attention_corr_indices: List[List[int]],
        size: int,
        is_sdxl: bool = True
    ) -> torch.Tensor:
        """
        Computes the Attend-and-Excite loss for negation constraints.
        This version is modified for SDXL by adjusting token indices and handling multi-scale attention.

        :param attention_maps: The computed attention maps.
        :param attention_corr_indices: List of attention indices for correspondence loss.
        :param size: Normalization factor for loss computation.
        :param is_sdxl: Boolean flag for SDXL adjustments.
        :return: Computed loss and relevant attention maps.
        """
        loss = 0
        eps = 1e-8  # Small epsilon to prevent log(0)

        for ind in attention_corr_indices:
            last_idx = -1 if not is_sdxl else -2  # Adjust indexing for SDXL

            # Normalize attention maps across token indices
            attention_normalize = attention_maps[:, :, 1:last_idx].clone()
            attention_normalize *= 100  # Scaling factor for softmax stability
            attention_maps[:, :, 1:last_idx] = torch.nn.functional.softmax(attention_normalize, dim=-1)

            # Adjust token indices for SDXL (2*i, 2*i+1)
            idxA = [1 * 2, 1 * 2 + 1] if is_sdxl else [1]

            attenA = attention_maps[:, :, idxA].mean(dim=-1)  # Aggregate SDXL tokens

            print(f"Max attenA: {attenA.max()}, Min attenA: {attenA.min()}")

            ########### Compute Negation Loss ###########
            loss_neg = -torch.sum(torch.log(1 - attenA + eps)) / size
            loss_neg *= 0.1  # Scale loss

            loss += loss_neg
            print(f"Loss Value (Negation): {loss_neg}")

        print(f"Final Loss: {loss}")
        attention_for_obj = [attenA]
        return loss, attention_for_obj


    def _perform_iterative_refinement_fixed_step_att(
        self,
        latents: torch.Tensor,
        loss: torch.Tensor,
        threshold: float,
        batch_size: int, 
        text_embeddings: torch.Tensor,
        add_text_embeds: torch.Tensor,
        add_time_ids: torch.Tensor,
        attention_corr_indices: List[List[int]],
        attention_leak_indices: List[List[int]],
        attention_exist_indices: List[int],
        attention_possession_indices: List[List[int]],
        attention_store: AttentionStore,
        step_size: float,
        t: int,
        attention_res: int = 32,  # Adjusted for SDXL (default: 64)
        max_refinement_steps: int = 20,
        attention_size: int = 512,  # Adjusted for SDXL
        attention_loss_function: str = "attention_exist",
        is_sdxl: bool = True  # Flag to determine SDXL adjustments
    ):
        """
        Performs iterative latent refinement. Adapted for SDXL by handling multi-resolution attention maps
        and dual text encoders.
        
        :param latents: The latent tensor.
        :param loss: Initial loss value.
        :param threshold: Threshold for refinement.
        :param text_embeddings: Encoded text embeddings.
        :param attention_corr_indices: Token pairs for correspondence loss.
        :param attention_leak_indices: Token pairs for leak loss.
        :param attention_exist_indices: Tokens for existence loss.
        :param attention_possession_indices: Token pairs for possession loss.
        :param attention_store: Stores attention values.
        :param step_size: Learning rate for latent updates.
        :param t: Current denoising timestep.
        :param attention_res: Resolution of attention maps (Adjusted for SDXL).
        :param max_refinement_steps: Maximum number of latent refinement iterations.
        :param attention_size: Size of attention map for normalization (Adjusted for SDXL).
        :param attention_loss_function: Which loss function to use.
        :param is_sdxl: Whether we are running in SDXL mode.
        :return: Final loss and refined latents.
        """

        for iteration in range(5):
            latents = latents.clone().detach().requires_grad_(True)

            # Adjust text encoder selection for SDXL
            encoder_hidden_states = text_embeddings[batch_size:]
            added_cond_neg_kwargs = {"text_embeds": add_text_embeds[batch_size:], "time_ids": add_time_ids[1::2]}
            noise_pred_text = self.unet(latents, t, encoder_hidden_states=encoder_hidden_states, cross_attention_kwargs=self.cross_attention_kwargs,
                                    added_cond_kwargs=added_cond_neg_kwargs,
                                    return_dict=False, 
                                    )[0]
            self.unet.zero_grad()

            # Aggregate attention maps at multiple resolutions for SDXL
            attention_maps = self._aggregate_attention_per_token(
                attention_store=attention_store,
                attention_res=attention_res,
                is_sdxl=is_sdxl
            )

            # Choose loss function based on SDXL compatibility
            if attention_loss_function == "attention_product_prob":
                loss, attention_for_obj = self._compute_loss_attention_product_prob(
                    attention_maps=attention_maps,
                    size=attention_size,
                    attention_corr_indices=attention_corr_indices,
                    attention_leak_indices=attention_leak_indices,
                    attention_exist_indices=attention_exist_indices,
                    attention_possession_indices=attention_possession_indices,
                    is_sdxl=is_sdxl
                )
            elif attention_loss_function == "attention_exist":
                loss, attention_for_obj = self._compute_loss_attention_exist(
                    attention_maps=attention_maps,
                    attention_corr_indices=attention_corr_indices,
                    size=attention_size,
                    is_sdxl=is_sdxl
                )
            elif attention_loss_function == "attention_negation":
                loss, attention_for_obj = self._compute_loss_attention_negation(
                    attention_maps=attention_maps,
                    attention_corr_indices=attention_corr_indices,
                    size=attention_size,
                    is_sdxl=is_sdxl
                )

            # Update latents if loss is non-zero
            if loss != 0:
                latents = self._update_latent(latents, loss, step_size, is_sdxl=is_sdxl)

            # Run denoising step
            with torch.no_grad():

                added_uncond_neg_kwargs = {"text_embeds": add_text_embeds[:batch_size], "time_ids": add_time_ids[0::2]}
                noise_pred_uncond = self.unet(latents, t, encoder_hidden_states=text_embeddings[:batch_size], cross_attention_kwargs=self.cross_attention_kwargs,
                                    added_cond_kwargs=added_uncond_neg_kwargs,
                                    return_dict=False,
                                    )[0]

                added_cond_neg_kwargs = {"text_embeds": add_text_embeds[batch_size:], "time_ids": add_time_ids[1::2]}
                noise_pred_text = self.unet(latents, t, encoder_hidden_states=text_embeddings[batch_size:], cross_attention_kwargs=self.cross_attention_kwargs,
                                    added_cond_kwargs=added_cond_neg_kwargs,
                                    return_dict=False,
                                    )[0]

            print(f"iter{iteration}_loss_{loss}")

        # Final refinement step without updating latents
        latents = latents.clone().detach().requires_grad_(True)

        added_cond_neg_kwargs = {"text_embeds": add_text_embeds[batch_size:], "time_ids": add_time_ids[1::2]}
        noise_pred_text = self.unet(latents, t, encoder_hidden_states=text_embeddings[batch_size:], cross_attention_kwargs=self.cross_attention_kwargs,
                                added_cond_kwargs=added_cond_neg_kwargs,
                                return_dict=False,
                                )[0]
        self.unet.zero_grad()

        # Aggregate attention maps one last time
        attention_maps = self._aggregate_attention_per_token(
            attention_store=attention_store,
            attention_res=attention_res,
            is_sdxl=is_sdxl
        )

        # Compute final loss
        if attention_loss_function == "attention_product_prob":
            loss, attention_for_obj = self._compute_loss_attention_product_prob(
                attention_maps=attention_maps,
                size=attention_size,
                attention_corr_indices=attention_corr_indices,
                attention_leak_indices=attention_leak_indices,
                attention_exist_indices=attention_exist_indices,
                attention_possession_indices=attention_possession_indices,
                is_sdxl=is_sdxl
            )
        elif attention_loss_function == "attention_exist":
            loss, attention_for_obj = self._compute_loss_attention_exist(
                attention_maps=attention_maps,
                attention_corr_indices=attention_corr_indices,
                size=attention_size,
                is_sdxl=is_sdxl
            )
        elif attention_loss_function == "attention_negation":
            loss, attention_for_obj = self._compute_loss_attention_negation(
                attention_maps=attention_maps,
                attention_corr_indices=attention_corr_indices,
                size=attention_size,
                is_sdxl=is_sdxl
            )

        print(f"\t Finished with loss of: {loss}")
        return loss, latents

    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]],
            neg_prompt: Union[str, List[str]],
            attention_store: AttentionStore,
            indices_to_alter: List[int],
            attention_res: int = 32,
            prompt_2: Optional[Union[str, List[str]]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            timesteps: List[int] = None,
            denoising_end: Optional[float] = None,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            negative_prompt_2: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            pooled_prompt_embeds: Optional[torch.Tensor] = None,
            negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: Optional[int] = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            original_size: Optional[Tuple[int, int]] = None,
            crops_coords_top_left: Tuple[int, int] = (0, 0),
            target_size: Optional[Tuple[int, int]] = None,
            negative_original_size: Optional[Tuple[int, int]] = None,
            negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
            negative_target_size: Optional[Tuple[int, int]] = None,
            max_iter_to_alter: Optional[int] = 25,
            run_standard_sd: bool = False,
            run_attention_sd: bool = True, 
            thresholds: Optional[dict] = {0: 0.99, 5: 0.5, 20: 0.8},
            scale_factor: int = 20,
            scale_range: Tuple[float, float] = (1., 0.8),
            smooth_attentions: bool = True,
            sigma: float = 0.5,
            kernel_size: int = 3,
            sd_2_1: bool = False,
            is_sdxl: bool = True,
            attention_corr_indices: List[List[int]] = None,
            attention_leak_indices: List[List[int]] = None,
            attention_exist_indices: List[int] = None,
            attention_save_t: List[int] = None,
            attention_possession_indices: List[List[int]] = None,
            loss_function: str = "attention",
            
            
    ):
        """
        Function invoked when calling the pipeline for generation.
        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in both text-encoders
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf) `guidance_scale` is defined as `φ` in equation 16. of
                [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
                Guidance rescale factor should fix overexposure when using zero terminal SNR.
            original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                If `original_size` is not the same as `target_size` the image will appear to be down- or upsampled.
                `original_size` defaults to `(height, width)` if not specified. Part of SDXL's micro-conditioning as
                explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                `crops_coords_top_left` can be used to generate an image that appears to be "cropped" from the position
                `crops_coords_top_left` downwards. Favorable, well-centered images are usually achieved by setting
                `crops_coords_top_left` to (0, 0). Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                For most cases, `target_size` should be set to the desired height and width of the generated image. If
                not specified it will default to `(height, width)`. Part of SDXL's micro-conditioning as explained in
                section 2.2 of [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            negative_original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                To negatively condition the generation process based on a specific image resolution. Part of SDXL's
                micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            negative_crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                To negatively condition the generation process based on a specific crop coordinates. Part of SDXL's
                micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            negative_target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                To negatively condition the generation process based on a target image resolution. It should be as same
                as the `target_size` for most cases. Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
        Examples:
        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
            :type attention_store: object
        """



        # Initialize WandB
        wandb.init(project="attention-visualization", name="PredictedDiffPipeline")  # Replace with your project and run names

        # 0. Default height and width to unet
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)
        
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._cross_attention_kwargs = cross_attention_kwargs

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0
         # 3. Encode input prompt
        
        (
            prompt_embeds, add_text_embeds, add_time_ids,
        ) = self._encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        )
        

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps
        )

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        # print("3. Latents:", latents.shape)
        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare added time ids & embeddings
  
        prompt_embeds = prompt_embeds.to(device)
        # print("3.prompt_embeds:",prompt_embeds.shape)
        add_text_embeds = add_text_embeds.to(device)
        # print("3.add_text_embeds:",add_text_embeds.shape)
        # print("3.add_time_ids:",add_time_ids.shape)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)
        # print("4.add_time_ids:",add_time_ids.shape)

        scale_range = np.linspace(scale_range[0], scale_range[1], len(self.scheduler.timesteps))
        

        if max_iter_to_alter is None:
            max_iter_to_alter = len(self.scheduler.timesteps) + 1

        attention_maps_t = []
        attention_size = attention_res ** 2
        loss_value_per_step = []

        #attention_for_obj_t = []
        attention_for_obj_t = torch.zeros(51,16,16,20)


        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                print("I",i)
                with torch.enable_grad():

                    latents = latents.clone().detach().requires_grad_(True)

                    # Forward pass of denoising with text conditioning
                    # Adjust for SDXL's Dual Text Encoders
                    added_cond_kwargs = {"text_embeds": add_text_embeds[batch_size:], "time_ids": add_time_ids[1::2]}
                    # print("add_text_embeds[batch_size:].unsqueeze(0):", add_text_embeds[batch_size:].unsqueeze(0).shape)
                    # print("time_ids: add_time_ids[1::2]:",add_time_ids[1::2].unsqueeze(0).shape)
                    noise_pred_text = self.unet(
                    latents,
                    t,
                    encoder_hidden_states=prompt_embeds[batch_size:],
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]
                    self.unet.zero_grad()
                    print("encoder_hidden_states",prompt_embeds.shape)
                    print("noise_pred_text",noise_pred_text.shape)
                
                    # Get attention maps per token 
                    # attention maps : [16,16,77]
                    """
                    attention_maps = self._aggregate_attention_per_token(
                        attention_store=attention_store,
                        attention_res=attention_res,
                    )
                    """
                    # attention_maps = self._aggregate_attention_per_token(attention_store=attention_store,attention_res=attention_res)
                    
                
                    if run_attention_sd:
                        #attention_maps = self._aggregate_attention_per_token(attention_store=attention_store,attention_res=attention_res)
                        all_prompt_attention_maps = []
                        prompt_losses = [] # add a list to store losses for each image in the batch.
                        for batch_idx in range(batch_size):  # Iterate through the batch
                            attention_maps = aggregate_attention(
                            attention_store=attention_store,
                            res=attention_res,
                            from_where=("up", "down", "mid"),
                            is_cross=True,
                            select=batch_idx  # Select the current batch index
                        )
                        all_prompt_attention_maps.append(attention_maps)

                        print(loss_function)
                        
                        # Now, all_prompt_attention_maps is a list of attention maps, one for each prompt
                        for prompt_idx, attention_maps in enumerate(all_prompt_attention_maps):
                            
                            attention_corr_indices=attention_corr_indices[prompt_idx], # changed to use prompt index
                            attention_leak_indices=attention_leak_indices[prompt_idx], # changed to use prompt index
                            attention_exist_indices=attention_exist_indices[prompt_idx], # changed to use prompt index
                            attention_possession_indices=attention_possession_indices[prompt_idx], # changed to use prompt index
                            
                            if loss_function == "attention":
                                loss = self._compute_loss_attention(attention_maps=attention_maps,attention_corr_indices=attention_corr_indices,size=attention_size)
                                if i < max_iter_to_alter:
                                    if loss != 0:
                                        latents = self._update_latent(latents=latents, loss=loss,
                                                                step_size=scale_factor * np.sqrt(scale_range[i]))
                                        print(f'Iteration {i} Prompt {prompt_idx}| Loss: {loss:0.4f}')  
                                wandb.log({f"attention_loss_prompt_{prompt_idx}": loss}, step=i)  # Log loss
                                wandb.log({f"attention_maps_prompt_{prompt_idx}": wandb.Image(attention_maps.cpu().numpy())}, step=i)  # Log attention maps
                            elif loss_function == "attention_product_prob":
                                
                                loss,attention_for_obj = self._compute_loss_attention_product_prob(attention_maps=attention_maps,size=attention_size,
                                                                                                    attention_corr_indices=attention_corr_indices,
                                                                                                    attention_leak_indices=attention_leak_indices,
                                                                                                    attention_exist_indices=attention_exist_indices,
                                                                                                    attention_possession_indices= attention_possession_indices,
                                                                                                    is_sdxl=is_sdxl)
                                run_negation = False
                                if run_negation == True:
                                    if neg_prompt is None:
                                        1/0
                                    else:
                                        #prompt_add_neg = prompt + ' ' + neg_prompt
                                        prompt_neg = neg_prompt
                                        #print("prompt_add_neg",neg_prompt)
                                

                                    prompt_add_neg_embeds, add_text_neg_embeds, add_time_neg_ids = self._encode_prompt(prompt_neg,device,num_images_per_prompt,do_classifier_free_guidance,negative_prompt,prompt_embeds=prompt_embeds,negative_prompt_embeds=negative_prompt_embeds)
                                    added_cond_neg_kwargs = {"text_embeds": add_text_neg_embeds[batch_size:], "time_ids": add_time_neg_ids[1::2]}
                                    noise_pred_text = self.unet(
                                    latents,
                                    t,
                                    encoder_hidden_states=prompt_add_neg_embeds[batch_size:],
                                    cross_attention_kwargs=self.cross_attention_kwargs,
                                    added_cond_kwargs=added_cond_neg_kwargs,
                                    return_dict=False,
                                    )[0]
                                    
                                    self.unet.zero_grad()

                                    attention_maps_for_negation = self._aggregate_attention_per_token(attention_store=attention_store,attention_res=attention_res)
                                    loss_neg,attention_for_obj_neg = self._compute_loss_attention_negation(attention_maps=attention_maps_for_negation,attention_corr_indices=attention_corr_indices,size=attention_size)

                                    loss = loss + loss_neg
                                
                                if i < max_iter_to_alter:
                                    if loss != 0:
                                        latents = self._update_latent(latents=latents, loss=loss,
                                                                step_size=scale_factor * np.sqrt(scale_range[i]))
                                        print(f'Iteration {i} Prompt {prompt_idx}| Loss: {loss:0.4f}')

                                wandb.log({f"product_prob_loss_prompt_{prompt_idx}": loss}, step=i)  # Log loss
                                wandb.log({f"product_prob_attention_maps_prompt_{prompt_idx}": wandb.Image(attention_maps.cpu().numpy())}, step=i)  # Log attention maps
                                # Optionally, log attention_for_obj if it's relevant
                                if attention_for_obj is not None:
                                    wandb.log({f"attention_for_obj_prompt_{prompt_idx}": wandb.Image(attention_for_obj.cpu().numpy())}, step=i)

                            elif loss_function == "attention_exist":
                                print("Exist")
                                loss,attention_for_obj = self._compute_loss_attention_exist(attention_maps=attention_maps,attention_corr_indices=attention_corr_indices,size=attention_size)
                                if i < max_iter_to_alter:
                                    if loss != 0:
                                        latents = self._update_latent(latents=latents, loss=loss,
                                                                step_size=scale_factor * np.sqrt(scale_range[i]))
                                        print(f'Iteration {i} Prompt {prompt_idx}| Loss: {loss:0.4f}')
                                wandb.log({f"exit_loss_prompt_{prompt_idx}": loss}, step=i)  # Log loss
                                wandb.log({f"exit_attention_maps_prompt_{prompt_idx}": wandb.Image(attention_maps.cpu().numpy())}, step=i)  # Log attention maps
                                # Optionally, log attention_for_obj if it's relevant
                                if attention_for_obj is not None:
                                    wandb.log({f"attention_for_obj_prompt_{prompt_idx}": wandb.Image(attention_for_obj.cpu().numpy())}, step=i)
                            
                            elif loss_function == "attention_negation":
                                print("Negation")
                                if neg_prompt is None:
                                    1/0
                                else:
                                    #prompt_add_neg = prompt + ' ' + neg_prompt
                                    prompt_add_neg = neg_prompt
                                #print("prompt_add_neg",neg_prompt)
                                

                                # text_add_neg_inputs,prompt_add_neg_embeds = self._encode_prompt(prompt_add_neg,device,num_images_per_prompt,do_classifier_free_guidance,negative_prompt,prompt_embeds=prompt_embeds,negative_prompt_embeds=negative_prompt_embeds)
                                prompt_add_neg_embeds, add_text_neg_embeds, add_time_neg_ids = self._encode_prompt(prompt_add_neg,device,num_images_per_prompt,do_classifier_free_guidance,negative_prompt,prompt_embeds=prompt_embeds,negative_prompt_embeds=negative_prompt_embeds)
                                added_cond_neg_kwargs = {"text_embeds": add_text_neg_embeds[batch_size:], "time_ids": add_time_neg_ids[1::2]}
                                noise_pred_text = self.unet(
                                    latents,
                                    t,
                                    encoder_hidden_states=prompt_add_neg_embeds[batch_size:],
                                    cross_attention_kwargs=self.cross_attention_kwargs,
                                    added_cond_kwargs=added_cond_neg_kwargs,
                                    return_dict=False,
                                    )[0]

                                # noise_pred_text = self.unet(latents, t, encoder_hidden_states=prompt_add_neg_embeds[1].unsqueeze(0), cross_attention_kwargs=cross_attention_kwargs).sample
                                self.unet.zero_grad()

                                attention_maps = self._aggregate_attention_per_token(attention_store=attention_store,attention_res=attention_res)
                                loss,attention_for_obj = self._compute_loss_attention_negation(attention_maps=attention_maps,attention_corr_indices=attention_corr_indices,size=attention_size)
                                
                                if i < max_iter_to_alter:
                                    if loss != 0:
                                        latents = self._update_latent(latents=latents, loss=loss,
                                                                step_size=scale_factor * np.sqrt(scale_range[i]))
                                        print(f'Iteration {i} Prompt {prompt_idx}| Loss: {loss:0.4f}')
                                
                                wandb.log({f"negation_loss_prompt_{prompt_idx}": loss}, step=i)  # Log loss
                                wandb.log({f"negation_attention_maps_prompt_{prompt_idx}": wandb.Image(attention_maps.cpu().numpy())}, step=i)  # Log attention maps
                                # Optionally, log attention_for_obj if it's relevant
                                if attention_for_obj is not None:
                                    wandb.log({f"attention_for_obj_prompt_{prompt_idx}": wandb.Image(attention_for_obj.cpu().numpy())}, step=i)
                        
                            
                            
                            if i in thresholds.keys(): #and loss > 1. - thresholds[i]:
                                if loss_function != "attention_negation":
                                    loss,latents = self._perform_iterative_refinement_fixed_step_att(
                                        latents=latents,
                                        loss=loss,
                                        threshold=thresholds[i],
                                        batch_size = batch_size,
                                        text_embeddings=prompt_embeds,
                                        add_text_embeds= add_text_embeds, 
                                        add_time_ids=add_time_ids,
                                        attention_store=attention_store,
                                        step_size=scale_factor * np.sqrt(scale_range[i]),
                                        t=t,
                                        attention_res=attention_res,
                                        max_refinement_steps=20,
                                        attention_corr_indices=attention_corr_indices,
                                        attention_leak_indices=attention_leak_indices,
                                        attention_exist_indices=attention_exist_indices,
                                        attention_possession_indices= attention_possession_indices,
                                        attention_loss_function=loss_function)
                                else:
                                    loss,latents = self._perform_iterative_refinement_fixed_step_att(
                                        latents=latents,
                                        loss=loss,
                                        threshold=thresholds[i],
                                        batch_size=batch_size,
                                        text_embeddings=prompt_add_neg_embeds,
                                        add_text_embeds= add_text_neg_embeds, 
                                        add_time_ids=add_time_neg_ids,
                                        attention_store=attention_store,
                                        step_size=scale_factor * np.sqrt(scale_range[i]),
                                        t=t,
                                        attention_res=attention_res,
                                        max_refinement_steps=20,
                                        attention_corr_indices=attention_corr_indices,
                                        attention_leak_indices=attention_leak_indices,
                                        attention_exist_indices=attention_exist_indices,
                                        attention_possession_indices= attention_possession_indices,
                                        attention_loss_function=loss_function)
                                    
                                wandb.log({f"threshold_loss_prompt_{prompt_idx}": loss}, step=i)  # Log loss


                            prompt_losses.append(loss.detach().cpu().numpy()) # append the loss for the current prompt.
                        loss_value_per_step.append(prompt_losses)#loss.detach().cpu().numpy()
                        #loss_value_per_step = torch.cat(loss_value_per_step,dim=0)
                        #print(loss_value_per_step)
                        

                attention_for_obj_t[i,:,:,:]=torch.stack(all_prompt_attention_maps, dim=0)[:,:,:,0:20] # changed to stack the maps.
                    

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                #print("latent_model_input",latent_model_input.shape)
                
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                self.unet.zero_grad()
                if i == 50:

                    attention_maps = self._aggregate_attention_per_token(attention_store=attention_store,attention_res=attention_res)

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)


                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)
                


                
                # save attention maps for t
                if i in attention_save_t:
                    """
                    text_add_neg_inputs,prompt_add_neg_embeds = self._encode_prompt(neg_prompt,device,num_images_per_prompt,do_classifier_free_guidance,negative_prompt,prompt_embeds=prompt_embeds,negative_prompt_embeds=negative_prompt_embeds)
                    noise_pred_text = self.unet(latents, t,
                                                encoder_hidden_states=prompt_add_neg_embeds[1].unsqueeze(0), cross_attention_kwargs=cross_attention_kwargs).sample
                    self.unet.zero_grad()
                    """
                    """
                    attention_maps = self._aggregate_attention_per_token(attention_store=attention_store,attention_res=attention_res) 
                    print("max",attention_maps[:,:,1].max())
                    print("ave",torch.mean(attention_maps[:,:,1]))
                    """

                    if run_attention_sd == False:
                        attention_for_obj = []

                    """
                    print("len_token",num_tokens)
                    last_idx = num_tokens - 1
                    #last_idx = -1
                    attention_normalize = attention_maps[:, :, 1:last_idx]
                    attention_normalize *= 100
                    attention_maps[:,:,1:last_idx] = torch.nn.functional.softmax(attention_normalize, dim=-1)
                    print("AttShape",attention_maps[:,:,3].max())
                    """
                    
                
                    #attention_maps_t.append(attention_maps)
                

                
                #attention_for_obj_t.append(attention_maps[:,:,0:20])
                    
        #attention_for_obj_t[50,:,:,:]=attention_maps[:,:,0:20]       
                
            
        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
            elif latents.dtype != self.vae.dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    self.vae = self.vae.to(latents.dtype)

            # unscale/denormalize the latents
            # denormalize with the mean and std if available and not None
            has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
            has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None
            if has_latents_mean and has_latents_std:
                latents_mean = (
                    torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                )
                latents_std = (
                    torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                )
                latents = latents * latents_std / self.vae.config.scaling_factor + latents_mean
            else:
                latents = latents / self.vae.config.scaling_factor

            image = self.vae.decode(latents, return_dict=False)[0]

            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        print("image_numpy",image.shape)

        # 10. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)
            
            #attention_maps = attention_maps.to('cpu').numpy()
            

        if not return_dict:
            return (image,)

        
        
        return StableDiffusionXLPipelineOutput(images=image),loss_value_per_step,attention_for_obj_t