import numpy as np
import torch
import torch.nn as nn
from PIL import Image
# from torch.cuda.amp import autocast
# from PIL import Image
# import diffusers
# from diffusers import StableDiffusionXLPipeline, AutoPipelineForText2Image, DDIMScheduler
import torch.nn.functional as F
import copy
import random
import wandb



def cal_attn_mask_xl(total_length,id_length,sa32,sa64,height,width,device="cuda",dtype= torch.float16):
    nums_1024 = (height // 32) * (width // 32)
    nums_4096 = (height // 16) * (width // 16)
    bool_matrix1024 = torch.rand((1, total_length * nums_1024),device = device,dtype = dtype) < sa32
    bool_matrix4096 = torch.rand((1, total_length * nums_4096),device = device,dtype = dtype) < sa64
    bool_matrix1024 = bool_matrix1024.repeat(total_length,1)
    bool_matrix4096 = bool_matrix4096.repeat(total_length,1)
    for i in range(total_length):
        bool_matrix1024[i:i+1,id_length*nums_1024:] = False
        bool_matrix4096[i:i+1,id_length*nums_4096:] = False
        bool_matrix1024[i:i+1,i*nums_1024:(i+1)*nums_1024] = True
        bool_matrix4096[i:i+1,i*nums_4096:(i+1)*nums_4096] = True
    mask1024 = bool_matrix1024.unsqueeze(1).repeat(1,nums_1024,1).reshape(-1,total_length * nums_1024)
    mask4096 = bool_matrix4096.unsqueeze(1).repeat(1,nums_4096,1).reshape(-1,total_length * nums_4096)
    # Log the masks to Wandb
    wandb.log({"mask1024": wandb.Image(mask1024.float().cpu().numpy())})
    wandb.log({"mask4096": wandb.Image(mask4096.float().cpu().numpy())})
    
    return mask1024,mask4096

class AttnProcessor(nn.Module):
    r"""
    Default processor for performing attention-related computations.
    """
    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
    ):
        super().__init__()

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        # print("AttnProcessor : ", batch_size)
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        # Log the attention layer outputs
        # Log the attention layer outputs as heatmaps
        if wandb.run:
            # Compute the heatmap by taking the mean across the feature dimension (axis=2)
            # print(f"Final hidden_states shape: {hidden_states.shape}")
            heatmaps = hidden_states.cpu().detach().numpy().mean(axis=2)  # Shape: [8, 1024]
            # print("HeatMap Shape: ", heatmaps.shape)
            # Normalize the heatmap values to the range [0, 1]
            heatmaps = (heatmaps - heatmaps.min()) / (heatmaps.max() - heatmaps.min())

            # Dynamically calculate the heatmap dimensions
            sequence_length = heatmaps.shape[1]
            height = int(np.sqrt(sequence_length))
            width = sequence_length // height

            # Reshape to [batch_size, height, width] if sequence_length represents spatial dimensions
            # height, width = 32, 32  # Example dimensions, adjust as needed
            heatmaps = heatmaps.reshape(batch_size, height, width)

            # Convert to PIL images and log with Wandb
            heatmap_images = [Image.fromarray((heatmap * 255).astype(np.uint8), mode='L') for heatmap in heatmaps]
            wandb.log({"heatmaps": [wandb.Image(img) for img in heatmap_images]})
            # wandb.log({"hidden_states": wandb.Image(hidden_states.cpu().detach().numpy())})

        return hidden_states


class AttnProcessor2_0(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """
    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
    ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        # print("AttnProcessor2_0 : ", batch_size)

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        # Log the attention layer outputs
        # Log the attention layer outputs as heatmaps
        if wandb.run:
            # Compute the heatmap by taking the mean across the feature dimension (axis=2)
            # print(f"Final hidden_states shape: {hidden_states.shape}")
            heatmaps = hidden_states.cpu().detach().numpy().mean(axis=2)  # Shape: [8, 1024]
            # print("HeatMap Shape: ", heatmaps.shape)
            # Normalize the heatmap values to the range [0, 1]
            heatmaps = (heatmaps - heatmaps.min()) / (heatmaps.max() - heatmaps.min())

            # Dynamically calculate the heatmap dimensions
            sequence_length = heatmaps.shape[1]
            height = int(np.sqrt(sequence_length))
            width = sequence_length // height

            # Reshape to [batch_size, height, width] if sequence_length represents spatial dimensions
            # height, width = 32, 32  # Example dimensions, adjust as needed
            heatmaps = heatmaps.reshape(batch_size, height, width)

            # Convert to PIL images and log with Wandb
            heatmap_images = [Image.fromarray((heatmap * 255).astype(np.uint8), mode='L') for heatmap in heatmaps]
            wandb.log({"heatmaps": [wandb.Image(img) for img in heatmap_images]})
            # wandb.log({"hidden_states": wandb.Image(hidden_states.cpu().detach().numpy())})
        

        return hidden_states

class SpatialAttnProcessor2_0(torch.nn.Module):
    r"""
    Attention processor for IP-Adapater for PyTorch 2.0.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        text_context_len (`int`, defaults to 77):
            The context length of the text features.
        scale (`float`, defaults to 1.0):
            the weight scale of image prompt.
    """

    def __init__(self, hidden_size = None, cross_attention_dim=None,id_length = 4,total_count=0,sa32 = 0.5, sa64 = 0.5, height = 768, width = 768, attn_count=0, curr_step=0, write = False, device = "cuda",dtype = torch.float16):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.device = device
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.total_length = id_length + 1
        self.id_length = id_length
        self.total_count = total_count
        self.id_bank = {}
        self.height = height
        self.width = width
        self.sa32 = sa32
        self.sa64 = sa64
        self.mask1024, self.mask4096 = cal_attn_mask_xl(
            self.total_length, id_length, sa32, sa64, height, width, device=device, dtype=dtype
        )
        self.attn_count = attn_count
        self.cur_step = curr_step
        self.write = write # store write as instance variable

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None):

        if self.write:
            # print(f"white:{cur_step}")
            self.id_bank[self.cur_step] = [hidden_states[:self.id_length], hidden_states[self.id_length:]]
        else:
            encoder_hidden_states = torch.cat((self.id_bank[self.cur_step][0].to(self.device),hidden_states[:1],self.id_bank[self.cur_step][1].to(self.device),hidden_states[1:]))
        # skip in early step
        if self.cur_step <5:
            hidden_states = self.__call2__(attn, hidden_states,encoder_hidden_states,attention_mask,temb)
        else:   # 256 1024 4096
            random_number = random.random()
            if self.cur_step <20:
                rand_num = 0.3
            else:
                rand_num = 0.1
            if random_number > rand_num:
                if not self.write:
                    if hidden_states.shape[1] == (self.height//32) * (self.width//32):
                        attention_mask = self.mask1024[self.mask1024.shape[0] // self.total_length * self.id_length:]
                    else:
                        attention_mask = self.mask4096[self.mask4096.shape[0] // self.total_length * self.id_length:]
                else:
                    if hidden_states.shape[1] == (self.height//32) * (self.width//32):
                        attention_mask = self.mask1024[:self.mask1024.shape[0] // self.total_length * self.id_length,:self.mask1024.shape[0] // self.total_length * self.id_length]
                    else:
                        attention_mask = self.mask4096[:self.mask4096.shape[0] // self.total_length * self.id_length,:self.mask4096.shape[0] // self.total_length * self.id_length]
                hidden_states = self.__call1__(attn, hidden_states,encoder_hidden_states,attention_mask,temb)
            else:
                hidden_states = self.__call2__(attn, hidden_states,None,attention_mask,temb)
        self.attn_count +=1
        if self.attn_count == self.total_count:
            self.attn_count = 0
            self.cur_step += 1
            self.mask1024,self.mask4096 = cal_attn_mask_xl(self.total_length,self.id_length,self.sa32,self.sa64,self.height,self.width, device=self.device, dtype= self.dtype)
        # Log the attention layer outputs
        # wandb.log({"hidden_states": wandb.Image(hidden_states.cpu().numpy())})
        return hidden_states
    def __call1__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            total_batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(total_batch_size, channel, height * width).transpose(1, 2)
        total_batch_size,nums_token,channel = hidden_states.shape
        img_nums = total_batch_size//2
        hidden_states = hidden_states.view(-1,img_nums,nums_token,channel).reshape(-1,img_nums * nums_token,channel)

        batch_size, sequence_length, _ = hidden_states.shape
        # print("SpatialAttnProcessor2_0 :Call1: ",hidden_states.shape)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states  # B, N, C
        else:
            encoder_hidden_states = encoder_hidden_states.view(-1,self.id_length+1,nums_token,channel).reshape(-1,(self.id_length+1) * nums_token,channel)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)


        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(total_batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)



        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)


        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(total_batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        print(hidden_states.shape)
        # Log the attention layer outputs
        if wandb.run:
            # Compute the heatmap by taking the mean across the feature dimension (axis=2)
            heatmaps = hidden_states.cpu().detach().numpy().mean(axis=2)  # Shape: [8, 1024]

            # Normalize the heatmap values to the range [0, 1]
            heatmaps = (heatmaps - heatmaps.min()) / (heatmaps.max() - heatmaps.min())

            # Dynamically calculate the heatmap dimensions
            sequence_length = heatmaps.shape[1]
            height = int(np.sqrt(sequence_length))
            width = sequence_length // height

            # Reshape to [batch_size, height, width] if sequence_length represents spatial dimensions
            # height, width = 16, 16  # Example dimensions, adjust as needed
            heatmaps = heatmaps.reshape(batch_size, height, width)

            # Convert to PIL images and log with Wandb
            heatmap_images = [Image.fromarray((heatmap * 255).astype(np.uint8), mode='L') for heatmap in heatmaps]
            wandb.log({"heatmaps": [wandb.Image(img) for img in heatmap_images]})
            # wandb.log({"hidden_states": wandb.Image(hidden_states.cpu().detach().numpy())})

        return hidden_states
    def __call2__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, channel = (
            hidden_states.shape
        )
        # print("SpatialAttnProcessor2_0 :Call2: ",hidden_states.shape)
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states  # B, N, C
        else:
            encoder_hidden_states = encoder_hidden_states.view(-1,self.id_length+1,sequence_length,channel).reshape(-1,(self.id_length+1) * sequence_length,channel)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        # Log the attention layer outputs
        if wandb.run:
            # Compute the heatmap by taking the mean across the feature dimension (axis=2)
            heatmaps = hidden_states.cpu().detach().numpy().mean(axis=2)  # Shape: [8, 1024]

            # Normalize the heatmap values to the range [0, 1]
            heatmaps = (heatmaps - heatmaps.min()) / (heatmaps.max() - heatmaps.min())
            # Dynamically calculate the heatmap dimensions
            sequence_length = heatmaps.shape[1]
            height = int(np.sqrt(sequence_length))
            width = sequence_length // height

            # Reshape to [batch_size, height, width] if sequence_length represents spatial dimensions
            # height, width = 16, 16  # Example dimensions, adjust as needed
            heatmaps = heatmaps.reshape(batch_size, height, width)

            # Convert to PIL images and log with Wandb
            heatmap_images = [Image.fromarray((heatmap * 255).astype(np.uint8), mode='L') for heatmap in heatmaps]
            wandb.log({"heatmaps": [wandb.Image(img) for img in heatmap_images]})
            # wandb.log({"hidden_states": wandb.Image(hidden_states.cpu().detach().numpy())})

        return hidden_states