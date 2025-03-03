import abc

# import cv2
import numpy as np
import torch
# from IPython.display import display

from typing import Union, Tuple, List
import torch.nn.functional as F

from diffusers.models.attention_processor import Attention
from spatial_processor import AttnProcessor, AttnProcessor2_0, SpatialAttnProcessor2_0
from packaging import version
import importlib

def is_torch2_available():
    return version.parse(torch.__version__) >= version.parse("2.0.0")

class AttendExciteCrossAttnProcessor():
    def __init__(self, attnstore, place_in_unet):
        super().__init__()
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, scale=1.0):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        ### softmax(Q.T * K) ###
        
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        self.attnstore(attention_probs, is_cross, self.place_in_unet)
        
       
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
    
def is_torch2_available():
    return hasattr(F, "scaled_dot_product_attention")

# def register_attention_control(model, controller, id_length, sa32, sa64, height, width):

#     attn_procs = {}
#     cross_att_count = 0
#     attn_count = 0
#     total_count = 0
#     cur_step = 0
#     for name, attn in model.unet.attn_processors.items():
        
#         cross_attention_dim = None if name.endswith("attn1.processor") else model.unet.config.cross_attention_dim
#         if name.startswith("mid_block"):
#             hidden_size = model.unet.config.block_out_channels[-1]
#             place_in_unet = "mid"
#         elif name.startswith("up_blocks"):
#             block_id = int(name[len("up_blocks.")])
#             hidden_size = list(reversed(model.unet.config.block_out_channels))[block_id]
#             place_in_unet = "up"
#         elif name.startswith("down_blocks"):
#             block_id = int(name[len("down_blocks.")])
#             hidden_size = model.unet.config.block_out_channels[block_id]
#             place_in_unet = "down"
#         else:
#             continue

#         if cross_attention_dim is None:
#             # Self-attention
#             if name.startswith("up_blocks") :
#                 attn_procs[name] = SpatialAttnProcessor2_0(id_length = id_length,total_count=total_count, sa32 = sa32, sa64 = sa64, height = height, width = width)
#                 total_count +=1
#             else:    
#                 if is_torch2_available():
#                     attn_procs[name] = AttnProcessor2_0()
#                 else:
#                     attn_procs[name] = AttnProcessor()
#         else:
#             # Cross-attention
#             cross_att_count += 1
#             attn_procs[name] = AttendExciteCrossAttnProcessor(
#                 attnstore=controller, place_in_unet=place_in_unet
#             )
    
#     model.unet.set_attn_processor(attn_procs)
#     controller.num_att_layers = cross_att_count
    
#     write = True # set to True when you want to write the id_bank.
#     for name, proc in model.unet.attn_processors.items():
#         if isinstance(proc, SpatialAttnProcessor2_0):
#             proc.write = write
#             proc.cur_step = 0
#             proc.attn_count = 0

def register_attention_control(model, controller):

    attn_procs = {}
    cross_att_count = 0
    for name in model.unet.attn_processors.keys():
        
        cross_attention_dim = None if name.endswith("attn1.processor") else model.unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = model.unet.config.block_out_channels[-1]
            place_in_unet = "mid"
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(model.unet.config.block_out_channels))[block_id]
            place_in_unet = "up"
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = model.unet.config.block_out_channels[block_id]
            place_in_unet = "down"
        else:
            continue

        cross_att_count += 1
        attn_procs[name] = AttendExciteCrossAttnProcessor(
            attnstore=controller, place_in_unet=place_in_unet
        )
    
    #print(cross_att_count)
    
    model.unet.set_attn_processor(attn_procs)
    controller.num_att_layers = cross_att_count

    
class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            self.forward(attn, is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
            #print("bet_steps")
            

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        


class EmptyControl(AttentionControl):

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        return attn


class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        #print("append",attn.shape)
        
        return attn

    def between_steps(self):
        self.attention_store = self.step_store
        if self.save_global_store:
            with torch.no_grad():
                if len(self.global_store) == 0:
                    self.global_store = self.step_store
                else:
                    for key in self.global_store:
                        for i in range(len(self.global_store[key])):
                            self.global_store[key][i] += self.step_store[key][i].detach()
        self.step_store = self.get_empty_store()
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = self.attention_store
        return average_attention

    def get_average_global_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.global_store[key]] for key in
                             self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.global_store = {}

    def __init__(self, save_global_store=False):
        '''
        Initialize an empty AttentionStore
        :param step_index: used to visualize only a specific step in the diffusion process
        '''
        super(AttentionStore, self).__init__()
        self.save_global_store = save_global_store
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.global_store = {}
        self.curr_step_index = 0
        


def aggregate_attention(attention_store: AttentionStore,
                        res: int,
                        from_where: List[str],
                        is_cross: bool,
                        select: int) -> torch.Tensor:
    """ Aggregates the attention across the different layers and heads at the specified resolution. """
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(1, -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    #print("out",out.shape)
    out = out.sum(0) / out.shape[0]
    #out = torch.min(out,dim=0).values
    #print("out",out.shape)
    #print("out_max",out[:,:,11].max())
    #print("out_min",out.min())
    """
    last_idx = -1
    out = out[:, :, 1:last_idx]
    out *= 100
    out = torch.nn.functional.softmax(out,-1)
    """
    """
    out_max = out.reshape(-1,out.shape[2])
    out_max = torch.max(out_max,dim=0).values
    out = out / out_max
    """
    
    
    return out