import numpy as np
import math

import torch
import torch.nn.functional as F

import pdb

'''
https://github.com/xh-liu/SDG_code/blob/main/sdg/gaussian_diffusion.py
'''

def start_loss(positions, start, args):    
    loss = F.mse_loss(positions[0,0,0,1],start[1])
    return -loss

def goal_loss(positions, goal, args):    
    loss = F.mse_loss(positions[0,-1,1,:],goal)
    return -loss

'''
https://github.com/scenediffuser/Scene-Diffuser/blob/main/models/optimizer/motion_in_scene.py
'''
def collision_loss(positions, s_grid_sdf, init_root_y, args):
    joint_sdf_batch = F.grid_sample(torch.transpose(s_grid_sdf,0,2).unsqueeze(0).unsqueeze(1),positions.view(1,-1,1,1,3),padding_mode="border",align_corners=False)
    if joint_sdf_batch.lt(0).sum().item() < 1:
        sdf_pene = torch.tensor(0.0, dtype=torch.float32, device="cuda")
    else:
        sdf_pene = joint_sdf_batch[joint_sdf_batch < 0].abs().mean()

    return -sdf_pene


