'''This module handles task-dependent operations (A) and noises (n) to simulate a measurement y=Ax+n.'''

from abc import ABC, abstractmethod
from functools import partial
import yaml
import einops 
from torch.nn import functional as F
from torchvision import torch

from data_loaders.humanml.scripts.motion_process import recover_from_ric
from torchcubicspline import NaturalCubicSpline, natural_cubic_spline_coeffs

#from utils.data_util import data_conversion, data_initial_align
#from data_loaders.humanml.scripts.motion_process import recover_from_ric, recover_from_ric_hand

import pdb

# =================
# Operation classes
# =================

__OPERATOR__ = {}

def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OPERATOR__[name] = cls
        return cls
    return wrapper


def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)


class LinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        # calculate A * X
        pass

    @abstractmethod
    def transpose(self, data, **kwargs):
        # calculate A^T * X
        pass
    
    def ortho_project(self, data, **kwargs):
        # calculate (I - A^T * A)X
        return data - self.transpose(self.forward(data, **kwargs), **kwargs)

    def project(self, data, measurement, **kwargs):
        # calculate (I - A^T * A)Y - AX
        return self.ortho_project(measurement, **kwargs) - self.forward(data, **kwargs)


@register_operator(name='noise')
class DenoiseOperator(LinearOperator):
    def __init__(self, device):
        self.device = device
    
    def forward(self, data):
        return data

    def transpose(self, data):
        return data
    
    def ortho_project(self, data):
        return data

    def project(self, data):
        return data


class NonLinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        pass

    def project(self, data, measurement, **kwargs):
        return data + measurement - self.forward(data) 



@register_operator(name='traj')
class TrajOperator(NonLinearOperator):
    def __init__(self, device, mean, std, motion_len, t0=0, t1=-1, select_joints = 0, gt_joints=None, reparam=False, data_rep='glo_root'):
        self.device = device
        self.mean = mean#.to(device)
        self.std = std#.to(device)
        self.motion_len = motion_len
        self.t0 = t0
        self.t1 = t1
        self.select_joints = select_joints
        self.gt_joints = self.arc_length_parameterize_and_resample(gt_joints[0,self.t0:self.t1,select_joints]).detach().to(self.device)
        self.reparam = reparam
        self.data_rep = data_rep
    
    def transpose(self, data):
        return data * self.std + self.mean
    
    def arc_length_parameterize_and_resample(self, joints):
        """
        Input: joints [motion_len, 3]
        Output: Resampled joints [motion_len, 3] with uniform arc length spacing
        """
        # 1. Compute arc length
        diffs = torch.diff(joints, dim=0)  # [motion_len-1, 3]
        seg_lengths = torch.norm(diffs, dim=1)  # [motion_len-1]
        cumulative_length = torch.cat([
            torch.zeros(1, device=joints.device),
            torch.cumsum(seg_lengths, dim=0)
        ])  # [motion_len]
        total_length = cumulative_length[-1]
        
        # 2. Normalize arc length to [0, 1]
        s_normalized = cumulative_length / total_length  # [motion_len]
        
        # 3. Cubic spline interpolation (diffable)
        coeffs = natural_cubic_spline_coeffs(s_normalized, joints)
        spline = NaturalCubicSpline(coeffs)
        
        # 4. Resample at uniform arc length positions
        s_uniform = torch.linspace(0, 1, len(joints), device=joints.device)
        resampled_joints = spline.evaluate(s_uniform)  # [motion_len, 3]
        
        return resampled_joints

    def traj_error(self, joints, gt_joints, reparam=False):
        if reparam:
            joints = self.arc_length_parameterize_and_resample(joints)

        loss = torch.norm(joints - gt_joints, dim=-1).mean()

        return loss
    
    def traj_len_error(self, joints, gt_joints, reparam=False):
        if reparam:
            joints = self.arc_length_parameterize_and_resample(joints)

        
        def compute_total_length(traj):
            diffs = torch.diff(traj, dim=0)
            return torch.sum(torch.norm(diffs, dim=1))
        
        pred_length = compute_total_length(joints)
        gt_length = compute_total_length(gt_joints)
        length_loss = torch.abs(pred_length - gt_length)

        return length_loss


    def forward(self, data, **kwargs):
        
        sample = self.transpose(data)
        sample = einops.rearrange(sample[:,:,0,:], 'b d s -> b s d')

        if self.data_rep == "glo_joints":
            joints = torch.cat([recover_from_ric(sample, joints_num=1, abs_3d=True), sample[:,:,4:].reshape(1,sample.shape[1],-1,3)], dim=2)
        else:
            joints = recover_from_ric(sample, joints_num=6, abs_3d=True)

        sub_joints = joints[:,:self.motion_len][0,self.t0:self.t1,self.select_joints]

        traj_loss = self.traj_error(sub_joints, self.gt_joints, reparam=self.reparam )
        length_loss = self.traj_len_error(sub_joints, self.gt_joints, reparam=self.reparam )
        
        loss = traj_loss #+ 0.001*length_loss

        print(traj_loss.item(), 0.001*length_loss.item())

        return loss
       
    def project(self, data, measurement, **kwargs):
        return data

@register_operator(name='task_goal')
class GoalPositionRetrievalOperator(NonLinearOperator):
    def __init__(self, device, mean, std, motion_len, gt_joints, select_joints):
        self.device = device
        self.mean = mean
        self.std = std
        self.motion_len = motion_len
        self.gt_joints = gt_joints.to("cuda")
        self.select_joints = select_joints
    
    def transpose(self, data):
        return data * self.std + self.mean

    def forward(self, data, **kwargs):

        sample = self.transpose(data)
        sample = einops.rearrange(sample[:,:,0,:], 'b d s -> b s d')
        
        positions = torch.cat([recover_from_ric(sample, joints_num=1, abs_3d=True), sample[:,:,4:].reshape(1,sample.shape[1],-1,3)], dim=2)
        #positions = recover_from_ric(sample, joints_num=6, abs_3d=True)

        #print(positions[:,self.motion_len-1,self.select_joints[0]])
        #print(self.gt_joints[:,self.motion_len-1,self.select_joints[0]])
       
        loss = torch.norm(positions[:,self.motion_len-1,self.select_joints[0]] - self.gt_joints[:,self.motion_len-1,self.select_joints[0]], dim=-1).mean()

        return loss
       
    def project(self, data, measurement, **kwargs):
        return data

@register_operator(name='goal')
class GoalPositionRetrievalOperator(NonLinearOperator):
    def __init__(self, device, mean, std):
        self.device = device
        self.mean = mean
        self.std = std
    
    def transpose(self, data):
        return data * self.std + self.mean

    def forward(self, data, **kwargs):
        abs_3d = kwargs['model_kwargs']['y']['abs_3d']
        abs_3d_root_only = kwargs['model_kwargs']['y']['abs_3d_root_only']
        y = kwargs['model_kwargs']['y']

        sample = self.transpose(data)
        sample = einops.rearrange(sample[:,:,0,:], 'b d s -> b s d')
        if y['dataset'] == "circle_keypoint":
            positions = recover_from_ric_hand(sample, abs_3d=abs_3d, abs_3d_root_only=abs_3d_root_only)
        elif y['dataset'] == "circle_motion":
            positions = recover_from_ric(sample, abs_3d=abs_3d, abs_3d_root_only=abs_3d_root_only)
        #positions = data_conversion(sample)

        if "root_quat_init_inv" in y.keys():
            positions = data_initial_align(positions, y['root_quat_init_inv'])
        if y['init_root_y_zero']:
            positions = positions + y['root_pose_init_y']
        
        if y['dataset'] == "circle_keypoint":
            return positions[0,-1,1,:]
        elif y['dataset'] == "circle_motion":
            return positions[0,y["target_frame"],y["n_hand"],:]
       
    def project(self, data, measurement, **kwargs):
        return data

@register_operator(name='start')
class GoalPositionRetrievalOperator(NonLinearOperator):
    def __init__(self, device, mean, std, source=False):
        self.device = device
        self.mean = mean
        self.std = std
        self.source = source
        self.n_joints = 22
    
    def transpose(self, data):
        return data * self.std + self.mean

    def forward(self, data, **kwargs):
        if self.source:
            return data[:,0:4+(self.n_joints- 1)*3,:,0]
        abs_3d = kwargs['model_kwargs']['y']['abs_3d']
        abs_3d_root_only = kwargs['model_kwargs']['y']['abs_3d_root_only']
        y = kwargs['model_kwargs']['y']

        sample = self.transpose(data)
        sample = einops.rearrange(sample[:,:,0,:], 'b d s -> b s d')
        if y['dataset'] == "circle_keypoint":
            positions = recover_from_ric_hand(sample, abs_3d=abs_3d, abs_3d_root_only=abs_3d_root_only)
        elif y['dataset'] == "circle_motion":
            positions = recover_from_ric(sample, abs_3d=abs_3d, abs_3d_root_only=abs_3d_root_only)
        #positions = data_conversion(sample)

        if "root_quat_init_inv" in y.keys():
            positions = data_initial_align(positions, y['root_quat_init_inv'])
        if y['init_root_y_zero']:
            positions = positions + y['root_pose_init_y']

        if y['dataset'] == "circle_keypoint":
            return torch.stack([positions[0,0,0,:],positions[0,0,1,:]],axis=1)
        elif y['dataset'] == "circle_motion":
            return positions[0,0,:,:]
        
    def project(self, data, measurement, **kwargs):
        return data

@register_operator(name='collision')
class GoalPositionRetrievalOperator(NonLinearOperator):
    def __init__(self, device, mean, std, size, scale, sdf_offset, s_grid_sdf):
        self.device = device
        self.mean = mean
        self.std = std
        self.size = size
        self.scale = scale
        self.sdf_offset = sdf_offset
        self.s_grid_sdf = torch.transpose(s_grid_sdf,0,2).unsqueeze(0).unsqueeze(1)
    
    def transpose(self, data):
        return data * self.std + self.mean

    def sdf_transform(self, data):
        scaled_data = data * self.scale
        sdf_data = scaled_data * self.size / (2*self.sdf_offset+1)

        return sdf_data

    def forward(self, data, **kwargs):
        y = kwargs['model_kwargs']['y']
        abs_3d = y['abs_3d']
        abs_3d_root_only = y['abs_3d_root_only']

        sample = self.transpose(data)
        sample = einops.rearrange(sample[:,:,0,:], 'b d s -> b s d')
        if y['dataset'] == "circle_keypoint":
            positions = recover_from_ric_hand(sample, abs_3d=abs_3d, abs_3d_root_only=abs_3d_root_only)
        elif y['dataset'] == "circle_motion":
            positions = recover_from_ric(sample, abs_3d=abs_3d, abs_3d_root_only=abs_3d_root_only)
        #positions = data_conversion(sample)

        if "root_quat_init_inv" in y.keys():
            positions = data_initial_align(positions, y['root_quat_init_inv'])

        sdf_positions = positions.clone()
        if not y['init_root_y_zero']:
            sdf_positions[:,:,:,1] = sdf_positions[:,:,:,1].clone() - y['root_pose_init_y'][1]

        sdf_positions = self.sdf_transform(sdf_positions)

        joint_sdf_batch = F.grid_sample(self.s_grid_sdf,sdf_positions.view(1,-1,1,1,3),padding_mode="border",align_corners=False)
        if joint_sdf_batch.lt(0).sum().item() < 1:
            sdf_pene = torch.tensor(0.0, dtype=torch.float32, device="cuda")
        else:
            sdf_pene = joint_sdf_batch[joint_sdf_batch < 0].abs().mean()

        return sdf_pene
        
    def project(self, data, measurement, **kwargs):
        return data

@register_operator(name='keypoint')  #TODO condition with keypoint trajectory?
class GoalPositionRetrievalOperator(NonLinearOperator):
    def __init__(self, device, mean, std):
        self.device = device
        self.mean = mean
        self.std = std
        self.n_joints = 22
        self.joint_num = self.n_joints
        self.n_hand = 21
        # n_joints = 22 if sample.shape[1] == 263 else 21
    
    def transpose(self, data):
        return data * self.std + self.mean

    def forward(self, data, r_sample=False, **kwargs):
        y = kwargs['model_kwargs']['y']
        condition_type = y['condition_type']
        target_frame = y['target_frame']
        
        if condition_type == "KeyJoints":
            root = data[:,0:4,:,:target_frame]
            ric_hand = data[:,4:4+(self.joint_num - 1)*3,:,:target_frame][:,(self.n_hand-1)*3:self.n_hand*3]
            return torch.cat([root, ric_hand],dim=1)
        
        elif condition_type == "KeyJoints_All":
            root = data[:,0:4,:,:target_frame]
            ric_hand = data[:,4:4+(self.joint_num - 1)*3,:,:target_frame][:,(self.n_hand-1)*3:self.n_hand*3]
            rot_hand = data[:,4+(self.joint_num - 1)*3:4+(self.joint_num - 1)*9,:,:target_frame][:,(self.n_hand-1)*6:self.n_hand*6]
            vel_root = data[:,4+(self.joint_num - 1)*9:4+(self.joint_num - 1)*9+self.joint_num*3,:,:target_frame][:,0:3]
            vel_hand = data[:,4+(self.joint_num - 1)*9:4+(self.joint_num - 1)*9+self.joint_num*3,:,:target_frame][:,self.n_hand*3:(self.n_hand+1)*3]
            return torch.cat([root, ric_hand, rot_hand, vel_root, vel_hand],dim=1)
        
    def project(self, data, measurement, **kwargs):
        return data

@register_operator(name='mdm_goal')
class GoalPositionRetrievalOperator(NonLinearOperator):
    def __init__(self, device, mean, std):
        self.device = device
        self.mean = mean
        self.std = std
        self.n_joints = 22
        self.n_hand = 21
        # n_joints = 22 if sample.shape[1] == 263 else 21
    
    def transpose(self, data):
        return data * self.std + self.mean

    def forward(self, data, r_sample=False, **kwargs):
        
        sample = self.transpose(data.permute(0, 2, 3, 1))
        sample = recover_from_ric(sample, self.n_joints)
        sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1) #[1,22,3,120]
        
        if r_sample:
            return sample

        root_traje = sample[0,0,:,:]
        goal = sample[0,self.n_hand,:,-1]
        root_goal = sample[0,0,:,-1]
        return root_traje
        
    def project(self, data, measurement, **kwargs):
        return data

# =============
# Noise classes
# =============


__NOISE__ = {}

def register_noise(name: str):
    def wrapper(cls):
        if __NOISE__.get(name, None):
            raise NameError(f"Name {name} is already defined!")
        __NOISE__[name] = cls
        return cls
    return wrapper

def get_noise(name: str, **kwargs):
    if __NOISE__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    noiser = __NOISE__[name](**kwargs)
    noiser.__name__ = name
    return noiser

class Noise(ABC):
    def __call__(self, data):
        return self.forward(data)
    
    @abstractmethod
    def forward(self, data):
        pass

@register_noise(name='clean')
class Clean(Noise):
    def forward(self, data):
        return data

@register_noise(name='gaussian')
class GaussianNoise(Noise):
    def __init__(self, sigma):
        self.sigma = sigma
    
    def forward(self, data):
        return data + torch.randn_like(data, device=data.device) * self.sigma


@register_noise(name='poisson')
class PoissonNoise(Noise):
    def __init__(self, rate):
        self.rate = rate

    def forward(self, data):
        '''
        Follow skimage.util.random_noise.
        '''

        # TODO: set one version of poisson
       
        # version 3 (stack-overflow)
        import numpy as np
        data = (data + 1.0) / 2.0
        data = data.clamp(0, 1)
        device = data.device
        data = data.detach().cpu()
        data = torch.from_numpy(np.random.poisson(data * 255.0 * self.rate) / 255.0 / self.rate)
        data = data * 2.0 - 1.0
        data = data.clamp(-1, 1)
        return data.to(device)

        # version 2 (skimage)
        # if data.min() < 0:
        #     low_clip = -1
        # else:
        #     low_clip = 0

    
        # # Determine unique values in iamge & calculate the next power of two
        # vals = torch.Tensor([len(torch.unique(data))])
        # vals = 2 ** torch.ceil(torch.log2(vals))
        # vals = vals.to(data.device)

        # if low_clip == -1:
        #     old_max = data.max()
        #     data = (data + 1.0) / (old_max + 1.0)

        # data = torch.poisson(data * vals) / float(vals)

        # if low_clip == -1:
        #     data = data * (old_max + 1.0) - 1.0
       
        # return data.clamp(low_clip, 1.0)