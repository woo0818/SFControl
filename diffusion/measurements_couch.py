'''This module handles task-dependent operations (A) and noises (n) to simulate a measurement y=Ax+n.'''

from abc import ABC, abstractmethod
from functools import partial
import yaml
import einops 
from torch.nn import functional as F
from torchvision import torch
from utils.data_util import data_conversion, data_initial_align
from utils.rotation_conversions import *
from data_loaders.humanml.scripts.motion_process import recover_from_ric, recover_from_ric_hand

from utils import utils_transform
from human_body_prior.tools.rotation_tools import aa2matrot , matrot2aa #, local2global_pose

import mano
import open3d as o3d
from copy import deepcopy
from utils.meshviewer import points2sphere, get_arrow
from tools.meshviewer import Mesh, points2sphere, colors, segment2cyliner
import pdb

import numpy as np

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


@register_operator(name='couch_keypoint_goal')
class GoalPositionRetrievalOperator(NonLinearOperator):
    def __init__(self, device, mean, std):
        self.device = device
        self.mean = mean
        self.std = std
    
    def transpose(self, data):
        return data * self.std + self.mean

    def forward(self, data, **kwargs):
        y = kwargs['model_kwargs']['y']
        goal_pos = y['goal']

        sample = self.transpose(data)
        sample = einops.rearrange(sample[:,:,0,:], 'b d s -> b s d')[0]

        #rot_global_aa = utils_transform.sixd2aa(sample[-1,0:12][0:6].reshape(-1, 1, 6), batch=True).reshape(-1, 3)
        #palm_mat = aa2matrot(rot_global_aa)[0].T
        hand_position = sample[-1,-6:] # - palm_mat[0]/10. - palm_mat[1]/10.
        
        return torch.norm(goal_pos-hand_position) # weight_type 1

        #if kwargs['model_kwargs']['weight_type']["goal"] == 1:
        #    return torch.norm(obj_pos-palm_position) # weight_type 1
        #elif kwargs['model_kwargs']['weight_type']["goal"] == 0:
        #    return sample[-1,12:18][3:6] # weight_type 0
       
    def project(self, data, measurement, **kwargs):
        return data


@register_operator(name='circle_smpl_keypoint_object')
class GoalPositionRetrievalOperator(NonLinearOperator):
    def __init__(self, device, mean, std):
        self.device = device
        self.mean = mean
        self.std = std
    
    def transpose(self, data):
        return data * self.std + self.mean

    def forward(self, data, **kwargs):
        y = kwargs['model_kwargs']['y']
        obj_pos = y['goal']

        sample = self.transpose(data)
        sample = einops.rearrange(sample[:,:,0,:], 'b d s -> b s d')[0]

        rot_global_aa = utils_transform.sixd2aa(sample[-1,0:12][0:6].reshape(-1, 1, 6), batch=True).reshape(-1, 3)
        palm_mat = aa2matrot(rot_global_aa)[0].T
        palm_position = sample[-1,12:18][3:6] - palm_mat[0]/10. - palm_mat[1]/10.
        
        return torch.norm(obj_pos-palm_position) # weight_type 1

        #if kwargs['model_kwargs']['weight_type']["goal"] == 1:
        #    return torch.norm(obj_pos-palm_position) # weight_type 1
        #elif kwargs['model_kwargs']['weight_type']["goal"] == 0:
        #    return sample[-1,12:18][3:6] # weight_type 0
       
    def project(self, data, measurement, **kwargs):
        return data

@register_operator(name='couch_keypoint_start')
class GoalPositionRetrievalOperator(NonLinearOperator):
    def __init__(self, device, mean, std, couch_root):
        self.device = device
        self.mean = mean
        self.std = std
        self.couch_root = couch_root
    
    def transpose(self, data):
        return data * self.std + self.mean

    def forward(self, data, **kwargs):
        sample = self.transpose(data)
        sample = einops.rearrange(sample[:,:,0,:], 'b d s -> b s d')[0]

        #pdb.set_trace()
        if self.couch_root:
            return torch.cat((sample[0,6:18], sample[0,21:27]), axis=0)
        else:
            return sample[0]
        #return torch.cat((sample[0,0:12],sample[0,12:18][0:6]), axis=0)
       
    def project(self, data, measurement, **kwargs):
        return data

@register_operator(name='circle_smpl_keypoint_hand_collision')
class GoalPositionRetrievalOperator(NonLinearOperator):
    def __init__(self, device, mean, std, size, scale, sdf_offset, sdf_grid, rotmat, path_type):
        self.device = device
        self.mean = mean
        self.std = std
        self.size = size
        self.scale = scale
        self.sdf_offset = sdf_offset
        self.sdf_grid = sdf_grid.unsqueeze(0).unsqueeze(1)
        self.inv_rotmat = torch.inverse(rotmat).float().to("cuda")
        self.path_type = path_type
        #torch.transpose(s_grid_sdf,0,2).unsqueeze(0).unsqueeze(1)
    
    def transpose(self, data):
        return data * self.std + self.mean

    def sdf_transform(self, data):

        return (data / (2.0 / (self.scale*self.size))) / self.sdf_offset

    def forward(self, data, **kwargs):
        y = kwargs['model_kwargs']['y']

        sample = self.transpose(data)
        sample = einops.rearrange(sample[:,:,0,:], 'b d s -> b s d')[0]

        positions = sample[:,12:18][:,3:6]
        #
        positions = (positions.clone() - y['root']).float()
        
        if self.path_type == "a_star":
            positions = torch.matmul(self.inv_rotmat, positions.T).T

        sdf_positions = self.sdf_transform(positions)


        joint_sdf_batch = F.grid_sample(self.sdf_grid, sdf_positions[:,[2,1,0]].view(1,-1,1,1,3),padding_mode="border",align_corners=False)
        if joint_sdf_batch.lt(0).sum().item() < 1:
            sdf_pene = torch.tensor(0.0, dtype=torch.float32, device="cuda")
        else:
            sdf_pene = joint_sdf_batch[joint_sdf_batch < 0].abs().mean()

        return sdf_pene
        
    def project(self, data, measurement, **kwargs):
        return data

@register_operator(name='circle_smpl_keypoint_bbox_collision')
class GoalPositionRetrievalOperator(NonLinearOperator):
    def __init__(self, device, mean, std, size, scale, sdf_offset, sdf_grid, rotmat, path_type, bbox_model, nframes):
        self.device = device
        self.mean = mean.float()
        self.std = std.float()
        self.size = size
        self.scale = scale
        self.sdf_offset = sdf_offset
        self.sdf_grid = sdf_grid.unsqueeze(0).unsqueeze(1)
        self.inv_rotmat = torch.inverse(rotmat).float().to("cuda")
        self.path_type = path_type
        self.bbox_net = bbox_model.model.net.eval()
        self.bbox_dataset = bbox_model.train_dataset.dataset.dataset
        self.npoints = 1000
        self.nframes = nframes
        self.r = 1.0

        self.lines = [[0, 1], [1, 2], [2, 3], [0, 3],
                        [4, 5], [5, 6], [6, 7], [4, 7],
                        [0, 4], [1, 5], [2, 6], [3, 7]]

        #torch.transpose(s_grid_sdf,0,2).unsqueeze(0).unsqueeze(1)

        self.points = self.sample_points()

    
    def transpose(self, data):
        return data * self.std + self.mean

    def sdf_transform(self, data):

        return (data / (2.0/(self.scale*self.size))) / self.sdf_offset

    def sample_points(self, ):

        #torch.rand(self.npoints, 2)

        rand_z = torch.rand(self.npoints, requires_grad = False, device="cuda").unsqueeze(1)
        rand_theta = torch.rand(self.npoints, requires_grad = False, device="cuda").unsqueeze(1)

        rand_theta -= 0.5
        rand_theta *= 2*torch.pi

        rand_z -= 0.5

        rand_sample = torch.cat([0.5*torch.cos(rand_theta), rand_z, 0.5*torch.sin(rand_theta)], dim=1)


        return rand_sample.repeat(self.nframes,1,1).float()

    def _axis_angle_rotation(self, axis: str, angle, inv=False):
        
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        one = torch.ones_like(angle)
        zero = torch.zeros_like(angle)

        if axis == "X":
            R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
        if axis == "Y":
            if inv:
                R_flat = (cos, zero, - sin, zero, one, zero, sin, zero, cos)
            else:
                R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
        if axis == "Z":
            R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)

        return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))

    def sample(self, points, theta, trans, scale):

        rotmat = self._axis_angle_rotation("Y", theta, inv=True)
        

        scale_points = torch.mul(points, scale.unsqueeze(1)) 

        rot_sample = torch.matmul(rotmat, scale_points.transpose(1,2))

        sample = rot_sample + torch.matmul(rotmat, trans.unsqueeze(-1))
        
        return sample.transpose(1,2)

    
    def bbox(self, root_pos, root_mat, bbox):
                
        x_min, y_min, z_min, x_max, y_max, z_max = bbox

        corner_box = torch.stack([
            root_pos + x_min*root_mat[0] + y_min*root_mat[1] + z_min*root_mat[2],
            root_pos + x_min*root_mat[0] + y_min*root_mat[1] + z_max*root_mat[2],
            root_pos + x_min*root_mat[0] + y_max*root_mat[1] + z_max*root_mat[2],
            root_pos + x_min*root_mat[0] + y_max*root_mat[1] + z_min*root_mat[2],
            root_pos + x_max*root_mat[0] + y_min*root_mat[1] + z_min*root_mat[2],
            root_pos + x_max*root_mat[0] + y_min*root_mat[1] + z_max*root_mat[2],
            root_pos + x_max*root_mat[0] + y_max*root_mat[1] + z_max*root_mat[2],
            root_pos + x_max*root_mat[0] + y_max*root_mat[1] + z_min*root_mat[2]])

        return corner_box


    def forward(self, data, **kwargs):
        ev = o3d.visualization.ExternalVisualizer()


        y = kwargs['model_kwargs']['y']

        sample = self.transpose(data)
        sample = einops.rearrange(sample[:,:,0,:], 'b d s -> b s d')[0]

        positions = sample[:,12:18][:,3:6]

        bbox_input_positions = sample.clone()
        bbox_input_positions[:,12:18][:,0] = 0.0
        bbox_input_positions[:,12:18][:,2] = 0.0
        bbox_input_positions[:,12:18][:,3] -= bbox_input_positions[:,12:18][:,0].clone()
        bbox_input_positions[:,12:18][:,5] -= bbox_input_positions[:,12:18][:,2].clone()

        bbox_input = self.bbox_dataset.normalize_motion(bbox_input_positions)

        bbox_pred = self.bbox_net(bbox_input)
        bbox_out = self.bbox_dataset.unnormalize_bbox(bbox_pred)

        

        upper_sample = self.sample(self.points, bbox_out[:,12], (bbox_out[:,0:3]+bbox_out[:,3:6])/2.0,(bbox_out[:,3:6]-bbox_out[:,0:3])) + sample[:,12:15].unsqueeze(1)
        lower_sample = self.sample(self.points, bbox_out[:,13], (bbox_out[:,6:9]+bbox_out[:,9:12])/2.0,(bbox_out[:,9:12]-bbox_out[:,6:9])) + sample[:,12:15].unsqueeze(1)
        '''
        for f_num in range(0,self.nframes,10):
            corner_bbox = self.bbox(sample[f_num,12:15], self._axis_angle_rotation("Y", bbox_out[f_num,12]), bbox_out[f_num,0:6])
            cylinders_mesh = segment2cyliner(corner_bbox, self.lines)
            cylinder_mesh = o3d.geometry.TriangleMesh()
            cylinder_mesh.vertices = o3d.utility.Vector3dVector(np.array(cylinders_mesh.vertices))
            cylinder_mesh.triangles = o3d.utility.Vector3iVector(np.array(cylinders_mesh.faces))
            cylinder_mesh.paint_uniform_color([0,0,1])

            ev.set(obj=[(cylinder_mesh, 'upper_rotate_check+{}'.format(f_num))])

            #upper_sample_zero = self.sample(self.points, bbox_out[f_num:f_num+1,12], (bbox_out[f_num:f_num+1,0:3]+bbox_out[f_num:f_num+1,3:6]),(bbox_out[f_num:f_num+1,3:6]-bbox_out[f_num:f_num+1,0:3]))
            upper_sample_zero = upper_sample[f_num].reshape(-1,3)

            measurement_mesh = points2sphere(upper_sample_zero.detach().cpu().numpy(), radius = 0.01)
            
            start_hand_mesh = o3d.geometry.TriangleMesh()
            start_hand_mesh.vertices = o3d.utility.Vector3dVector(measurement_mesh.vertices)
            start_hand_mesh.triangles = o3d.utility.Vector3iVector(measurement_mesh.faces)
            start_hand_mesh.compute_vertex_normals()
            start_hand_mesh.paint_uniform_color([0.6, 0.2, 0.2])
            ev.set(obj=[(start_hand_mesh,'sample points_bbox+{}'.format(f_num))])

        # measurement_mesh = points2sphere(self.points.detach().cpu().numpy(), radius = 0.01)
        
        # start_hand_mesh = o3d.geometry.TriangleMesh()
        # start_hand_mesh.vertices = o3d.utility.Vector3dVector(measurement_mesh.vertices)
        # start_hand_mesh.triangles = o3d.utility.Vector3iVector(measurement_mesh.faces)
        # start_hand_mesh.compute_vertex_normals()
        # start_hand_mesh.paint_uniform_color([0.6, 0.2, 0.2])
        # ev.set(obj=[(start_hand_mesh,'sample points')])


        #pdb.set_trace()
        '''

        upper_sample = upper_sample.clone() - y['root'].unsqueeze(0).unsqueeze(0)
        lower_sample = lower_sample.clone() - y['root'].unsqueeze(0).unsqueeze(0)
        
        if self.path_type == "a_star":
            upper_sample = torch.matmul(self.inv_rotmat, upper_sample.T).T
            lower_sample = torch.matmul(self.inv_rotmat, lower_sample.T).T

        sdf_upper_sample = self.sdf_transform(upper_sample)
        sdf_lower_sample = self.sdf_transform(lower_sample)

        upper_sdf_batch = F.grid_sample(self.sdf_grid, sdf_upper_sample[:,[2,1,0]].reshape(1,-1,1,1,3),padding_mode="border",align_corners=False)
        lower_sdf_batch = F.grid_sample(self.sdf_grid, sdf_lower_sample[:,[2,1,0]].reshape(1,-1,1,1,3),padding_mode="border",align_corners=False)


        if upper_sdf_batch.lt(0).sum().item() < 1:
            upper_sdf_pene = torch.tensor(0.0, dtype=torch.float32, device="cuda")
        else:
            upper_sdf_pene = upper_sdf_batch[upper_sdf_batch < 0].abs().mean()

        if lower_sdf_batch.lt(0).sum().item() < 1:
            lower_sdf_pene = torch.tensor(0.0, dtype=torch.float32, device="cuda")
        else:
            lower_sdf_pene = lower_sdf_batch[lower_sdf_batch < 0].abs().mean()

        return upper_sdf_pene + lower_sdf_pene
        
    def project(self, data, measurement, **kwargs):
        return data


@register_operator(name='couch_smpl_goal')
class GoalPositionRetrievalOperator(NonLinearOperator):
    def __init__(self, device, kintree, smpl_layer, betas, mean, std):
        self.device = device
        self.betas = betas.to(device)
        self.smpl_layer = smpl_layer
        self.mean = mean
        self.std = std
        self.kintree_table = kintree
        self.idx = 0
    
    def transpose(self, data):
        return data * self.std + self.mean

    def local2global_pose(self, local_pose, kintree):
        bs = local_pose.shape[0]

        local_pose = local_pose.view(bs, -1, 3, 3)

        global_pose = local_pose.clone()

        for jId in range(len(kintree)):
            parent_id = kintree[jId]
            if parent_id >= 0:
                #print(parent_id, jId)
                global_pose[:, jId] = torch.matmul(global_pose[:, parent_id].clone(), global_pose[:, jId].clone())

        return global_pose

    def forward(self, data, **kwargs):
        self.idx += 1
        y = kwargs['model_kwargs']['y']

        obj_pos = y['goal']

        sample = self.transpose(data)
        sample = einops.rearrange(sample[:,:,0,:], 'b d s -> b s d')[0]

        root_trans = sample[-1,:3].reshape(-1,3)
        root_6d = sample[-1,3:9].reshape(-1,6)
        local_6d = sample[-1,9:147].reshape(-1,138)

        root_aa = utils_transform.sixd2aa(root_6d.reshape(-1, 1, 6), batch=True).reshape(-1, 3)
        local_aa = utils_transform.sixd2aa(local_6d.reshape(-1, 23, 6), batch=True).reshape(-1, 23*3)

        bdata_poses = utils_transform.sixd2aa(sample[:,3:147].reshape(-1, 24, 6), batch=True).reshape(-1, 24*3)

        beta = torch.tensor(self.betas).repeat(root_trans.shape[0],1).float()

        #pdb.set_trace()

        verts, Jtr = self.smpl_layer(torch.Tensor(bdata_poses[-1:]), torch.Tensor(beta).to("cuda"), torch.Tensor(root_trans).to("cuda").float())
        
    

        return torch.norm(obj_pos-Jtr[:,[10, 11]].reshape(-1))


       
    def project(self, data, measurement, **kwargs):
        return data

@register_operator(name='couch_smpl_start')
class GoalPositionRetrievalOperator(NonLinearOperator):
    def __init__(self, device, kintree, smpl_layer, betas, mean, std):
        self.device = device
        self.betas = betas.to(device)
        self.smpl_layer = smpl_layer
        self.mean = mean
        self.std = std
        self.kintree_table = kintree
        self.idx = 0
    
    def transpose(self, data):
        return data * self.std + self.mean

    def local2global_pose(self, local_pose, kintree):
        bs = local_pose.shape[0]

        local_pose = local_pose.view(bs, -1, 3, 3)

        global_pose = local_pose.clone()

        for jId in range(len(kintree)):
            parent_id = kintree[jId]
            if parent_id >= 0:
                #print(parent_id, jId)
                global_pose[:, jId] = torch.matmul(global_pose[:, parent_id].clone(), global_pose[:, jId].clone())

        return global_pose

    def forward(self, data, **kwargs):
        self.idx += 1
        y = kwargs['model_kwargs']['y']

        obj_pos = y['start']

        sample = self.transpose(data)
        sample = einops.rearrange(sample[:,:,0,:], 'b d s -> b s d')[0]

        root_trans = sample[0,:3].reshape(-1,3)
        root_6d = sample[0,3:9].reshape(-1,6)
        local_6d = sample[0,9:147].reshape(-1,138)

        root_aa = utils_transform.sixd2aa(root_6d.reshape(-1, 1, 6), batch=True).reshape(-1, 3)
        local_aa = utils_transform.sixd2aa(local_6d.reshape(-1, 23, 6), batch=True).reshape(-1, 23*3)

        bdata_poses = utils_transform.sixd2aa(sample[:,3:147].reshape(-1, 24, 6), batch=True).reshape(-1, 24*3)

        beta = torch.tensor(self.betas).repeat(root_trans.shape[0],1).float()

        

        verts, Jtr = self.smpl_layer(torch.Tensor(bdata_poses[0:1]), torch.Tensor(beta).to("cuda"), torch.Tensor(root_trans).to("cuda").float())
        
        

        return torch.norm(obj_pos-Jtr[:,[10, 11]].reshape(-1))


       
    def project(self, data, measurement, **kwargs):
        return data

@register_operator(name='circle_smpl_keypoint')
class GoalPositionRetrievalOperator(NonLinearOperator):
    def __init__(self, device, kintree, bm, betas, mean, std):
        self.device = device
        self.betas = betas.to(device)
        self.bm = bm
        self.mean = mean
        self.std = std
        self.kintree_table = kintree
        self.idx = 0
    
    def transpose(self, data):
        return data * self.std + self.mean

    def local2global_pose(self, local_pose, kintree):
        bs = local_pose.shape[0]

        local_pose = local_pose.view(bs, -1, 3, 3)

        global_pose = local_pose.clone()

        for jId in range(len(kintree)):
            parent_id = kintree[jId]
            if parent_id >= 0:
                #print(parent_id, jId)
                global_pose[:, jId] = torch.matmul(global_pose[:, parent_id].clone(), global_pose[:, jId].clone())

        return global_pose

    def forward(self, data, **kwargs):
        self.idx += 1
        y = kwargs['model_kwargs']['y']

        obj_pos = y['goal']

        hand_pos = y['hand_trans']
        hand_aa = y['hand_aa']
        hand_matrot = y['hand_matrot']

        root_pos = y['root_trans']
        root_aa = y['root_aa']
        root_matrot = y['root_matrot']

        sample = self.transpose(data)
        sample = einops.rearrange(sample[:,:,0,:], 'b d s -> b s d')[0]

        root_trans = sample[-1,:3].reshape(-1,3)
        root_6d = sample[-1,3:9].reshape(-1,6)
        local_6d = sample[-1,9:135].reshape(-1,126)

        root_aa = utils_transform.sixd2aa(root_6d.reshape(-1, 1, 6), batch=True).reshape(-1, 3)
        local_aa = utils_transform.sixd2aa(local_6d.reshape(-1, 21, 6), batch=True).reshape(-1, 21*3)

        #body_pose_world = self.bm(
        #        root_orient=root_aa.float(),
        #        pose_body=local_aa.float(),
        #        trans=root_trans.float(),
        #        return_verts=True,
        #        return_tips =True)



        body_pose_world = self.bm(global_orient=root_aa,
                    body_pose=local_aa,
                    transl=root_trans,
                    betas=self.betas.repeat(root_aa.shape[0],1),
                    return_verts=True,
                    return_tips =True)

        hand_position = body_pose_world.joints[
            :, 21, :
        ]

        root_position = body_pose_world.joints[
            :, 0, :
        ]

        bdata_poses = torch.cat((root_aa, local_aa), axis=-1)

        rotation_local_matrot = aa2matrot(
            bdata_poses.clone().reshape(-1, 3)
        ).reshape(bdata_poses.shape[0], -1, 9)

        rotation_global_matrot = self.local2global_pose(
            rotation_local_matrot.clone(), self.kintree_table
        )
# 
        palm_mat = rotation_global_matrot[0,21].T
        palm_position = hand_position - palm_mat[0]/10. - palm_mat[1]/20.

        return torch.norm(hand_position-hand_pos)
       
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