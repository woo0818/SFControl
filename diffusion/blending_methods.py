from abc import ABC, abstractmethod
import torch
import einops

from utils.rotation_conversions import *
from utils import utils_transform
from human_body_prior.tools.rotation_tools import aa2matrot
import mano
import open3d as o3d
from copy import deepcopy

from utils.meshviewer import points2sphere, get_arrow
import pdb

__BLENDING_METHOD__ = {}

def register_blending_method(name: str):
    def wrapper(cls):
        if __BLENDING_METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __BLENDING_METHOD__[name] = cls
        return cls
    return wrapper

def get_blending_method(name: str, **kwargs):
    if __BLENDING_METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __BLENDING_METHOD__[name](**kwargs)

    
class BlendingMethod(ABC):
    def __init__(self, mean, std, n_frames, **kwargs):
        super().__init__()
    
    @abstractmethod
    def blending(self, x_t, idx, measurements, **kwargs):
        pass
    

@register_blending_method(name='linear')
class LinearBlending(BlendingMethod):
    def __init__(self, mean, std, n_frames,  **kwargs):
        super().__init__(mean, std, n_frames, **kwargs)
        self.mean = mean
        self.std = std
        self.n_frames = n_frames

        self.max_steps = 999

        self.linear = torch.tensor([i/(self.n_frames-1) for i in range(self.n_frames)])
        self.linear_inv = torch.flip(self.linear, [0])

        self.linear = self.linear.unsqueeze(1).to("cuda")
        self.linear_inv = self.linear_inv.unsqueeze(1).to("cuda")

    def transpose(self, data):
        return data * self.std + self.mean

    def inv_transpose(self, data):
        return (data - self.mean) / self.std

    def blending(self, x_prev, idx, measurements, **kwargs):
        y = kwargs['model_kwargs']['y']
        obj_pos = y['goal']

        start_pos = y['start'][-3:]

        
        sample = self.transpose(x_prev)
        sample = einops.rearrange(sample[:,:,0,:], 'b d s -> b s d')[0]

        rot_global_aa = utils_transform.sixd2aa(sample[-1,0:12][6:12].reshape(-1, 1, 6), batch=True).reshape(-1, 3)
        palm_mat = aa2matrot(rot_global_aa)[0].T
        palm_position = sample[-1,12:18][3:6] - palm_mat[0]/10. - palm_mat[1]/20.
        
        delta_goal = (obj_pos-palm_position).unsqueeze(0).float()
        delta_start = (start_pos - sample[0,12:18][3:6]).unsqueeze(0).float()

        blend_offset_goal = torch.matmul(self.linear, delta_goal)
        sample[:,12:18][:,3:6] += blend_offset_goal * 1 #(idx/self.max_steps)
        
        blend_offset_start = torch.matmul(self.linear_inv, delta_start)
        sample[:,12:18][:,3:6] += blend_offset_start * 1 #(idx/self.max_steps)

        #pdb.set_trace()

        if idx % 2000 == -1:
            s_data = sample
            rot_global_6d = s_data[:,:12]
            pos_global = s_data[:,12:18]

            rot_global_aa = utils_transform.sixd2aa(rot_global_6d.reshape(-1, 2, 6), batch=True).reshape(-1, 6)
            root_rot_global_aa = rot_global_aa[:,:3].detach().cpu()
            hand_rot_global_aa = rot_global_aa[:,3:6].detach().cpu()
            root_pos_global = pos_global[:,:3].detach().cpu()
            hand_pos_global = pos_global[:,3:6].detach().cpu()

            mano_path = "body_models/mano/MANO_RIGHT.pkl" 

            rh_model = mano.load(model_path=mano_path,
                    model_type='mano',
                    is_right= True,
                    num_pca_comps=45,
                    batch_size=self.n_frames,
                    flat_hand_mean=True)
                    
            hand_offset = torch.tensor([0.095669, 0.0063834, 0.0061863]).float()
            
            sample_hand_mesh = rh_model(global_orient=hand_rot_global_aa.float(),
                hand_pose=torch.zeros(self.n_frames,45).float(),
                transl=hand_pos_global.float() - hand_offset,
                return_verts=True,
                return_tips =True)
            

            #for f_num in range(0,self.n_frames,5):
            f_num = self.n_frames-1

            coord_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
            hand_coord_mesh_mv  = deepcopy(coord_mesh).translate(hand_pos_global[f_num])
            R = coord_mesh.get_rotation_matrix_from_axis_angle(hand_rot_global_aa[f_num])

            ev = o3d.visualization.ExternalVisualizer()
            hand_mesh = o3d.geometry.TriangleMesh()
            hand_mesh.vertices = o3d.utility.Vector3dVector(sample_hand_mesh.vertices[f_num].detach().cpu().numpy())
            hand_mesh.triangles = o3d.utility.Vector3iVector(rh_model.faces)
            hand_mesh.compute_vertex_normals()
            hand_mesh.paint_uniform_color([0.3,0.3,0.4])
            ev.set(obj=[(hand_mesh,'hand_mesh_'+ str(idx) +str(f_num))])

            palm = hand_pos_global[f_num]-R.T[0]/10.-R.T[1]/20.
            palm_estimate_mesh = points2sphere(palm.detach().cpu().numpy(), radius = 0.02)
            palm_mesh = o3d.geometry.TriangleMesh()
            palm_mesh.vertices = o3d.utility.Vector3dVector(palm_estimate_mesh.vertices)
            palm_mesh.triangles = o3d.utility.Vector3iVector(palm_estimate_mesh.faces)
            palm_mesh.compute_vertex_normals()
            palm_mesh.paint_uniform_color([0.6, 0.6, 0.0])
            ev.set(obj=[(palm_mesh,'palm' + str(idx)+str(f_num))])

            pdb.set_trace()

        sample = einops.rearrange(sample, 's d -> d s').unsqueeze(1).unsqueeze(0)
        
        x_t = self.inv_transpose(sample).float()
        
        return x_t
 