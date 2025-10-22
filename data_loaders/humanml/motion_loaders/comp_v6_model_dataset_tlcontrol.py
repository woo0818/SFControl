import torch
from utils.fixseed import fixseed
from data_loaders.humanml.networks.modules import *
from data_loaders.humanml.networks.trainers import CompTrainerV6
from torch.utils.data import Dataset, DataLoader
from os.path import join as pjoin
from tqdm import tqdm
from utils import dist_util
import os
import copy
from functools import partial

from data_loaders.humanml.data.dataset import abs3d_to_rel, sample_to_motion, rel_to_abs3d
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from data_loaders.humanml.utils.metrics import calculate_skating_ratio
from sample.gmd.condition import (cond_fn_key_location, get_target_from_kframes, get_target_and_inpt_from_kframes_batch,
                              log_trajectory_from_xstart, get_inpainting_motion_from_traj, get_inpainting_motion_from_gt,
                              cond_fn_key_location, compute_kps_error, cond_fn_sdf,
                              CondKeyLocations, CondKeyLocationsWithSdf, compute_kps_error_arbitrary)
from utils.editing_util import get_keyframes_mask
import open3d as o3d
import pdb


from visualize.simplify_loc2rot import joints2smpl


def hardCode_inv_transform_traj(data):
    traj_std = torch.load( "./sample/info_traj_std.pt").to(data.device)
    traj_mean = torch.load( "./sample/info_traj_mean.pt").to(data.device)
    return data * traj_std + traj_mean

def hardCode_transform_traj(data):
    traj_std = torch.load( "./sample/info_traj_std.pt").to(data.device)
    traj_mean = torch.load( "./sample/info_traj_mean.pt").to(data.device)
    return (data - traj_mean) / traj_std

def hardCode_inv_transform(data):
    motion_std = torch.load( "./sample/info_motion_std.pt")
    motion_mean = torch.load( "./sample/info_motion_mean.pt")
    motion_std = torch.Tensor(motion_std).to(data.device)
    motion_mean = torch.Tensor(motion_mean).to(data.device)
    return data * motion_std + motion_mean

def create_fix_traj_in_vis(waypoints, L = 196):
    # convert trajectories into visulization mode
    # assert waypoints.shape == ( N(196), K(1), 3)
    # output shape (N, N*K, 3)
    N, K = waypoints.shape[:2]
    waypoints = waypoints.reshape((-1, 3))
    waypoints = waypoints.unsqueeze(0)
    waypoints = waypoints.repeat((L, 1, 1))
    return waypoints


def correct_foot_skid(foot_loc):

    HEIGHT_THRESHOLD = 0.02
    T, _, _ = foot_loc.shape
    corrected_foot_loc = foot_loc.clone()
    
    left_hight = torch.mean(foot_loc[:, [0, 1], :], dim=1)
    right_hight = torch.mean(foot_loc[:, [2, 3], :], dim=1)
    
    left_last_valid_loc = foot_loc[0, [0, 1], :]
    left_last_valid_loc[:, 2] = .0
    
    right_last_valid_loc = foot_loc[0, [2, 3], :]
    right_last_valid_loc[:, 2] = .0

    for t in range(T):
        if left_hight[t, 2] > HEIGHT_THRESHOLD:   
            left_last_valid_loc = foot_loc[t, [0, 1], :]
        else:
            corrected_foot_loc[t, [0, 1], :] = left_last_valid_loc
            corrected_foot_loc[t, [0, 1], 2] = .0
        
        if right_hight[t, 2] > HEIGHT_THRESHOLD:   
            right_last_valid_loc = foot_loc[t, [2, 3], :]
        else:
            corrected_foot_loc[t, [2, 3], :] = right_last_valid_loc
            corrected_foot_loc[t, [2, 3], 2] = .0 
    return corrected_foot_loc


def latent_fit(optimizer, smpl, source_kpts_model, static_vars, vp_model, extra_params={}, on_step=None, gstep=0, motionLen = 196, control_joints = [0], motion_traj_mask = None):

    data_loss = extra_params.get('data_loss', torch.nn.SmoothL1Loss(reduction='mean'))

    opt_map = [
        [0, 0], #root  
        [15, 1], #head
        [20, 2], #hand1    #left
        [21, 3], #hand2   #right
        [10, 4], #foot1   #left
        [11, 5], #foot2  #right
    ]
    control_joints = [0,1,2,3,4,5]
    opt_map = [opt_map[joint] for joint in control_joints]
    opt_jointNum = np.array(opt_map)[:, 1].tolist()
    opt_trajNum = np.array(opt_map)[:, 1].tolist()


    
    def fit(free_vars, motion_length, data_transform):
        fit.gstep += 1
        optimizer.zero_grad()
        
        pre_Joint = vp_model.vqvae.forward_decoder_from_quantized_codes(free_vars)
        
        sample = data_transform(pre_Joint[0].permute(1, 2, 0)).float()
        sample = recover_from_ric(sample, 22)[0, ...]
        sample[:, :, [1, 2]] = sample[:, :, [2, 1]]
        
        opt_objs = {}
        
        if motion_traj_mask is None:
            opt_objs['data'] = data_loss(sample[:motion_length, opt_jointNum,:], source_kpts_model['traj'][:motion_length,opt_trajNum,:].cuda()) #originally remove motion_length
        else:
            sample_masked = sample[:, [0, 15, 20, 21, 10, 11], :] * motion_traj_mask
            source_masked = source_kpts_model['traj'][:motion_length, ...].cuda() * motion_traj_mask[:motion_length, ...]
            opt_objs['data'] = data_loss(sample_masked[:motion_length, opt_jointNum,:], source_masked[:motion_length,opt_trajNum,:])
        
        #print(opt_objs['data'].item())
        sample_foot_target = correct_foot_skid( sample[:motion_length, [7, 10, 8, 11], :])
        opt_objs['foot'] = 1.0 * data_loss(sample[:motion_length, [7, 10, 8, 11], :], sample_foot_target)
        
        loss_total = torch.sum(torch.stack(list(opt_objs.values())))
        loss_total.backward(retain_graph=True)
        fit.free_vars = free_vars
        fit.final_loss = loss_total
        return loss_total

    
    fit.gstep = gstep
    fit.final_loss = None
    fit.free_vars = {}
    return fit

def joint_dist(a, b, mask, lengths):
    # assuming a.shape == b.shape == bs, J, Jdim, seqlen
    # assuming mask.shape == bs, 1, 1, seqlen

    diff = a - b
    distance_sq = torch.sum((a - b)** 2, dim=2)  # Shape: (bs, J, seq_len)
    distance = torch.sqrt(distance_sq + 1e-8)
    l2_dist_error = []
    for _idx, length in enumerate(lengths):
        l2_valid_dist = distance[_idx,:,:length] * mask[_idx,:,0,:length]
        l2_mean_dist = l2_valid_dist.sum() / mask[_idx,:,0,:length].sum()
        l2_dist_error.append(l2_mean_dist)

    # print('mse_loss_val', mse_loss_val)
    return torch.tensor(l2_dist_error)

def joint_dist_single(a, b, mask):
    # assuming a.shape == b.shape == bs, J, Jdim, seqlen
    # assuming mask.shape == bs, 1, 1, seqlen

    diff = a - b
    distance_sq = torch.sum((a - b)** 2, dim=2)  # Shape: (bs, J, seq_len)
    distance = torch.sqrt(distance_sq + 1e-8)

    
    l2_dist_error = []

    l2_valid_dist = distance * mask[:, :, 0]
    l2_mean_dist = l2_valid_dist.sum() / mask[:, :, 0].sum()
    l2_dist_error.append(l2_mean_dist)

    # print('mse_loss_val', mse_loss_val)
    return torch.tensor(l2_dist_error)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data class for generated motion by *conditioning*
class CompMDMGeneratedDatasetTlControl(Dataset):

    #def __init__(self, model_dict, diffusion_dict, dataloader, mm_num_samples, mm_num_repeats,
    #             max_motion_length, num_samples_limit, text_scale=1., keyframe_scale=1., save_dir=None, impute_until=0, skip_first_stage=False,
    #             seed=None, use_ddim=False, args=None):
    def __init__(self, net, transformer_traj_model, stage2_dataloader, mm_num_samples, mm_num_repeats, max_motion_length, num_samples_limit, scale=1., save_dir=None, seed=None, foot_phys=False, args=None):
    
        assert seed is not None, "must provide seed"
        self.dataloader = stage2_dataloader
        self.dataset = stage2_dataloader.dataset
        self.save_dir = save_dir
        self.args = args
        assert save_dir is not None
        assert mm_num_samples < len(stage2_dataloader.dataset)
        # This affect the trajectory model if we do two-stage, if not, it will affect the motion model
        # For trajectory model, the output traj will be imptued until 20 (set by impute_slack)
        # create the target directory
        os.makedirs(self.save_dir, exist_ok=True)

        clip_denoised = True
        self.max_motion_length = max_motion_length
        
        log_motion = False
        # guidance_mode = 'no'
        abs_3d = True

        # create the target directory
        os.makedirs(self.save_dir, exist_ok=True)


        self.max_motion_length = max_motion_length

        real_num_batches = len(stage2_dataloader)
        if num_samples_limit is not None:
            real_num_batches = num_samples_limit // stage2_dataloader.batch_size + 1
        print('real_num_batches', real_num_batches)

        generated_motion = []
        # NOTE: mm = multi-modal
        mm_generated_motions = []
        mm_motions = []

        if mm_num_samples > 0:
            mm_idxs = np.random.choice(real_num_batches, mm_num_samples // stage2_dataloader.batch_size +1, replace=False)
            mm_idxs = np.sort(mm_idxs)
        else:
            mm_idxs = []
        print('mm_idxs', mm_idxs)

        edit_mode = 'sparse_right_hand'


        endeffector_joints = stage2_dataloader.dataset.t2m_dataset.endeffector_joints

        with torch.no_grad():
            for i, (motion, model_kwargs) in tqdm(enumerate(stage2_dataloader)):
                '''For each datapoint, we do the following
                    1. Sample 3-10 (?) points from the ground truth trajectory to be used as conditions
                    2. Generate trajectory with trajectory model
                    3. Generate motion based on the generated traj using inpainting and cond_fn.
                '''

                if num_samples_limit is not None and len(generated_motion) >= num_samples_limit:
                    break

                tokens = [t.split('_') for t in model_kwargs['y']['tokens']]

                gt_sample = motion.cpu().permute(0, 2, 3, 1)
                gt_poses = stage2_dataloader.dataset.t2m_dataset.inv_transform(gt_sample).float()
                
                # gt_poses = gt_poses * self.dataset.std + self.dataset.mean  # [bs, 1, 196, 263] # TODO: mean and std are absolute mean and std and this is done on purpose! Why?  dataset: The 'eval' is here because we want inv_transform to work the same way at inference for model with abs3d,regradless of which dataset is loaded.
                # TODO: gt_poses = gt_poses * self.dataset.std_rel + self.dataset.mean_rel
                # (x,y,z) [bs, 1, 120, njoints=22, nfeat=3]
                gt_skel_motions = recover_from_ric(gt_poses.float(), 22, abs_3d=True)
                gt_skel_motions = gt_skel_motions.view(-1, *gt_skel_motions.shape[2:]).permute(0, 2, 3, 1)

                

                joint_pcd = o3d.geometry.PointCloud()
                joint_pcd.points = o3d.utility.Vector3dVector(gt_skel_motions.permute(0,3,1,2).cpu().numpy().reshape(-1,3))
                joint_pcd.paint_uniform_color([1, 0.706, 0])
                o3d.io.write_point_cloud(os.path.join("gt_skel.pcd"), joint_pcd)

                waypoints = hardCode_transform_traj(gt_skel_motions[:, [0, 15, 20, 21, 10, 11]].permute(0, 3, 1, 2))


                obs_joint_mask = torch.zeros((stage2_dataloader.batch_size, 196, 6, 3), dtype=bool, device=waypoints.device)


                for bs_i in range(stage2_dataloader.batch_size):
                    length = model_kwargs['y']['lengths'][bs_i].item()
                    gt_indices = np.array(range(length)[::self.args.transition_length])

                    assert edit_mode == 'sparse_right_hand'
                    
                    for _, frame_indices in enumerate(gt_indices):
                        num_joints = np.random.randint(1, 6)
                        joints_indices = np.random.choice(6, num_joints, replace=False)
                        
                        #obs_joint_mask[bs_i, frame_indices, joints_indices, :] = True  # set keyframes
                        obs_joint_mask[bs_i, frame_indices, 4, :] = True  # set keyframes
                        #pdb.set_trace()
                    #obs_joint_mask[bs_i, 0, :, gt_indices] = True  # set keyframes

                    obs_joint_mask[bs_i, length:, :, :] = True  # set keyframes

                waypoints_traj = waypoints * obs_joint_mask

                

                with torch.no_grad():
                    _, pre_codes = transformer_traj_model(waypoints_traj.to(device), model_kwargs['y']['text']) # get inital codes

                #####################
                # Two ways of picking codes from distribution:
                codes_pick = torch.argmax(F.softmax(pre_codes, dim = -1), dim = -1)
                x_quantized_fromIds = []
                for bs_i in range(stage2_dataloader.batch_size):

                    x_quantized_fromIds.append(net.vqvae.get_x_quantized_from_x_ids(codes_pick[[bs_i]].permute(0, 2, 1)))

                    
                waypoints = hardCode_inv_transform_traj(waypoints)

                target_waypoints = waypoints.clone()

                waypoints[:, :, :, [1, 2]] = waypoints[:, :, :, [2, 1]]

                import time
                t0 = time.time()
                joint_error = []
                cur_motion = []
                all_samples = []


                for bs_i in range(stage2_dataloader.batch_size):
                    goal_dict = {"id": [0, 1, 2, 3, 4, 5],
                                "traj": waypoints[bs_i, :, :,:] }


                    # Prepare for Optimization
                    x_quantized_init = x_quantized_fromIds[bs_i]
                    free_vars = []
                    for ele in x_quantized_init:
                        ele = ele.detach()
                        ele.requires_grad = True
                        free_vars.append(ele)

                    
                    optimizer = torch.optim.LBFGS(free_vars,
                                                lr=0.1,
                                                max_iter=1000,
                                                tolerance_change= 1e-6,#1e-10, #1e-30,
                                                max_eval=None,
                                                history_size= 200,
                                                line_search_fn='strong_wolfe')
                    # Optimize
                    gstep = 0

                    control_Joints = [0,1,2,3,4,5]

                    frame_len = model_kwargs['y']['lengths'][bs_i].item()

                    #motion_traj_mask = torch.zeros_like(waypoints_traj[bs_i]).cuda()
                    #motion_traj_mask[::30, control_Joints, :] = 1.

                    motion_traj_mask = obs_joint_mask[bs_i].float().to(device)


                    

                    closure = latent_fit(optimizer,
                                    smpl = None,
                                    source_kpts_model=goal_dict,
                                    static_vars=None,
                                    vp_model=net,
                                    on_step=None,
                                    gstep=gstep,
                                    motionLen = frame_len,
                                    control_joints=control_Joints,
                                    motion_traj_mask = motion_traj_mask)
                    
                    optimizer.step(lambda: closure(free_vars, motion_length = frame_len, data_transform = hardCode_inv_transform))
                    free_vars = closure.free_vars
                    #print("optimization done.")

                    

                    with torch.no_grad():
                        output_all = net.vqvae.forward_decoder_from_quantized_codes(free_vars)
                        output_all = output_all.cpu()
                    
                    pre_Joint = output_all.cpu()
                    sample = hardCode_inv_transform(pre_Joint[0].permute(1, 2, 0)).float()

                    
                    sample = recover_from_ric(sample, 22)[0, ...]


                    joint_pcd = o3d.geometry.PointCloud()
                    joint_pcd.points = o3d.utility.Vector3dVector(sample[:frame_len].cpu().numpy().reshape(-1,3))
                    joint_pcd.paint_uniform_color([1, 0.706, 0])
                    o3d.io.write_point_cloud(os.path.join("csampels.pcd"), joint_pcd)

                    joint_pcd = o3d.geometry.PointCloud()
                    joint_pcd.points = o3d.utility.Vector3dVector(target_waypoints[bs_i, :frame_len, :,:].detach().cpu().numpy().reshape(-1,3))
                    joint_pcd.paint_uniform_color([1, 0.706, 0])
                    o3d.io.write_point_cloud(os.path.join("target_waypoints.pcd"), joint_pcd)

                    all_samples.append(sample)
                    cur_motion.append(sample)
                    
                    
                    j_error = joint_dist_single(sample[:frame_len, [0,15,20,21,10,11]].detach().cpu(), target_waypoints[bs_i, :frame_len, :,:].detach().cpu(), motion_traj_mask[:frame_len].detach().cpu())
                    joint_error.append(j_error)
                    #print('j_error :', j_error.item())

                    
                t1 = time.time()
                print("---------------------------------------------------------------")
                print("time", (t1-t0)/stage2_dataloader.batch_size)



                lengths = model_kwargs['y']['lengths']
                skate_ratio_list = []
                for _idx, length in enumerate(lengths.tolist()):
                    skate_ratio, skate_vel = calculate_skating_ratio(cur_motion[_idx].permute(1,2,0).unsqueeze(0))
                    skate_ratio_list.append(skate_ratio)

                print('skate_ratio: {}'.format(np.array(skate_ratio_list).mean()))
                print('skate_ratio_max: {}'.format(np.array(skate_ratio_list).max()))


                joint_error = torch.tensor(joint_error)
                joint_error = torch.where(torch.isnan(joint_error), torch.tensor(1.0), joint_error)


                print("joint_error: {}", joint_error.mean().item())
                #motion = abs3d_to_rel(torch.stack(all_samples).permute(0,2,3,1), self.dataset, None)

                #pdb.set_trace()

                motion = abs3d_to_rel(torch.stack(all_samples).permute(0,2,3,1), self.dataset, None)

                #motion = abs3d_to_rel(gt_skel_motions, self.dataset, None)

                joint_pcd = o3d.geometry.PointCloud()
                joint_pcd.points = o3d.utility.Vector3dVector(torch.stack(all_samples).detach().cpu().numpy().reshape(-1,3))
                joint_pcd.paint_uniform_color([1, 0.706, 0])
                o3d.io.write_point_cloud(os.path.join("samples.pcd"), joint_pcd)

                #pdb.set_trace()

                base_folder = 'tlcontrol_test_motion'

                if edit_mode is not None:
                    base_folder += "_" + str(edit_mode)

                cur_motion = torch.stack(cur_motion).permute(0,2,3,1)
                for n_idx in range(32):
                        
                        
                    motion_len = model_kwargs['y']['lengths'][n_idx].item()
                    
                    if (torch.norm(cur_motion[n_idx,0].permute(1,0)[motion_len-1][[0,2]]) > 0.5) and (i>11):
                        os.makedirs('{}/{}_{}'.format(base_folder, i, n_idx), exist_ok=True)

                        text_description = model_kwargs['y']['text'][n_idx]



                        cur_motion[n_idx,[0,10,11,20,21,15],:,:motion_len].permute(2,0,1)


                        # Open the file in write mode and save the text
                        with open('{}/{}_{}/{}'.format(base_folder, i, n_idx,text_description), 'w') as file:
                            file.write(text_description)

                        joint_pcd = o3d.geometry.PointCloud()
                        joint_pcd.points = o3d.utility.Vector3dVector(cur_motion[n_idx,[0,10,11,20,21,15],:,:motion_len].permute(2,0,1).cpu().numpy().reshape(-1,3))
                        joint_pcd.paint_uniform_color([0.7, 1.0, 0])
                        o3d.io.write_point_cloud(os.path.join("{}/{}_{}/output_endeffector_joints.pcd".format(base_folder, i, n_idx)), joint_pcd)

                        points = np.asarray(joint_pcd.points)  # Convert Open3D point cloud to NumPy array
                        npy_path = "{}/{}_{}/endeffectors.npy".format(base_folder, i, n_idx)
                        np.save(npy_path, points)


                        j2s = joints2smpl(num_frames=motion_len , device_id=0, cuda=True, load_smplx=True)

                        motion_tensor, opt_dict = j2s.joint2smpl(cur_motion[n_idx].permute(2,0,1)[:motion_len ,:22].cpu().numpy())
                                        
                        transl = motion_tensor[:,-1,:].float()
                        #root_trans = opt_dict['cam']
                        global_orient = motion_tensor[:,0,:].float()
                        body_pose = motion_tensor[:,1:-1,:].float()
                        betas = opt_dict['betas'].float()

                        body_params = {'body_pose': body_pose.detach().cpu().float(), 'transl': transl.detach().cpu().float(), 'global_orient': global_orient.detach().cpu().float(), 'betas': None}
                        
                        
                        body_parms_file = os.path.join('{}/{}_{}/ours_conditioning.pickle'.format(base_folder, i, n_idx))

                        import pickle
                        with open(body_parms_file,'wb') as fw:
                            pickle.dump(body_params, fw)
                        
                        
                    

                #if t == 0:

                sub_dicts = []
                for bs_i in range(stage2_dataloader.batch_size):
                    joint_err = joint_error[bs_i].cpu().numpy()
                    
                    # joint_error가 NaN이면 이전 bs_i-1 값을 사용
                    if np.isnan(joint_err).any() and bs_i > 0:
                        joint_err = sub_dicts[-1]['joint_error']
                    
                    sub_dicts.append({
                        'motion': motion[bs_i].squeeze().permute(1, 0).cpu().numpy(),
                        'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy() - 1,  # modified to only contain valid ones
                        'caption': model_kwargs['y']['text'][bs_i],
                        'tokens': tokens[bs_i],
                        'cap_len': len(tokens[bs_i]),
                        'skate_ratio': skate_ratio_list[bs_i],
                        'joint_error': joint_error[bs_i].cpu().numpy(),
                    })


                #sub_dicts = [{'motion': motion[bs_i].squeeze().permute(1,0).cpu().numpy(),
                #            #'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy(),
                #            'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy() - 1, # modified to only contain valid ones
                #            'caption': model_kwargs['y']['text'][bs_i],
                #            'tokens': tokens[bs_i],
                #            'cap_len': len(tokens[bs_i]),
                #            #'dist_error': kps_error[bs_i].cpu().numpy(),
                #            'skate_ratio': skate_ratio_list[bs_i],
                #            'joint_error': joint_error[bs_i].cpu().numpy(),
                #            #'keyframe_error': keyframe_error[bs_i].cpu().numpy(),
                #            #'num_keyframes': len(keyframes_indices[bs_i]) if keyframes_indices[bs_i] is not None else 0,
                #            } for bs_i in range(stage2_dataloader.batch_size)]
                generated_motion += sub_dicts


                #if is_mm:
                mm_motions += [{'motion': motion[bs_i].squeeze().permute(1, 0).cpu().numpy(),
                                #'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy(),
                                'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy() - 1, # modified to only contain valid ones
                                } for bs_i in range(stage2_dataloader.batch_size)]

            #if is_mm:
                mm_generated_motions += [{
                                'caption': model_kwargs['y']['text'][bs_i],
                                'tokens': tokens[bs_i],
                                'cap_len': len(tokens[bs_i]),
                                'mm_motions': mm_motions[bs_i::stage2_dataloader.batch_size],  # collect all 10 repeats from the (32*10) generated motions
                                } for bs_i in range(stage2_dataloader.batch_size)]


        self.generated_motion = generated_motion
        self.mm_generated_motion = mm_generated_motions
        self.w_vectorizer = stage2_dataloader.dataset.w_vectorizer


    def __len__(self):
        return len(self.generated_motion)


    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion, m_length, caption, tokens = data['motion'], data['length'], data['caption'], data['tokens']
        dist_error = data['joint_error']
        skate_ratio = data['skate_ratio']
        sent_len = data['cap_len']
        #keyframe_error = data['keyframe_error']
        #num_keyframes = data['num_keyframes']

        if self.dataset.mode == 'eval':
            normed_motion = motion
            if self.dataset.absolute_3d:
                # Denorm with rel_transform because the inv_transform() will have the absolute mean and std
                # The motion is already converted to relative after inference
                denormed_motion = (normed_motion * self.dataset.std_rel) + self.dataset.mean_rel
            else:
                denormed_motion = self.dataset.t2m_dataset.inv_transform(normed_motion)
            renormed_motion = (denormed_motion - self.dataset.mean_for_eval) / self.dataset.std_for_eval  # according to T2M norms
            motion = renormed_motion
            # This step is needed because T2M evaluators expect their norm convention

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), dist_error, skate_ratio #, keyframe_error, num_keyframes


    def get_keyframe_indices(self, keyframes_mask):
        keyframe_indices = []
        for sample_i in range(keyframes_mask.shape[0]):
            keyframe_indices.append([int(e) for e in keyframes_mask[sample_i].sum(dim=0).squeeze().nonzero().squeeze(-1)])
        return keyframe_indices


    def set_inference_editing_args(self, model_kwargs, input_motions):
        """ Set arguments for inference-time editing according to edit.py

        Args:
            model_kwargs (dict): arguments for the model
            input_motions (torch.tensor): ground-truth motion with absolute-root representation

        Returns:
            torch.tensor: keyframe_indices
            torch.tensor: joint_mask
        """
        model_kwargs['y']['inpainted_motion'] = input_motions
        model_kwargs['y']['imputate'] = self.args.imputate
        model_kwargs['y']['replacement_distribution'] = self.args.replacement_distribution
        model_kwargs['y']['reconstruction_guidance'] = self.args.reconstruction_guidance
        model_kwargs['y']['reconstruction_weight'] = self.args.reconstruction_weight
        model_kwargs['y']['diffusion_steps'] = self.args.diffusion_steps
        model_kwargs['y']['gradient_schedule'] = self.args.gradient_schedule
        model_kwargs['y']['stop_imputation_at'] = self.args.stop_imputation_at
        model_kwargs['y']['stop_recguidance_at'] = self.args.stop_recguidance_at

        # if args.text_condition == '':
        #     args.guidance_param = 0.  # Force unconditioned generation

        model_kwargs['y']['inpainting_mask'], joint_mask = get_keyframes_mask(data=model_kwargs['y']['inpainted_motion'],
                                                                                lengths=model_kwargs['y']['lengths'],
                                                                                edit_mode=self.args.edit_mode,
                                                                                trans_length=self.args.transition_length,
                                                                                feature_mode=self.args.editable_features,
                                                                                get_joint_mask=True, n_keyframes=self.args.n_keyframes)

        return self.get_keyframe_indices(model_kwargs['y']['inpainting_mask']), joint_mask


    def set_conditional_synthesis_args(self, model_kwargs, input_motions, data_rep):
        """ Set arguments for conditional sampling according to conditional_synthesis.py

        Args:
            model_kwargs (dict): arguments for the model
            input_motions (torch.tensor): ground-truth motion with absolute-root representation

        Returns:
            torch.tensor: keyframe_indices
            torch.tensor: joint_mask
        """

        model_kwargs['obs_x0'] = input_motions.to("cuda")
        if data_rep == "glo_root":
            edit_mode = 'benchmark_sparse_endeffectors_with_root'
        elif data_rep == "glo_joints":
            edit_mode = 'benchmark_sparse_endeffectors'
        model_kwargs['obs_mask'], joint_mask = get_keyframes_mask(data=input_motions, lengths=model_kwargs['y']['lengths'], edit_mode=edit_mode, #self.args.edit_mode,
                                                                  feature_mode='pos', trans_length=self.args.transition_length,
                                                                  get_joint_mask=True, n_keyframes=self.args.n_keyframes)
        
        model_kwargs['obs_mask'] = model_kwargs['obs_mask'].to("cuda")
        joint_mask = joint_mask.to("cuda") 
        


        return self.get_keyframe_indices(model_kwargs['obs_mask']), joint_mask
