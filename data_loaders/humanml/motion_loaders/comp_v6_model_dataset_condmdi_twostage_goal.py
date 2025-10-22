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
from data_loaders.humanml.scripts.motion_process import recover_from_ric, convert_to_humanml3d
from data_loaders.humanml.utils.metrics import calculate_skating_ratio
from sample.gmd.condition import (cond_fn_key_location, get_target_from_kframes, get_target_and_inpt_from_kframes_batch,
                              log_trajectory_from_xstart, get_inpainting_motion_from_traj, get_inpainting_motion_from_gt,
                              cond_fn_key_location, compute_kps_error, cond_fn_sdf,
                              CondKeyLocations, CondKeyLocationsWithSdf, compute_kps_error_arbitrary)
from utils.editing_util import get_keyframes_mask
import open3d as o3d
import pdb


from visualize.simplify_loc2rot import joints2smpl

from diffusion.nn import mean_flat, sum_flat
from math import sqrt

l2_loss = lambda a, b: (
            a - b
        )**2 

def masked_l2(a, b, mask):
    # assuming a.shape == b.shape == bs, J, Jdim, seqlen
    # assuming mask.shape == bs, 1, 1, seqlen
    loss = l2_loss(a, b)
    loss = sum_flat(loss *mask.float())  # gives \sigma_euclidean over unmasked elements
    n_entries = a.shape[1] * a.shape[2]
    non_zero_elements = sum_flat(mask) * n_entries
    # print('mask', mask.shape)
    # print('non_zero_elements', non_zero_elements)
    # print('loss', loss)
    mse_loss_val = loss / non_zero_elements
    # print('mse_loss_val', mse_loss_val)
    return mse_loss_val

def joint_dist(a, b, mask, lengths):
    # assuming a.shape == b.shape == bs, J, Jdim, seqlen
    # assuming mask.shape == bs, 1, 1, seqlen

    diff = a - b
    distance_sq = torch.sum((a - b)** 2, dim=2)  # Shape: (bs, J, seq_len)
    distance = torch.sqrt(distance_sq + 1e-8)
    l2_dist_error = []
    l2_start_error = []
    l2_goal_error = []
    masks = []
    for _idx, length in enumerate(lengths):
        l2_valid_dist = distance[_idx,:,:length] * mask[_idx,:,0,:length]
        l2_mean_dist = l2_valid_dist.sum() / mask[_idx,:,0,:length].sum()
        l2_dist_error.append(l2_mean_dist)

        #l2_start_error.append(l2_valid_dist[:,0][mask[_idx,:,0,0]])
        #l2_goal_error.append(l2_valid_dist[:,length-1][mask[_idx,:,0,length-1]])

        l2_start_error.append(l2_valid_dist[:,0])
        l2_goal_error.append(l2_valid_dist[:,length-1])
        masks.append(mask[_idx])


    # print('mse_loss_val', mse_loss_val)
    return torch.tensor(l2_dist_error), l2_start_error, l2_goal_error, masks

# Data class for generated motion by *conditioning*
class CompMDMGeneratedDatasetCondTwoStageGoal(Dataset):

    def __init__(self, stage1_model, stage1_diffusion, stage2_model, stage2_diffusion, stage1_dataloader, stage2_dataloader, mm_num_samples, mm_num_repeats, max_motion_length, num_samples_limit, scale=1., save_dir=None, seed=None, foot_phys=False, args=None):
        assert seed is not None, "seed must be provided"
        self.dataloader = stage1_dataloader
        self.dataset = stage1_dataloader.dataset
        self.save_dir = save_dir
        self.args = args
        assert save_dir is not None
        assert mm_num_samples < len(stage1_dataloader.dataset)

        # create the target directory
        os.makedirs(self.save_dir, exist_ok=True)

        use_ddim = False  # FIXME - hardcoded
        # NOTE: I have updated the code in gaussian_diffusion.py so that it won't clip denoise for xstart models.
        # hence, always set the clip_denoised to True
        clip_denoised = True
        self.max_motion_length = max_motion_length
        stage1_sample_fn = (
            stage1_diffusion.p_sample_loop if not use_ddim else stage1_diffusion.ddim_sample_loop
        )
        stage2_sample_fn = (
            stage2_diffusion.p_sample_loop if not use_ddim else stage2_diffusion.ddim_sample_loop
        )

        ### Basic settings
        # motion_classifier_scale = 100.0
        # print("motion classifier scale", motion_classifier_scale)
        log_motion = False
        # guidance_mode = 'no'
        abs_3d = True
        use_random_proj = self.dataset.use_rand_proj
        # print("guidance mode", guidance_mode)


        # create the target directory
        os.makedirs(self.save_dir, exist_ok=True)

        # use_ddim = False  # FIXME - hardcoded
        # NOTE: I have updated the code in gaussian_diffusion.py so that it won't clip denoise for xstart models.
        # hence, always set the clip_denoised to True
        # clip_denoised = True
        self.max_motion_length = max_motion_length

        # sample_fn_motion = (
        #     motion_diffusion.p_sample_loop if not use_ddim else motion_diffusion.ddim_sample_loop
        # )

        real_num_batches = len(stage1_dataloader)
        if num_samples_limit is not None:
            real_num_batches = num_samples_limit // stage1_dataloader.batch_size + 1
        print('real_num_batches', real_num_batches)

        generated_motion = []
        # NOTE: mm = multi-modal
        mm_generated_motions = []
        if mm_num_samples > 0:
            mm_idxs = np.random.choice(real_num_batches, mm_num_samples // stage1_dataloader.batch_size +1, replace=False)
            mm_idxs = np.sort(mm_idxs)
        else:
            mm_idxs = []
        print('mm_idxs', mm_idxs)

        stage1_model.eval()
        stage2_model.eval()

        stage1_data_rep = stage1_dataloader.dataset.t2m_dataset.data_rep
        endeffector_dim = stage1_dataloader.dataset.t2m_dataset.endeffector_dim
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

                #tokens = [t.split('_') for t in model_kwargs['y']['tokens']]

                model_kwargs['y']['body'] = model_kwargs['y']['body'].to("cuda")
                # add CFG scale to batch
                # add CFG scale to batch
                if scale != 1.:
                    model_kwargs['y']['scale'] = torch.ones(motion.shape[0],
                                                            device=dist_util.dev()) * scale
                    model_kwargs['y']['text_scale'] = torch.ones(motion.shape[0],
                                                            device=dist_util.dev()) * scale

                
                # Convert to 3D motion space
                # NOTE: the 'motion' will not be random projected if dataset mode is 'eval' or 'gt',
                # even if the 'self.dataset.t2m_dataset.use_rand_proj' is True
                # NOTE: the 'motion' will have relative representation if dataset mode is 'eval' or 'gt',
                # even if the 'self.dataset.t2m_dataset.use_abs3d' is True
                gt_sample = motion.cpu().permute(0, 2, 3, 1)
                gt_poses = stage2_dataloader.dataset.t2m_dataset.inv_transform(gt_sample).float()

                endeffector_dim = stage1_dataloader.dataset.t2m_dataset.endeffector_dim
                
                # gt_poses = gt_poses * self.dataset.std + self.dataset.mean  # [bs, 1, 196, 263] # TODO: mean and std are absolute mean and std and this is done on purpose! Why?  dataset: The 'eval' is here because we want inv_transform to work the same way at inference for model with abs3d,regradless of which dataset is loaded.
                # TODO: gt_poses = gt_poses * self.dataset.std_rel + self.dataset.mean_rel
                # (x,y,z) [bs, 1, 120, njoints=22, nfeat=3]
                gt_skel_motions = recover_from_ric(gt_poses.float(), 22, abs_3d=True)


                if stage1_data_rep == "glo_joints":
                    glo_joints_std, glo_joints_mean = stage1_dataloader.dataset.t2m_dataset.get_mean_std_stage1()
                    motion = torch.cat([torch.tensor(gt_poses[:, :, :, :4]), gt_skel_motions[:, :, :, 1:, :].reshape(gt_skel_motions.shape[0], 1, gt_skel_motions.shape[2], -1)], dim=-1)[:, :, :, stage1_dataloader.dataset.t2m_dataset.endeffector_dim]
                    motion = motion.permute(0,3,1,2)
                    motion = (motion - glo_joints_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)) / glo_joints_std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

                gt_skel_motions = gt_skel_motions.view(-1, *gt_skel_motions.shape[2:]).permute(0, 2, 1, 3)
                # gt_skel_motions.shape
                # torch.Size([32, 22, 3, 196])

                
                joint_pcd = o3d.geometry.PointCloud()
                joint_pcd.points = o3d.utility.Vector3dVector(gt_skel_motions[0].permute(2,0,1)[:model_kwargs['y']['lengths'][0].item() ,:22].cpu().numpy().reshape(-1,3))
                joint_pcd.paint_uniform_color([0.7, 1.0, 0])
                o3d.io.write_point_cloud(os.path.join("output_debug_joints.pcd"), joint_pcd)

                

                ### START OUR BUILD OF model_kwargs ###

                if stage1_data_rep == "glo_root":
                    motion_input = motion[:,stage1_dataloader.dataset.t2m_dataset.endeffector_dim]
                elif stage1_data_rep == "glo_joints":
                    motion_input = motion

                assert stage1_model.keyframe_conditioned


                fixed_set = model_kwargs['y']['select_endeffector_dim']
                
                if stage1_model.keyframe_conditioned:

                    # Conditional synthesis arguments:
                    keyframes_indices, joint_mask = self.set_conditional_synthesis_args(model_kwargs, motion_input, stage1_data_rep, fixed_set)
                ### END OUR BUILD OF model_kwargs ###

                full_joint_mask = torch.zeros((motion.shape[0], 22, stage1_model.nfeats, motion.shape[-1]), dtype=bool, device=joint_mask.device)
                
                full_joint_mask[:,endeffector_joints] = joint_mask

                mm_num_now = len(mm_generated_motions) // stage2_dataloader.batch_size
                is_mm = i in mm_idxs
                repeat_times = mm_num_repeats if is_mm else 1
                mm_motions = []
                mm_trajectories = []
                for t in range(repeat_times):
                    seed_number = seed * 100_000 + i * 100 + t
                    fixseed(seed_number)
                    batch_file = f'{i:04d}_{t:02d}.pt'
                    batch_path = os.path.join(self.save_dir, batch_file)

                    model_kwargs['y']['stage'] = 1 
                    model_kwargs['y']['foot_phys'] = foot_phys

                    # reusing the batch if it exists
                    # if os.path.exists(batch_path):
                    if False:  # GUY - IGNORE CACHE FOR NOW
                        # [bs, njoints, nfeat, seqlen]
                        sample_motion = torch.load(batch_path, map_location=motion.device)
                        print(f'batch {batch_file} exists, loading from file')
                    else:
                        print(f'working on {batch_file}')
                        # for smoother motions
                        # impute_slack = 20
                        # NOTE: For debugging
                        # traj_model_kwargs['y']['log_name'] = self.save_dir
                        # traj_model_kwargs['y']['log_id'] = i
                        model_kwargs['y']['log_name'] = self.save_dir
                        model_kwargs['y']['log_id'] = i
                        

                        sample = stage1_sample_fn(
                            stage1_model,
                            (motion.shape[0], stage1_model.njoints, stage1_model.nfeats, motion.shape[-1]),
                            clip_denoised=clip_denoised,
                            model_kwargs=model_kwargs,
                            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                            init_image=None,
                            progress=True,
                            dump_steps=None,
                            noise=None,
                            const_noise=False,
                            # when experimenting guidance_scale we want to nutrileze the effect of noise on generation
                        )

                        sample = sample.cpu().permute(0, 2, 3, 1)
                
                        joints = stage1_dataloader.dataset.t2m_dataset.inv_transform_stage1(sample).float()
                        joints = joints.view(-1, *joints.shape[2:]).permute(0, 2, 1) # batch_size, n_joints=22, 3, n_frames


                        if stage1_data_rep  == "glo_joints":
                            joint_pos = torch.cat([recover_from_ric(joints[:,:,:].permute(0,2,1), joints_num=1, abs_3d=args.abs_3d), joints[:,4:,:].permute(0,2,1).reshape(joints.shape[0],joints.shape[-1],-1,3)], dim=2).permute(0, 2, 3, 1)
                        else:
                            joint_pos = recover_from_ric(joints[:,:,:].permute(0,2,1), joints_num=stage1_dataloader.dataset.t2m_dataset.select_joints_num, abs_3d=True).permute(0, 2, 3, 1)

                        model_kwargs['y']['stage'] = 2

                        motion = torch.zeros(*sample.shape[:3], 263, device=sample.device, dtype=torch.float)
                        # Define the end-effector dimensions you want to set from sample
                        #endeffector_dim = [0,1,2,3,31,32,33,34,35,36,61,62,63,64,65,66]
                        


                        # Now copy the corresponding channels from 'sample' into 'motion'
                        if stage1_data_rep  == "glo_joints":
                            glo_root_std, glo_root_mean = stage1_dataloader.dataset.t2m_dataset.get_mean_std_stage1_glo_root()
                            for _sample_idx in range(joints.shape[0]):
                                glo_root_sample = convert_to_humanml3d(joint_pos[_sample_idx].detach().permute(2,0,1).numpy(), joints[[_sample_idx],:,:].detach().permute(0,2,1), feet_thre=0.002, abs_3d=True).unsqueeze(0)
                                motion[_sample_idx, :, :, endeffector_dim] = (glo_root_sample - glo_root_mean.unsqueeze(0).unsqueeze(0).unsqueeze(0)) / glo_root_std.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                            
                        else:
                            motion[:, :, :, endeffector_dim] = sample
    
                        
                        motion = motion.permute(0, 3, 1, 2).to('cuda')

                        model_kwargs['obs_x0'] = motion # [bs, n_joints=263, n_feats=1, n_frames=224]
                        model_kwargs['obs_mask'], obs_joint_mask  = get_keyframes_mask(data=motion, lengths=model_kwargs['y']['lengths'], edit_mode='all_endeffectors', get_joint_mask=True) # [bs, n_joints=263, nfeats=1, n_frames=224]

                        sample = stage2_sample_fn(
                                stage2_model,
                                (motion.shape[0], stage2_model.njoints, stage2_model.nfeats, motion.shape[-1]),
                                clip_denoised=clip_denoised,
                                model_kwargs=model_kwargs,
                                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                                init_image=None,
                                progress=True,
                                dump_steps=None,
                                noise=None,
                                const_noise=False,
                                # when experimenting guidance_scale we want to nutrileze the effect of noise on generation
                            )

                        n_joints = 22 if (sample.shape[1] in [263, 264]) else 21
                        sample = sample.cpu().permute(0, 2, 3, 1)
                        sample = stage2_dataloader.dataset.t2m_dataset.inv_transform(sample).float()

                        

                        cur_motion = recover_from_ric(sample[:,0], n_joints, abs_3d=True)
                        cur_motion = cur_motion.view(-1, *cur_motion.shape[1:]).permute(0, 2, 3, 1) # batch_size, n_joints=22, 3, n_frames


                        torch.save(cur_motion, batch_path)

                        lengths = model_kwargs['y']['lengths']
                        skate_ratio_list = []
                        for _idx, length in enumerate(lengths.tolist()):
                            skate_ratio, skate_vel = calculate_skating_ratio(cur_motion[_idx:_idx+1,:,:,:length])
                            skate_ratio_list.append(skate_ratio)

                        print('skate_ratio: {}'.format(np.array(skate_ratio_list).mean()))
                        
                        joint_recon_error_max_list = []
                        joint_recon_error_mean_list = []

                        

                        #for _idx, length in enumerate(lengths.tolist()):
#
                        #    joint_recon_error_mean = (cur_motion[_idx:_idx+1,endeffector_joints,:,:length] - joint_pos[_idx:_idx+1,:,:,:length]).abs().mean().item()
                        #    joint_recon_error_max = (cur_motion[_idx:_idx+1,endeffector_joints,:,:length] - joint_pos[_idx:_idx+1,:,:,:length]).abs().max().item()
#
                        #    joint_recon_error_max_list.append(joint_recon_error_max)
                        #    joint_recon_error_mean_list.append(joint_recon_error_mean)
#
                        #print('recon_error_max: {}'.format(np.array(joint_recon_error_max_list).mean()))
                        #print('recon_error_mean: {}'.format(np.array(joint_recon_error_mean_list).mean()))

                        for _idx, length in enumerate(lengths.tolist()):

                            

                            dist = (cur_motion[_idx,[10,11,20,21,15],:,:length] - joints[_idx,4:,:length].reshape(5,3,-1))
                            dist = torch.sqrt(dist[:,0]**2 + dist[:,0]**2 + dist[:,2]**2)

                            joint_recon_error_mean = dist.abs().mean().item()
                            joint_recon_error_max = dist.abs().max().item()


                            joint_recon_error_max_list.append(joint_recon_error_max)
                            joint_recon_error_mean_list.append(joint_recon_error_mean)

                        print('recon_error_max: {}'.format(np.array(joint_recon_error_max_list).mean()))
                        print('recon_error_mean: {}'.format(np.array(joint_recon_error_mean_list).mean()))


                        save_motion = True
                        if save_motion:
                            n_idx = 7
                            joint_pcd = o3d.geometry.PointCloud()
                            joint_pcd.points = o3d.utility.Vector3dVector(joint_pos[n_idx].permute(2,0,1)[:model_kwargs['y']['lengths'][n_idx].item() ,:22].cpu().numpy().reshape(-1,3))
                            joint_pcd.paint_uniform_color([0.7, 1.0, 0])
                            o3d.io.write_point_cloud(os.path.join("output_endeffector_joints.pcd"), joint_pcd)

                            joint_pcd = o3d.geometry.PointCloud()
                            joint_pcd.points = o3d.utility.Vector3dVector(cur_motion[n_idx].permute(2,0,1)[:model_kwargs['y']['lengths'][n_idx].item() ,:22].cpu().numpy().reshape(-1,3))
                            joint_pcd.paint_uniform_color([1, 0.706, 0])
                            o3d.io.write_point_cloud(os.path.join("output_full_joints.pcd"), joint_pcd)

                        #if stage1_data_rep == "glo_joints":
                        gt_skel_motions = gt_skel_motions.permute(0,1,3,2)

                        joint_error, start_error, goal_error, masks = joint_dist(cur_motion.to("cuda"), gt_skel_motions.to("cuda"), full_joint_mask, lengths.tolist())
                    
                        #for n_idx in range(len(joint_error)):
                        #    task = model_kwargs['y']['task'][n_idx]
                        #    if task == "circle":
                        #        
                        #    elif task == "cimi4d":
#
                        #    elif task == "couch":

                    kps_error = compute_kps_error_arbitrary(cur_motion, gt_skel_motions, keyframes_indices, traj_only=True)
                    keyframe_error = compute_kps_error_arbitrary(cur_motion, gt_skel_motions, keyframes_indices, traj_only=False)


                    joint_error = torch.where(torch.isnan(joint_error), torch.tensor(0.0), joint_error)
                    kps_error = torch.where(torch.isnan(kps_error), torch.tensor(0.0), kps_error)
                    keyframe_error = torch.where(torch.isnan(keyframe_error), torch.tensor(0.0), keyframe_error)

                    print("joint_error: {}", joint_error.mean().item())


                    for n_idx in range(32):

                        motion_len = model_kwargs['y']['lengths'][n_idx].item()

                        os.makedirs('goal_test_motion/{}_{}'.format(i, n_idx), exist_ok=True)

                        text_description = model_kwargs['y']['task'][n_idx]

                        # Open the file in write mode and save the text
                        with open('goal_test_motion/{}_{}/{}'.format(i, n_idx,text_description), 'w') as file:
                            file.write(text_description)

                        error_description = str(joint_error[n_idx])
                        # Open the file in write mode and save the text
                        with open('goal_test_motion/{}_{}/{}'.format(i, n_idx,error_description), 'w') as file:
                            file.write(error_description)

                        j2s = joints2smpl(num_frames=motion_len , device_id=0, cuda=True, load_smplx=True)

                        motion_tensor, opt_dict = j2s.joint2smpl(cur_motion[n_idx].permute(2,0,1)[:motion_len ,:22].cpu().numpy())
                                        
                        transl = motion_tensor[:,-1,:].float()
                        #root_trans = opt_dict['cam']
                        global_orient = motion_tensor[:,0,:].float()
                        body_pose = motion_tensor[:,1:-1,:].float()
                        betas = opt_dict['betas'].float()

                        body_params = {'body_pose': body_pose.detach().cpu().float(), 'transl': transl.detach().cpu().float(), 'global_orient': global_orient.detach().cpu().float(), 'betas': None}
                        

                        body_parms_file = os.path.join('goal_test_motion/{}_{}/ours_conditioning.pickle'.format(i, n_idx))

                        import pickle
                        with open(body_parms_file,'wb') as fw:
                            pickle.dump(body_params, fw)
                        
                        motion_tensor, opt_dict = j2s.joint2smpl(gt_skel_motions[n_idx].permute(2,0,1)[:motion_len ,:22].cpu().numpy())
                                        
                        transl = motion_tensor[:,-1,:].float()
                        #root_trans = opt_dict['cam']
                        global_orient = motion_tensor[:,0,:].float()
                        body_pose = motion_tensor[:,1:-1,:].float()
                        betas = opt_dict['betas'].float()

                        body_params = {'body_pose': body_pose.detach().cpu().float(), 'transl': transl.detach().cpu().float(), 'global_orient': global_orient.detach().cpu().float(), 'betas': None}
                        
                        

                        body_parms_file = os.path.join('goal_test_motion/{}_{}/gt_conditioning.pickle'.format(i, n_idx))

                        import pickle
                        with open(body_parms_file,'wb') as fw:
                            pickle.dump(body_params, fw)

                        
        
                        joint_sig = gt_skel_motions[[n_idx],:,:,:motion_len].permute(0,1,3,2)[full_joint_mask[[n_idx],:,:,:motion_len].repeat(1,1,3,1).permute(0,1,3,2)].reshape(-1,3)

                        

                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(np.array(joint_sig.detach().cpu()).reshape(-1,3))
                        pcd.paint_uniform_color([1, 0.706, 0])
                        o3d.io.write_point_cloud("goal_test_motion/{}_{}/joints_signal.pcd".format(i, n_idx), pcd)

                        points = np.asarray(pcd.points)  # Convert Open3D point cloud to NumPy array
                        npy_path = "goal_test_motion/{}_{}/controL_joints.npy".format(i, n_idx)
                        np.save(npy_path, points)
                    

                    # print('cut the motion length from {} to {}'.format(sample_motion.shape[-1], self.max_motion_length))
                    #sample = sample_motion[:, :, :, :self.max_motion_length]

                    # Compute error for key xz locations
                    #cur_motion = sample_to_motion(sample, self.dataset, motion_model)
                    #kps_error = compute_kps_error(cur_motion, gt_skel_motions, keyframes)  # [batch_size, 5] in meter
                    
                    #print("keyframe_error {}", keyframe_error.mean().item())


                    # skate_ratio, skate_vel = calculate_skating_ratio(cur_motion)  # [batch_size]
                    # We can get the trajectory from here. Get only root xz from motion
                    cur_traj = cur_motion[:, 0, [0, 2], :]

                    # NOTE: To test if the motion is reasonable or not
                    log_motion = False
                    if log_motion:
                        from data_loaders.humanml.utils.plot_script import plot_3d_motion
                        for j in tqdm([1, 3, 4, 5], desc="generating motion"):
                            motion_id = f'{i:04d}_{t:02d}_{j:02d}'
                            plot_3d_motion(os.path.join(self.save_dir, f"motion_cond_{motion_id}.mp4"), self.dataset.kinematic_chain,
                            cur_motion[j].permute(2,0,1).detach().cpu().numpy(), 'title', 'humanml', fps=20)

                    if self.dataset.absolute_3d:
                        # NOTE: Changing the output from absolute space to the relative space here.
                        # The easiest way to do this is to go all the way to skeleton and convert back again.
                        # sample shape [32, 263, 1, 196]
                        n_idx = 7
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(np.array(cur_motion[n_idx,:,:,:model_kwargs['y']['lengths'][n_idx]].permute(2,0,1).detach().cpu()).reshape(-1,3))
                        pcd.paint_uniform_color([1, 0.706, 0])
                        o3d.io.write_point_cloud("joints.pcd", pcd)

                        #cur_motion = gt_skel_motions

                        sample = abs3d_to_rel(cur_motion, self.dataset, stage2_model)


                    if t == 0:
                        sub_dicts = [{'motion': sample[bs_i].squeeze().permute(1,0).cpu().numpy(),
                                    #'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy(),
                                    'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy() - 1, # modified to only contain valid ones
                                    'task': model_kwargs['y']['task'][bs_i], 
                                    'file': model_kwargs['y']['file'][bs_i],
                                    'start_error': start_error[bs_i].cpu().numpy(),
                                    'goal_error': goal_error[bs_i].cpu().numpy(), 
                                    'mask': masks[bs_i].cpu().numpy(),
                                    'dist_error': kps_error[bs_i].cpu().numpy(),
                                    'skate_ratio': skate_ratio_list[bs_i],
                                    'joint_error': joint_error[bs_i].cpu().numpy(),
                                    'keyframe_error': keyframe_error[bs_i].cpu().numpy(),
                                    'num_keyframes': len(keyframes_indices[bs_i]) if keyframes_indices[bs_i] is not None else 0,
                                    } for bs_i in range(stage2_dataloader.batch_size)]
                        generated_motion += sub_dicts

                    if is_mm:
                        mm_motions += [{'motion': sample[bs_i].squeeze().permute(1, 0).cpu().numpy(),
                                        #'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy(),
                                        'length': model_kwargs['y']['lengths'][bs_i].cpu().numpy() - 1, # modified to only contain valid ones
                                        'traj': cur_traj[bs_i].squeeze().permute(1, 0).cpu().numpy(),
                                        } for bs_i in range(stage2_dataloader.batch_size)]

                if is_mm:
                    mm_generated_motions += [{
                                    'mm_motions': mm_motions[bs_i::stage2_dataloader.batch_size],  # collect all 10 repeats from the (32*10) generated motions
                                    } for bs_i in range(stage2_dataloader.batch_size)]


        self.generated_motion = generated_motion
        self.mm_generated_motion = mm_generated_motions


    def __len__(self):
        return len(self.generated_motion)


    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion, m_length = data['motion'], data['length']
        dist_error = data['dist_error']
        skate_ratio = data['skate_ratio']
        keyframe_error = data['keyframe_error']
        joint_error = data['joint_error']
        start_error = data['start_error']
        goal_error = data['goal_error']
        num_keyframes = data['num_keyframes']
        mask = data['mask']
        task = data['task']
        file = data['file']

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


        return motion, m_length, dist_error, skate_ratio, keyframe_error, joint_error, num_keyframes, start_error, goal_error, mask, task, file

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


    def set_conditional_synthesis_args(self, model_kwargs, input_motions, data_rep, fixed_set):
        """ Set arguments for conditional sampling according to conditional_synthesis.py

        Args:
            model_kwargs (dict): arguments for the model
            input_motions (torch.tensor): ground-truth motion with absolute-root representation

        Returns:
            torch.tensor: keyframe_indices
            torch.tensor: joint_mask
        """
        assert self.args.editable_features == "pos"

        model_kwargs['obs_x0'] = input_motions.to("cuda")
        edit_mode = 'fixed'
        model_kwargs['obs_mask'], joint_mask = get_keyframes_mask(data=input_motions, lengths=model_kwargs['y']['lengths'], edit_mode=edit_mode, #self.args.edit_mode,
                                                                  feature_mode=self.args.editable_features, trans_length=self.args.transition_length,
                                                                  get_joint_mask=True, n_keyframes=self.args.n_keyframes, fixed_set=fixed_set)
        
        model_kwargs['obs_mask'] = model_kwargs['obs_mask'].to("cuda")
        joint_mask = joint_mask.to("cuda") 
        
        model_kwargs['y']['diffusion_steps'] = self.args.diffusion_steps
        # Add inpainting mask according to args
        if self.args.zero_keyframe_loss: # if loss is 0 over keyframes durint training, then must impute keyframes during inference
            model_kwargs['y']['imputate'] = 1
            model_kwargs['y']['stop_imputation_at'] = 0
            model_kwargs['y']['replacement_distribution'] = 'conditional'
            model_kwargs['y']['inpainted_motion'] = model_kwargs['obs_x0']
            model_kwargs['y']['inpainting_mask'] = model_kwargs['obs_mask'] # used to do [nsamples, nframes] --> [nsamples, njoints, nfeats, nframes]
            model_kwargs['y']['reconstruction_guidance'] = False
        elif self.args.imputate: # if loss was present over keyframes during training, we may use inpaiting at inference time
            model_kwargs['y']['imputate'] = 1
            model_kwargs['y']['stop_imputation_at'] = self.args.stop_imputation_at
            model_kwargs['y']['replacement_distribution'] = 'conditional' # TODO: check if should also support marginal distribution
            model_kwargs['y']['inpainted_motion'] = model_kwargs['obs_x0']
            model_kwargs['y']['inpainting_mask'] = model_kwargs['obs_mask']
            if self.args.reconstruction_guidance: # if loss was present over keyframes during training, we may use guidance at inference time
                model_kwargs['y']['reconstruction_guidance'] = self.args.reconstruction_guidance
                model_kwargs['y']['reconstruction_weight'] = self.args.reconstruction_weight
                model_kwargs['y']['gradient_schedule'] = self.args.gradient_schedule
                model_kwargs['y']['stop_recguidance_at'] = self.args.stop_recguidance_at
        elif self.args.reconstruction_guidance: # if loss was present over keyframes during training, we may use guidance at inference time
            model_kwargs['y']['inpainted_motion'] = model_kwargs['obs_x0']
            model_kwargs['y']['inpainting_mask'] = model_kwargs['obs_mask']
            model_kwargs['y']['reconstruction_guidance'] = self.args.reconstruction_guidance
            model_kwargs['y']['reconstruction_weight'] = self.args.reconstruction_weight
            model_kwargs['y']['gradient_schedule'] = self.args.gradient_schedule
            model_kwargs['y']['stop_recguidance_at'] = self.args.stop_recguidance_at


        return self.get_keyframe_indices(model_kwargs['obs_mask']), joint_mask
