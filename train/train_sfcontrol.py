# This code is based on https://github.com/openai/guided-diffusion,
# and is used to train a diffusion model on human motion sequences.

import os
import sys
import json
from pprint import pprint
from utils.fixseed import fixseed
from utils.parser_util import train_args
from utils import dist_util
from train.training_loop import TrainLoop
from data_loaders.get_data import DatasetConfig, get_dataset_loader
from utils.model_util import create_model_and_diffusion
from configs import card


def main():
    args = train_args(base_cls=card.motion_abs_unet_adagn_xl) # Choose the default full motion model from GMD

    if args.endeffector_only:
        if not args.endeffector_conditioned:
            args.endeffector_selection_scheme = "none"
        model_info = "t2m_stage1_{}_{}".format(args.data_rep, args.endeffector_selection_scheme)
        
    else:
        model_info = "t2m_stage2_{}_{}".format(args.data_rep, args.endeffector_selection_scheme)
        

    args.save_dir = os.path.join("save", model_info)
    pprint(args.__dict__)
    fixseed(args.seed)

    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    dist_util.setup_dist(args.device)

    print("creating data loader...")
    data_conf = DatasetConfig(
        name=args.dataset,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        use_abs3d=args.abs_3d,
        traject_only=args.traj_only,
        use_random_projection=args.use_random_proj,
        random_projection_scale=args.random_proj_scale,
        augment_type=args.augment_type,
        std_scale_shift=args.std_scale_shift,
        drop_redundant=args.drop_redundant,
        endeffector_only=args.endeffector_only,
        head=args.head,
        data_rep=args.data_rep,
        fixed_len=args.fixed_len
    )

    data = get_dataset_loader(data_conf)

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)
    model.to(dist_util.dev())

    if args.endeffector_only:
        diffusion.data_inv_transform_fn = data.dataset.t2m_dataset.inv_transform_stage1
    else:
        diffusion.data_inv_transform_fn = data.dataset.t2m_dataset.inv_transform


    print('Total params: %.2fM' %
          (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))
    print("Training...")
    TrainLoop(args, model, diffusion, data).run_loop()
    #wandb.finish()


if __name__ == "__main__":
    main()