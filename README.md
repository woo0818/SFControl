# Motion Synthesis with Sparse and Flexible Keyjoint Control

<p align="left">
  <a href="https://arxiv.org/abs/2503.15557">
    <img src="https://img.shields.io/badge/Arxiv-Pdf-A42C25?style=flat&logo=arXiv&logoColor=white">
  </a>
  <a href="https://inwoohwang.me/SFControl">
    <img src="https://img.shields.io/badge/Project-Page-green?style=flat&logo=Google%20chrome&logoColor=white">
  </a>
</p>

![Teaser Image](https://inwoohwang.me/SFControl/static/images/teasor.png)

---

## Dataset Preparation
1. **Download and preprocess** the [HumanML3D](https://github.com/EricGuo5513/HumanML3D) dataset under the `dataset/` directory.  
2. **Convert** the raw HumanML3D data into our **global representation** format.  
   *(The conversion script will be released soon.)*

---

## Training

### Stage1 Model 
```bash
python -m train.train_sfcontrol   --endeffector_conditioned   --endeffector_only   --endeffector_selection_scheme random_joints   --head   --diffusion_steps 50   --latent_dim 128   --lambda_ric_pos 0.1   --lambda_ric_vel 10.0   --lambda_ric_fc 10.0   --dataset humanml   --data_rep glo_joints
```

### Stage2 Model
```bash
python -m train.train_sfcontrol   --endeffector_conditioned   --endeffector_selection_scheme all_endeffectors   --head   --diffusion_steps 50   --latent_dim 512   --dataset humanml   --data_rep glo_root
```


## Citation
If you find our work helpful, please consider citing:
```
@misc{hwang2025motionsynthesissparseflexible,
    title={Motion Synthesis with Sparse and Flexible Keyjoint Control},
    author={Inwoo Hwang and Jinseok Bae and Donggeun Lim and Young Min Kim},
    year={2025},
    eprint={2503.15557},
    archivePrefix={arXiv},
    primaryClass={cs.GR},
    url={https://arxiv.org/abs/2503.15557}
}
```

---

## Acknowledgment

We sincerely thank the open-source projects that our code builds upon and draws inspiration from:
CondMDI, GMD and MDM.
