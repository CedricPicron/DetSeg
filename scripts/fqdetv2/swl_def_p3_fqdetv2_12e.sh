#!/bin/bash

exp_name=swl_def_p3_fqdetv2_12e
out=outputs/main/fqdetv2/$exp_name
mkdir -p $out

bat=2
wor=2

cpf=$out/checkpoint.pth
epo=12
lrd=9

arch=bch
back_cfg=configs/mmdet/backbones/swin_l_v0.py
core_cfg1=configs/cores/in_proj_v2.py
core_cfg2=configs/cores/deform_encoder_v0.py
head_cfg=configs/gvd/sel_v291.py

torchrun --nnodes=2 --nproc_per_node=4 main.py --output_dir $out --batch_size $bat --num_workers $wor \
--checkpoint_full $cpf --epochs $epo --lr_drops $lrd --arch_type $arch --backbone_type mmdet \
--mmdet_backbone_cfg_path $back_cfg --cores cfg cfg --core_cfg_paths $core_cfg1 $core_cfg2 --heads cfg \
--head_cfg_paths $head_cfg --lr_default 2e-4 --lr_backbone 1e-5 --lr_core 2e-4 --lr_heads 2e-4 --lr_offset 1e-4 \
--max_grad_norm 0.1
