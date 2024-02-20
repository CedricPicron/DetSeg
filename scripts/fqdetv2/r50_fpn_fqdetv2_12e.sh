#!/bin/bash

exp_name=r50_fpn_fqdetv2_12e
out=outputs/main/fqdetv2/$exp_name
mkdir -p $out

bat=2
wor=2

cpf=$out/checkpoint.pth
epo=12
lrd=9

arch=bch
core=mmdet
core_cfg=configs/mmdet/cores/fpn_v3.py
head_cfg=configs/gvd/sel_v291.py

torchrun --nproc_per_node=$nod main.py --output_dir $out --batch_size $bat --num_workers $wor --checkpoint_full $cpf \
--epochs $epo --lr_drops $lrd --arch_type $arch --resnet_out_ids 3 4 5 --cores $core --mmdet_core_cfg_path $core_cfg \
--heads cfg --head_cfg_paths $head_cfg
