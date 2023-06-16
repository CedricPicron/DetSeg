#!/bin/bash

# Set Cityscapes symbolic links
CITYSCAPES_DIR=/esat/visicsrodata/datasets/cityscapes

ln -s $CITYSCAPES_DIR/gtFine datasets/cityscapes/gtFine
ln -s $CITYSCAPES_DIR/leftImg8bit datasets/cityscapes/leftImg8bit

# Set COCO symbolic links
COCO_DIR=/esat/visicsrodata/datasets/coco

ln -s $COCO_DIR/images/test2017 datasets/coco/test2017
ln -s $COCO_DIR/images/train2017 datasets/coco/train2017
ln -s $COCO_DIR/images/val2017 datasets/coco/val2017

ln -s $COCO_DIR/annotations/image_info_test-dev2017.json datasets/coco/annotations/image_info_test-dev2017.json
ln -s $COCO_DIR/annotations/instances_train2017.json datasets/coco/annotations/instances_train2017.json
ln -s $COCO_DIR/annotations/instances_val2017.json datasets/coco/annotations/instances_val2017.json

ln -s $COCO_DIR/annotations/panoptic_train2017 datasets/coco/annotations/panoptic_train2017
ln -s $COCO_DIR/annotations/panoptic_train2017.json datasets/coco/annotations/panoptic_train2017.json
ln -s $COCO_DIR/annotations/panoptic_val2017 datasets/coco/annotations/panoptic_val2017
ln -s $COCO_DIR/annotations/panoptic_val2017.json datasets/coco/annotations/panoptic_val2017.json

# Convert COCO panoptic annotations to COCO detection format
python tools/convert_coco_panoptic.py

# Only track symbolic links of converted COCO panoptic annotations
ANN_DIR=/esat/dragon/cpicron/Projects/GroupDetr/datasets/coco/annotations

mv $ANN_DIR/panoptic_train2017_converted.json $ANN_DIR/panoptic_train2017_converted_ignored.json
mv $ANN_DIR/panoptic_val2017_converted.json $ANN_DIR/panoptic_val2017_converted_ignored.json

ln -s $ANN_DIR/panoptic_train2017_converted_ignored.json $ANN_DIR/panoptic_train2017_converted.json
ln -s $ANN_DIR/panoptic_val2017_converted_ignored.json $ANN_DIR/panoptic_val2017_converted.json

# Set LVIS symbolic links
LVIS_DIR=/esat/raidho/cpicron/Datasets/lvis

ln -s $LVIS_DIR/annotations/lvis_v0.5_train.json datasets/lvis/annotations/lvis_v0.5_train.json
ln -s $LVIS_DIR/annotations/lvis_v0.5_val.json datasets/lvis/annotations/lvis_v0.5_val.json
ln -s $LVIS_DIR/annotations/lvis_v1_train.json datasets/lvis/annotations/lvis_v1_train.json
ln -s $LVIS_DIR/annotations/lvis_v1_val.json datasets/lvis/annotations/lvis_v1_val.json

# Transform LVIS annotations to COCO format (COCO classes only)
cd datasets/coco/annotations
python lvis_v0.5_to_coco.py
python lvis_v1_to_coco.py
cd -

# Get LVIS annotations with only COCO classes
cd datasets/lvis/annotations
python cocofy_lvis.py
cd -

# Set Objects365 symbolic links
OBJECTS365_DIR=/esat/raidho/cpicron/Datasets/objects365

ln -s $OBJECTS365_DIR/annotations datasets/objects365/annotations
ln -s $OBJECTS365_DIR/train datasets/objects365/train
ln -s $OBJECTS365_DIR/val datasets/objects365/val
