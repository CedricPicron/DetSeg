"""
Convert COCO panoptic annotations to COCO detection format.
"""

from panopticapi.converters.panoptic2detection_coco_format import convert_panoptic_to_detection_coco_format


# Convert training annotations
in_json_file = 'datasets/coco/annotations/panoptic_train2017.json'
seg_folder = 'datasets/coco/annotations/panoptic_train2017'
out_json_file = 'datasets/coco/annotations/panoptic_train2017_converted.json'
cat_json = 'datasets/coco/annotations/panoptic_coco_categories.json'
things_only = False

args = (in_json_file, seg_folder, out_json_file, cat_json, things_only)
convert_panoptic_to_detection_coco_format(*args)

# Convert validation annotations
in_json_file = 'datasets/coco/annotations/panoptic_val2017.json'
seg_folder = 'datasets/coco/annotations/panoptic_val2017'
out_json_file = 'datasets/coco/annotations/panoptic_val2017_converted.json'
cat_json = 'datasets/coco/annotations/panoptic_coco_categories.json'
things_only = False

args = (in_json_file, seg_folder, out_json_file, cat_json, things_only)
convert_panoptic_to_detection_coco_format(*args)
