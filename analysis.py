"""
Analysis of trained sample decoder on COCO
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import DataLoader, RandomSampler

from datasets.build import build_dataset
from main import get_parser
from models.detr import build_detr
from utils.box_ops import box_cxcywh_to_xyxy
from utils.data import val_collate_fn


# Analysis hook function
def analysis_hook(module, input, output):
    sample_id = module.sample_id
    layer_id = module.layer_id
    image_dir = Path(f"./analysis/{sample_id}/{layer_id}")
    image_dir.mkdir(parents=True, exist_ok=True)

    _, seg_maps, _, curio_maps = output
    num_slots = seg_maps.shape[0]

    for slot_id in range(num_slots):
        seg_map = seg_maps[slot_id].cpu().numpy()
        curio_map = curio_maps[slot_id].cpu().numpy()

        plt.imsave(image_dir / f"{slot_id+1}a.eps", seg_map)
        plt.imsave(image_dir / f"{slot_id+1}b.eps", curio_map)


# COCO classes
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


def plot_and_save_predictions(pil_image, slot_indices, pred_probs, pred_boxes, sample_id):
    """
    Function used to plot and save predictions of the given PIL image.
    """

    # Initialize figure with PIL image and get axes
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_image)
    ax = plt.gca()

    # Add predictions to the image
    for slot_id, probs, (xmin, ymin, xmax, ymax) in zip(slot_indices, pred_probs, pred_boxes.tolist()):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, fill=False, linewidth=3))

        class_id = probs.argmax()
        class_text = f'{CLASSES[class_id]}: {probs[class_id]:0.2f} ({slot_id})'
        ax.text(xmin, ymin, class_text, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))

    # Remove figure axis and save figure
    plt.axis('off')
    plt.savefig(f"./analysis/{sample_id}/predictions.jpg")


def plot_and_save_groundtruth(pil_image, annotations, sample_id):
    """
    Function used to plot and save ground-truth annonations of the given PIL image.
    """

    # Get ground-truth class labels
    label_ids = [obj["category_id"] for obj in annotations]
    label_ids = torch.tensor(label_ids, dtype=torch.int64)

    # Get object boxes in (left, top, width, height) format
    boxes = [obj["bbox"] for obj in annotations]
    boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)

    # Transform boxes to (left, top, right, bottom) format
    boxes[:, 2:] += boxes[:, :2]

    # Crop boxes such that they fit within the image
    w, h = pil_image.size
    boxes[:, 0::2].clamp_(min=0, max=w)
    boxes[:, 1::2].clamp_(min=0, max=h)

    # Only keep objects with well-defined boxes
    keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
    label_ids = label_ids[keep]
    boxes = boxes[keep]

    # Initialize figure with PIL image and get axes
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_image)
    ax = plt.gca()

    # Add ground-truth annotations to the image
    for label_id, (xmin, ymin, xmax, ymax) in zip(label_ids, boxes.tolist()):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, fill=False, linewidth=3))

        class_name = f'{CLASSES[label_id]}'
        ax.text(xmin, ymin, class_name, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))

    # Remove figure axis and save figure
    plt.axis('off')
    plt.savefig(f"./analysis/{sample_id}/groundtruth.jpg")


# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default='', type=str, help='checkpoint path with sample decoder to be analyzed')
parser.add_argument('--num_samples', default=10, type=int, help='number of random validation samples for analysis')
parser.add_argument('--add_args', default=[], nargs=argparse.REMAINDER, help='additional args for main.py argparser')
analysis_args = parser.parse_args()
main_args = get_parser().parse_args(args=analysis_args.add_args)

# Use batch size one for simplicty
main_args.batch_size = 1
main_args.num_workers = 1

# Get validation dataset and sampler
_, val_dataset, _ = build_dataset(main_args)
sampler = RandomSampler(val_dataset)

# Get dataloader
dataloader_kwargs = {'collate_fn': val_collate_fn, 'num_workers': main_args.num_workers, 'pin_memory': True}
dataloader = DataLoader(val_dataset, main_args.batch_size, sampler=sampler, **dataloader_kwargs)

# Get model with default parameters and put it on correct device
device = torch.device(main_args.device)
model = build_detr(main_args).to(device)

# Load model parameters from checkpoint
checkpoint = torch.load(Path(analysis_args.checkpoint), map_location='cpu')
model.load_state_dict(checkpoint['model'])

# Set model in evaluation mode
model.eval()

# Register analysis hooks and add layer ids
for layer_id, layer in enumerate(model.decoder.layers, 1):
    layer.register_forward_hook(analysis_hook)
    layer.layer_id = layer_id

# Perform analysis on random validation samples
with torch.no_grad():
    for sample_id, (image, tgt_dict, eval_dict) in enumerate(dataloader, 1):

        # Compute model predictions
        [setattr(layer, 'sample_id', sample_id) for layer in model.decoder.layers]
        pred_list = model(image.to(device))

        # Get probalities and boxes
        pred_dict = pred_list[0]
        pred_probs = pred_dict['logits'].softmax(dim=-1)

        # Keep only predictions of objects with enough confidence
        pred_probs = pred_probs[:, :-1]
        keep = pred_probs.max(dim=-1).values > 0.5
        slot_indices = [i for i, keep_bool in enumerate(keep, 1) if keep_bool]

        pred_probs = pred_probs[keep]
        pred_boxes = pred_dict['boxes'][keep]

        # Get PIL image
        image_id = eval_dict['image_ids'][0].item()
        image_path = val_dataset.root / val_dataset.coco.loadImgs(image_id)[0]['file_name']
        pil_image = Image.open(image_path).convert('RGB')

        # Transform boxes in correct format and rescale them
        pred_boxes = box_cxcywh_to_xyxy(pred_boxes)
        width, height = pil_image.size
        scale = torch.tensor([width, height, width, height], dtype=pred_boxes.dtype, device=pred_boxes.device)
        pred_boxes = scale * pred_boxes

        # Plot and save image with predictions
        plot_and_save_predictions(pil_image, slot_indices, pred_probs, pred_boxes, sample_id)

        # Get ground-truth annotations
        annotation_ids = val_dataset.coco.getAnnIds(imgIds=image_id)
        annotations = val_dataset.coco.loadAnns(annotation_ids)
        annotations = [obj for obj in annotations if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        # Plot and save image with ground-truth annotations
        plot_and_save_groundtruth(pil_image, annotations, sample_id)

        if sample_id == analysis_args.num_samples:
            break
