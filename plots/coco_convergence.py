"""
Script plotting COCO convergence curves of specified models.
"""

import argparse
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from pathlib import Path


# Parse command-line arguments
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--legend_names', nargs='*', default='', type=str, help='names of models to appear in legend')
parser.add_argument('--model_names', nargs='+', default='', type=str, help='names of models to feature in plot')
parser.add_argument('--model_weights', nargs='*', default=[1.0], type=float, help='weights related to training speed')
parser.add_argument('--plot_name', default='two-stage', type=str, help='name of COCO convergence plot')
parser.add_argument('--text_offsets', nargs='*', default=[0.0], type=float, help='Y-offset of final COCO AP value')
args = parser.parse_args()

# Some model-independent preparation
models_dir = Path().resolve().parent / 'outputs'
num_models = len(args.model_names)

legend_names = args.legend_names if len(args.legend_names) == num_models else args.model_names
model_weights = args.model_weights if len(args.model_weights) == num_models else [1.0] * num_models
text_offsets = args.text_offsets if len(args.text_offsets) == num_models else [0.0] * num_models

plt.figure()
loc = plticker.MultipleLocator(base=1.0)
plt.gca().xaxis.set_major_locator(loc)

# Draw COCO convergence curve for every specified model
for i in range(num_models):
    file_name = models_dir / args.model_names[i] / 'log.txt'

    with open(file_name, 'r') as file:
        lines = file.readlines()

        model_weight = model_weights[i]
        x_data = [model_weight * j for j in range(1, len(lines)+1)]
        y_data = []

        for line in lines:
            eval_dict = json.loads(line)
            eval_keys = [key for key in eval_dict.keys() if 'val_eval' in key]
            ap = 100 * eval_dict[eval_keys[-1]][0]
            y_data.append(ap)

        plt.plot(x_data, y_data, label=legend_names[i])
        plt.text(x_data[-1] - 0.6, y_data[-1] + 0.2 + text_offsets[i], f'{y_data[-1]: .1f}')

# Label plot
if len(args.model_weights) == num_models:
    plt.title("COCO val AP vs. normalized training time")
    plt.xlabel("Normalized training time")

else:
    plt.title("COCO val AP vs. training epochs")
    plt.xlabel("Training epochs")

plt.ylabel("COCO val AP")
plt.ylim(top=(plt.ylim()[1] + 1))
plt.legend()

# Save plot
save_dir = Path() / 'coco_convergence'
save_dir.mkdir(exist_ok=True)
plt.savefig(save_dir / (args.plot_name + '.png'))
