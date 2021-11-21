"""
Script plotting COCO AP vs. model cost for different TPN models.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch


# Parse command-line arguments
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--curve_ids', nargs='*', default=[3, 0, 4, 5, 12], type=int, help='ids of models to use in curve')
parser.add_argument('--metric', default='params', type=str, help='metric measuring how expensive a TPN model is')
parser.add_argument('--plot_name', default='', type=str, help='name of plot (name of metric is used when missing)')
parser.add_argument('--x_offsets', nargs='*', default=[0.0], type=float, help='X-offsets of TPN configuration names')
parser.add_argument('--y_offsets', nargs='*', default=[0.0], type=float, help='Y-offsets of TPN configuration names')
args = parser.parse_args()

# Get dictionaries containing information about each TPN model
tpn_13 = {'cfg': (1, 3), 'ap': 38.6, 'params': 33.5, 'tfps': 2.06, 'tmem': 2.65, 'ifps': 6.35}
tpn_15 = {'cfg': (1, 5), 'ap': 39.3, 'params': 34.9, 'tfps': 1.84, 'tmem': 2.98, 'ifps': 5.80}
tpn_17 = {'cfg': (1, 7), 'ap': 39.6, 'params': 36.3, 'tfps': 1.65, 'tmem': 3.31, 'ifps': 5.35}
tpn_21 = {'cfg': (2, 1), 'ap': 38.3, 'params': 33.3, 'tfps': 2.12, 'tmem': 2.55, 'ifps': 6.49}
tpn_22 = {'cfg': (2, 2), 'ap': 39.6, 'params': 34.8, 'tfps': 1.88, 'tmem': 2.88, 'ifps': 5.89}
tpn_23 = {'cfg': (2, 3), 'ap': 40.0, 'params': 36.2, 'tfps': 1.69, 'tmem': 3.21, 'ifps': 5.42}
tpn_31 = {'cfg': (3, 1), 'ap': 39.1, 'params': 34.6, 'tfps': 1.92, 'tmem': 2.77, 'ifps': 6.01}
tpn_32 = {'cfg': (3, 2), 'ap': 40.0, 'params': 36.7, 'tfps': 1.64, 'tmem': 3.27, 'ifps': 5.33}
tpn_41 = {'cfg': (4, 1), 'ap': 39.6, 'params': 35.9, 'tfps': 1.77, 'tmem': 3.00, 'ifps': 5.64}
tpn_51 = {'cfg': (5, 1), 'ap': 39.9, 'params': 37.1, 'tfps': 1.63, 'tmem': 3.22, 'ifps': 5.30}
tpn_61 = {'cfg': (6, 1), 'ap': 40.2, 'params': 38.4, 'tfps': 1.51, 'tmem': 3.45, 'ifps': 4.98}
tpn_24 = {'cfg': (2, 4), 'ap': 0, 'params': 37.6, 'tfps': 1.53, 'tmem': 3.53, 'ifps': 5.02}
tpn_42 = {'cfg': (4, 2), 'ap': 0, 'params': 38.7, 'tfps': 1.46, 'tmem': 3.66, 'ifps': 4.85}
tpn_71 = {'cfg': (7, 1), 'ap': 40.2, 'params': 39.6, 'tfps': 1.43, 'tmem': 3.68, 'ifps': 4.70}
tpn_33 = {'cfg': (3, 3), 'ap': 40.5, 'params': 38.8, 'tfps': 1.44, 'tmem': 3.76, 'ifps': 4.75}

# Get list with all TPN models
tpn_models = [tpn_13, tpn_15, tpn_17, tpn_21, tpn_22, tpn_23, tpn_31, tpn_32, tpn_41, tpn_51, tpn_61, tpn_71, tpn_33]

# Get X-axis and Y-axis text offsets
num_models = len(tpn_models)
x_offsets = args.x_offsets if len(args.x_offsets) == num_models else [0.0] * num_models
y_offsets = args.y_offsets if len(args.y_offsets) == num_models else [0.0] * num_models

# Create new figure
plt.figure()

# Initialize empty lists containing the TPN curve coordinates
curve_x = []
curve_y = []

# Add data point corresponding to each TPN model
for i, tpn_model in enumerate(tpn_models):
    if args.metric == 'params':
        x_data = tpn_model['params']
        x_offset = 0.10 + x_offsets[i]
        y_offset = -0.10 + y_offsets[i]

    elif args.metric == 'train_latency':
        x_data = 1000 / tpn_model['tfps']
        x_offset = 3.5 + x_offsets[i]
        y_offset = -0.10 + y_offsets[i]

    elif args.metric == 'train_memory':
        x_data = tpn_model['tmem']
        x_offset = 0.02 + x_offsets[i]
        y_offset = -0.10 + y_offsets[i]

    elif args.metric == 'inference_latency':
        x_data = 1000 / tpn_model['ifps']
        x_offset = 1.0 + x_offsets[i]
        y_offset = -0.10 + y_offsets[i]

    else:
        error_msg = f"Unknown metric '{args.metric}' was provided."
        raise ValueError(error_msg)

    plt.plot(x_data, tpn_model['ap'], 'bs')
    plt.text(x_data + x_offset, tpn_model['ap'] + y_offset, str(tpn_model['cfg']), size='small')

    if i in args.curve_ids:
        curve_x.append(x_data)
        curve_y.append(tpn_model['ap'])

# Add TPN accuracy vs. efficiency trade-off curve
curve_x, sort_ids = torch.tensor(curve_x).sort(dim=0)
curve_y = torch.tensor(curve_y)[sort_ids]
curve_x = curve_x.tolist()
curve_y = curve_y.tolist()

plt.plot(curve_x, curve_y, 'm')
[plt.plot(x, y, 'ms') for x, y in zip(curve_x, curve_y)]

# Label plot
if args.metric == 'params':
    plt.title("COCO val AP vs. parameters")
    plt.xlabel("Parameters (M)")
    plt.xlim(left=(plt.xlim()[0] - 0.1), right=(plt.xlim()[1] + 0.4))

elif args.metric == 'train_latency':
    plt.title("COCO val AP vs. training latency")
    plt.xlabel("Training latency (ms)")
    plt.xlim(left=(plt.xlim()[0] - 1), right=(plt.xlim()[1] + 15))

elif args.metric == 'train_memory':
    plt.title("COCO val AP vs. training memory")
    plt.xlabel("Training memory (GB)")
    plt.xlim(left=(plt.xlim()[0] - 0.0), right=(plt.xlim()[1] + 0.08))

elif args.metric == 'inference_latency':
    plt.title("COCO val AP vs. inference latency")
    plt.xlabel("Inference latency (ms)")
    plt.xlim(left=(plt.xlim()[0] - 0.2), right=(plt.xlim()[1] + 4))

else:
    error_msg = f"Unknown metric '{args.metric}' was provided."
    raise ValueError(error_msg)

plt.ylabel("COCO val AP")
plt.ylim(bottom=(plt.ylim()[0] - 0.1), top=(plt.ylim()[1] + 0.2))

# Save plot
save_dir = Path() / 'tpn_curve'
save_dir.mkdir(exist_ok=True)

plot_name = args.plot_name if args.plot_name else args.metric
plt.savefig(save_dir / (plot_name + '.png'))
