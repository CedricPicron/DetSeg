"""
Script plotting COCO AP vs. model cost for different TPN models.
"""

import argparse
import matplotlib.pyplot as plt
from pathlib import Path


# Parse command-line arguments
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--metric', default='params', type=str, help='metric measuring how expensive a TPN model is')
parser.add_argument('--plot_name', default='', type=str, help='name of plot (name of metric is used when missing)')
parser.add_argument('--x_offsets', nargs='*', default=[0.0], type=float, help='X-offsets of TPN configuration names')
parser.add_argument('--y_offsets', nargs='*', default=[0.0], type=float, help='Y-offsets of TPN configuration names')
args = parser.parse_args()

# Get dictionaries containing information about each TPN model
tpn_13 = {'cfg': (1, 3), 'ap': 38.6, 'params': 33.5, 'tfps': 1.95, 'tmem': 2.65, 'ifps': 6.30}
tpn_15 = {'cfg': (1, 5), 'ap': 39.3, 'params': 34.9, 'tfps': 1.72, 'tmem': 2.98, 'ifps': 5.62}
tpn_17 = {'cfg': (1, 7), 'ap': 39.6, 'params': 36.3, 'tfps': 1.53, 'tmem': 3.31, 'ifps': 5.17}
tpn_21 = {'cfg': (2, 1), 'ap': 38.3, 'params': 33.3, 'tfps': 2.01, 'tmem': 2.55, 'ifps': 6.30}
tpn_22 = {'cfg': (2, 2), 'ap': 39.6, 'params': 34.8, 'tfps': 1.76, 'tmem': 2.88, 'ifps': 5.73}
tpn_23 = {'cfg': (2, 3), 'ap': 40.0, 'params': 36.2, 'tfps': 1.56, 'tmem': 3.21, 'ifps': 5.30}
tpn_31 = {'cfg': (3, 1), 'ap': 39.1, 'params': 34.6, 'tfps': 1.81, 'tmem': 2.77, 'ifps': 5.92}
tpn_32 = {'cfg': (3, 2), 'ap': 40.0, 'params': 36.7, 'tfps': 1.53, 'tmem': 3.27, 'ifps': 5.17}
tpn_41 = {'cfg': (4, 1), 'ap': 39.6, 'params': 35.9, 'tfps': 1.65, 'tmem': 3.00, 'ifps': 5.47}
tpn_51 = {'cfg': (5, 1), 'ap': 39.9, 'params': 37.1, 'tfps': 1.51, 'tmem': 3.22, 'ifps': 5.11}

# Get list with all TPN models
tpn_models = [tpn_13, tpn_15, tpn_17, tpn_21, tpn_22, tpn_23, tpn_31, tpn_32, tpn_41, tpn_51]

# Get X-axis and Y-axis text offsets
num_models = len(tpn_models)
x_offsets = args.x_offsets if len(args.x_offsets) == num_models else [0.0] * num_models
y_offsets = args.y_offsets if len(args.y_offsets) == num_models else [0.0] * num_models

# Create new figure
plt.figure()

# Add data point corresponding to each TPN model
for i, tpn_model in enumerate(tpn_models):
    if args.metric == 'params':
        x_data = tpn_model['params']
        x_offset = -0.16 + x_offsets[i]
        y_offset = 0.05 + y_offsets[i]

    elif args.metric == 'train_latency':
        x_data = 1000 / tpn_model['tfps']
        x_offset = -6.5 + x_offsets[i]
        y_offset = 0.05 + y_offsets[i]

    elif args.metric == 'train_memory':
        x_data = tpn_model['tmem']
        x_offset = -0.03 + x_offsets[i]
        y_offset = 0.05 + y_offsets[i]

    elif args.metric == 'inference_latency':
        x_data = 1000 / tpn_model['ifps']
        x_offset = -1.8 + x_offsets[i]
        y_offset = 0.05 + y_offsets[i]

    else:
        error_msg = f"Unknown metric '{args.metric}' was provided."
        raise ValueError(error_msg)

    plt.plot(x_data, tpn_model['ap'], 'bs')
    plt.text(x_data + x_offset, tpn_model['ap'] + y_offset, str(tpn_model['cfg']))

# Label plot
if args.metric == 'params':
    plt.title("COCO val AP vs. parameters")
    plt.xlabel("Parameters (M)")
    plt.xlim(left=(plt.xlim()[0] - 0.1), right=(plt.xlim()[1] + 0.4))

elif args.metric == 'train_latency':
    plt.title("COCO val AP vs. training latency")
    plt.xlabel("Training latency (ms)")
    plt.xlim(left=(plt.xlim()[0] - 5), right=(plt.xlim()[1] + 10))

elif args.metric == 'train_memory':
    plt.title("COCO val AP vs. training memory")
    plt.xlabel("Training memory (GB)")
    plt.xlim(left=(plt.xlim()[0] - 0.03), right=(plt.xlim()[1] + 0.06))

elif args.metric == 'inference_latency':
    plt.title("COCO val AP vs. inference latency")
    plt.xlabel("Inference latency (ms)")
    plt.xlim(left=(plt.xlim()[0] - 2), right=(plt.xlim()[1] + 3))

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
