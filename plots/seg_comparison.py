"""
Script plotting COCO AP vs. model cost for different instance segmentation models.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt


# Get metrics dictionary
metrics_dict = dict()
metrics_dict['Parameters'] = ('num_params', 'Parameters (M)')
metrics_dict['FLOPs'] = ('inf_flops', 'Inference GFLOPs')
metrics_dict['Latency'] = ('inf_fps', 'Inference latency (ms)')

# Get models dictionary
models_dict = dict()
models_dict['Mask R-CNN++'] = ('gvd_353', 'gvd_370')
models_dict['PointRend++'] = ('gvd_355', 'gvd_393')
models_dict['RefineMask++'] = ('gvd_356', 'gvd_388')
models_dict['EffSeg'] = ('gvd_386', 'gvd_394')

# Get lists with colors and markers
colors = ['darkorange', 'magenta', 'mediumblue', 'red']
markers = ['v', 'D', 's', 'o']

# Get figure for every metric
for metric, (analysis_key, xlabel) in metrics_dict.items():

    # Create new figure
    plt.figure()

    # Add data points
    for i, (model_name, models) in enumerate(models_dict.items()):
        color = colors[i]
        marker = markers[i]

        if model_name == 'EffSeg':
            x_data = []
            y_data = []

        for j, model in enumerate(models):
            model_dir = Path() / f'../outputs/{model}'
            analysis_file_name = model_dir / 'model_analysis.json'

            with open(analysis_file_name, 'r') as analysis_file:
                analysis_dict = json.load(analysis_file)
                x = analysis_dict[analysis_key]

                if metric == 'Latency':
                    x = 1000 / x

            if Path.exists(model_dir / 'eval_2017_val_single_scale.txt'):
                result_file_name = model_dir / 'eval_2017_val_single_scale.txt'
                eval_key = 'eval_2_segm'

            else:
                result_file_name = model_dir / 'log.txt'
                eval_key = 'eval_eval_2_segm'

            with open(result_file_name, 'r') as result_file:
                result_dict = json.loads(result_file.readlines()[-1])
                y = 100 * result_dict[eval_key][0]

            label = model_name if j == 0 else None
            plt.plot(x, y, color=color, marker=marker, linestyle='None', label=label)

            if model_name == 'EffSeg':
                x_data.append(x)
                y_data.append(y)

                if j == (len(models)-1):
                    plt.plot(x_data, y_data, color=color, linestyle='--')

    # Add legend
    plt.figlegend(loc='upper center', ncol=2)

    # Label plot
    plt.xlabel(xlabel)
    plt.ylabel("COCO validation AP")
    plt.ylim(bottom=(plt.ylim()[0] - 0.1), top=(plt.ylim()[1] + 0.2))

    # Save plot
    save_dir = Path() / 'seg_comparison'
    save_dir.mkdir(exist_ok=True)
    plt.savefig(save_dir / (metric.lower() + '.png'))
