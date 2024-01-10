"""
Collection of functions related to TIDE analysis.
"""
import contextlib
import json
import os
import warnings
from zipfile import ZipFile

from tidecv import datasets, TIDE


def tide_analysis(dataset, result_file_name, output_dir):
    """
    Function performing the TIDE analysis based on the provided JSON result file.

    Args:
        dataset (str): String containing the name of the dataset on which the results were obtained.
        result_file_name (str): String with name of result file used for TIDE analysis.
        output_dir (Path): Path to output directory containing the JSON file and used for TIDE analysis results.

    Raises:
        ValueError: Error when no JSON or ZIP file was found containing the result file.
        ValueError: Error when an invalid dataset name is provided.
    """

    # Get path to JSON file
    json_path = output_dir / result_file_name
    json_path = json_path.with_suffix('.json')

    # Obtain JSON file from ZIP file if needed
    from_zip_file = False

    if not json_path.is_file():
        zip_path = json_path.with_suffix('.zip')

        if zip_path.is_file:
            from_zip_file = True

            with ZipFile(zip_path, 'r') as zip_file:
                zip_file.extract(json_path.name, path=zip_path.parent)

        else:
            error_msg = f"No JSON or ZIP file was found for the path {zip_path.with_suffix('')}."
            raise ValueError(error_msg)

    # Get inputs for the TIDE analysis
    if dataset == 'coco':
        ground_truth = datasets.COCO(path='datasets/coco/annotations/instances_val2017.json')
        predictions = datasets.COCOResult(json_path)

        with open(json_path, 'r') as json_file:
            result_dicts = json.load(json_file)

            if 'segmentation' in result_dicts[0]:
                mode = TIDE.MASK
            else:
                mode = TIDE.BOX

    else:
        error_msg = f"Invalid dataset name for TIDE analysis (got '{dataset}')."
        raise ValueError(error_msg)

    # Perform TIDE analysis
    tide_dir = output_dir / 'tide'

    tide = TIDE()
    tide.evaluate_range(ground_truth, predictions, mode=mode)

    with open(tide_dir / f'{result_file_name}.txt', 'w') as txt_file:
        with contextlib.redirect_stdout(txt_file):
            tide.summarize()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        tide.plot(out_dir=tide_dir)

    # Remove JSON file if extracted from ZIP file
    if from_zip_file:
        os.remove(json_path)

    return
