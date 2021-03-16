"""
Utilities to compute FLOPS.
"""

from detectron2.utils.analysis import flop_count_operators


def compute_flops(model, dataset, num_samples=100):
    """
    Function computing the FLOPS of a model averaged over samples from the beginning of the given dataset.

    Needs improvement on two aspects:
        1) the Images and Boxes structures are currently not supported;
        2) some used operations do not have a corresponding flop count function or are not properly ignored.

    Args:
        model (nn.Module): Module implementing a model from which the average number of FLOPS is to be computed.
        dataset (Dataset): Dataset providing the input samples used to get the average number of FLOPS.
        num_samples (int): Number of input samples sampled from beginning of dataset (default=100).

    Returns:
        avg_flops (float): The average number of GFLOPS of the given model.
    """

    # Get model device and initialize average number of flops
    device = next(model.parameters()).device
    avg_flops = 0.0

    # Compute number of FLOPS for each input sample
    for i in range(num_samples):
        input_dict = {'image': dataset[i][0].to(device)}
        flops_dict = flop_count_operators(model, [input_dict])

        avg_flops += sum(flops_dict.values()) / num_samples
        print(f"({i+1}/{num_samples}): {dict(flops_dict)}")

    return avg_flops
