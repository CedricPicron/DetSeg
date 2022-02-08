"""
Collection of modules implementing losses.
"""

from fvcore.nn import sigmoid_focal_loss
import torch
from torch import nn

from models.build import MODELS


@MODELS.register_module()
class SigmoidFocalLoss(nn.Module):
    """
    Class implementing the sigmoid focal loss module.

    Attributes:
        alpha (float): Alpha value of the sigmoid focal loss function.
        gamma (float): Gamma value of the sigmoid focal loss function.
        weight (float): Factor weighting the sigmoid focal loss.
    """

    def __init__(self, alpha=0.25, gamma=2.0, weight=1.0):
        """
        Initializes the SigmoidFocalLoss module.

        Args:
            alpha (float): Alpha value of the sigmoid focal loss function (default=0.25).
            gamma (float): Gamma valud of the sigmoid focal loss function (default=2.0).
            weight (float): Factor weighting the sigmoid focal loss (default=1.0).
        """

        # Initialization of default nn.Module
        super().__init__()

        # Set attributes
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, pred_logits, tgt_labels, reduction='none'):
        """
        Forward method of the SigmoidFocalLoss module.

        Args:
            pred_logits (FloatTensor): Tensor with prediction logits of shape [*].
            tgt_labels (Tensor): Tensor with binary classification labels of shape [*].
            reduction (str): String specifying the reduction operation applied on element-wise losses (default='none').

        Returns:
            * If reduction is 'none':
                loss (FloatTensor): Tensor with element-wise sigmoid focal losses of shape [*]

            * If reduction is 'mean':
                loss (FloatTensor): Mean of tensor with element-wise sigmoid focal losses of shape [1].

            * If reduction is 'sum':
                loss (FloatTensor): Sum of tensor with element-wise sigmoid focal losses of shape [1].
        """

        # Get weighted sigmoid focal loss
        tgt_labels = tgt_labels.to(pred_logits.dtype)
        loss = self.weight * sigmoid_focal_loss(pred_logits, tgt_labels, self.alpha, self.gamma, reduction)

        return loss


@MODELS.register_module()
class SigmoidHillLoss(nn.Module):
    """
    Class implementing the sigmoid hill loss module.

    Attributes:
        low_left (float): Coordinate of low left constant to quadratic transition.
        mid_left (float): Coordinate of middle left quadratic to linear transition.
        up_left (float): Coordinate of up left linear to quadratic transition.
        top (float): Coordinate corresponding to the top of the hill.
        up_right (float): Coordinate of up right quadratic to linear transition.
        mid_right (float): Coordinae of middle right linear to quadratic transition.
        low_right (float): Coordinate of low right quadratic to constant transition.
        coeffs (List): List of size [8] with coefficients of the constant, linear and quadratic sub-functions.
    """

    def __init__(self, low_left=0.1, mid_left=0.2, up_left=0.4, top=0.5, up_right=0.6, mid_right=0.8, low_right=0.9):
        """
        Initializes the SigmoidHillLoss module.

        Args:
            low_left (float): Coordinate of low left constant to quadratic transition (default=0.1).
            mid_left (float): Coordinate of middle left quadratic to linear transition (default=0.2).
            up_left (float): Coordinate of up left linear to quadratic transition (default=0.4).
            top (float): Coordinate corresponding to the top of the hill (default=0.5).
            up_right (float): Coordinate of up right quadratic to linear transition (default=0.6).
            mid_right (float): Coordinae of middle right linear to quadratic transition (default=0.8).
            low_right (float): Coordinate of low right quadratic to constant transition (default=0.9).

        Raises:
            ValueError: Error when low left value is smaller than 0.
            ValueError: Error when low left value is greater than middle left value.
            ValueError: Error when middle left value is greater than up left value.
            ValueError: Error when up left value is greater than top value.
            ValueError: Error when top value is greater than up right value.
            ValueError: Error when up right value is greater than middle right value.
            ValueError: Error when middle right value is greater than low right value.
            ValueError: Error when low right value is greater than 1.
        """

        # Initialization of default nn.Module
        super().__init__()

        # Check inputs
        if low_left <= 0.0:
            error_msg = f"Low left value should be greater than 0 (got {low_left})."
            raise ValueError(error_msg)

        if low_left >= mid_left:
            error_msg = f"Middle left value ({mid_left}) should be greater than low left value ({low_left})."
            raise ValueError(error_msg)

        if mid_left >= up_left:
            error_msg = f"Up left value ({up_left}) should be greater than middle left value ({mid_left})."
            raise ValueError(error_msg)

        if up_left >= top:
            error_msg = f"Top value ({top}) should be greater than up left value ({up_left})."
            raise ValueError(error_msg)

        if top >= up_right:
            error_msg = f"Up right value ({up_right}) should be greater than top value ({top})."
            raise ValueError(error_msg)

        if up_right >= mid_right:
            error_msg = f"Middle right value ({mid_right}) should be greater than up right value ({up_right})."
            raise ValueError(error_msg)

        if mid_right >= low_right:
            error_msg = f"Low right value ({low_right}) should be greater than middle right value ({mid_right})."
            raise ValueError(error_msg)

        if low_right >= 1.0:
            error_msg = f"Low right value should be smaller than 1 (got {low_right})."
            raise ValueError(error_msg)

        # Set hill coordinate attributes
        self.low_left = low_left
        self.mid_left = mid_left
        self.up_left = up_left
        self.top = top
        self.up_right = up_right
        self.mid_right = mid_right
        self.low_right = low_right

        # Get coefficients of constant, linear and quadratic sub-functions
        self.coeffs = [0] * 8
        self.coeffs[0] = (0,)
        self.coeffs[1], f1 = self.get_quadratic(low_left, 0, 0, mid_left, 1)
        self.coeffs[2], f1 = self.get_linear(mid_left, f1, 1, up_left)
        self.coeffs[3], f1 = self.get_quadratic(up_left, f1, 1, top, 0)
        self.coeffs[4], f1 = self.get_quadratic(top, f1, 0, up_right, -1)
        self.coeffs[5], f1 = self.get_linear(up_right, f1, -1, mid_right)
        self.coeffs[6], f1 = self.get_quadratic(mid_right, f1, -1, low_right, 0)
        self.coeffs[7] = (f1,)

    @staticmethod
    def get_linear(x0, f0, df0, x1):
        """
        Get linear function with given properties, and apply it at coordinate x1.

        Args:
            x0 (float): Coordinate x0.
            f0 (float): Function value at x0.
            df0 (float): Derivative value at x0.
            x1 (float): Coordinate x1.

        Returns:
            coeffs (Tuple): Tuple containing the coefficients of the linear function.
            f1 (float): Function value at x1.
        """

        # Get coefficients of linear function
        a = df0
        b = f0 - a*x0
        coeffs = (a, b)

        # Get function value at x1
        f1 = a*x1 + b

        return coeffs, f1

    @staticmethod
    def get_quadratic(x0, f0, df0, x1, df1):
        """
        Get quadratic function with given properties, and apply it at coordinate x1.

        Args:
            x0 (float): Coordinate x0.
            f0 (float): Function value at x0.
            df0 (float): Derivative value at x0.
            x1 (float): Coordinate x1.
            df1 (float): Derivative value at x1.

        Returns:
            coeffs (Tuple): Tuple containing the coefficients of the quadratic function.
            f1 (float): Function value at x1.
        """

        # Get coefficients of quadratic function
        a = 0.5 * (df1-df0) / (x1-x0)
        b = df0 - 2*a*x0
        c = f0 - a*x0**2 - b*x0
        coeffs = (a, b, c)

        # Get function value at x1
        f1 = a*x1**2 + b*x1 + c

        return coeffs, f1

    @staticmethod
    def apply_linear(input, coeffs):
        """
        Apply linear function with given coefficients on given input.

        Args:
            input (FloatTensor): Input tensor of arbitrary shape.
            coeffs (Tuple): Tuple of size [2] containing the coefficients of the linear function.

        Returns:
            output (FloatTensor): Output tensor of same shape as input.

        Raises:
            ValueError: Error when the number of given coefficients is different from two.
        """

        # Check whether exactly two coefficients are provided
        if len(coeffs) != 2:
            error_msg = f"Two coefficients should be provided for linear function, but got {len(coeffs)}."
            raise ValueError(error_msg)

        # Apply linear function on input to get output
        a, b = coeffs
        output = a*input + b

        return output

    @staticmethod
    def apply_quadratic(input, coeffs):
        """
        Apply quadratic function with given coefficients on given input.

        Args:
            input (FloatTensor): Input tensor of arbitrary shape.
            coeffs (Tuple): Tuple of size [3] containing the coefficients of the quadratic function.

        Returns:
            output (FloatTensor): Output tensor of same shape as input.

        Raises:
            ValueError: Error when the number of given coefficients is different from three.
        """

        # Check whether exactly three coefficients are provided
        if len(coeffs) != 3:
            error_msg = f"Three coefficients should be provided for quadratic function, but got {len(coeffs)}."
            raise ValueError(error_msg)

        # Apply quadratic function on input to get output
        a, b, c = coeffs
        output = a*input**2 + b*input + c

        return output

    def forward(self, logits, reduction='mean'):
        """
        Forward method of the SigmoidHillLoss module.

        Args:
            logits (FloatTensor): Tensor with input logits of arbitrary shape.
            reduction (str): String specifying the reduction operation applied on losses tensor (default='mean').

        Returns:
            * If reduction is 'none':
                losses (FloatTensor): Tensor with element-wise sigmoid hill losses of same shape as input logits.

            * If reduction is 'mean':
                loss (FloatTensor): Mean of the tensor with element-wise sigmoid hill losses of shape [1].

            * If reduction is 'sum':
                loss (FloatTensor): Sum of the tensor with element-wise sigmoid hill losses of shape [1].

        Raises:
            ValueError: Error when invalid reduction string is provided.
        """

        # Compute sigmoid probabilities from input logits
        probs = torch.sigmoid(logits)

        # Get masks
        mask0 = probs < self.low_left
        mask1 = (probs >= self.low_left) & (probs < self.mid_left)
        mask2 = (probs >= self.mid_left) & (probs < self.up_left)
        mask3 = (probs >= self.up_left) & (probs < self.top)
        mask4 = (probs >= self.top) & (probs < self.up_right)
        mask5 = (probs >= self.up_right) & (probs < self.mid_right)
        mask6 = (probs >= self.mid_right) & (probs < self.low_right)
        mask7 = probs >= self.low_right

        # Compute sigmoid hill losses
        losses = torch.zeros_like(probs)
        losses[mask0] = self.coeffs[0][0]
        losses[mask1] = self.apply_quadratic(probs[mask1], self.coeffs[1])
        losses[mask2] = self.apply_linear(probs[mask2], self.coeffs[2])
        losses[mask3] = self.apply_quadratic(probs[mask3], self.coeffs[3])
        losses[mask4] = self.apply_quadratic(probs[mask4], self.coeffs[4])
        losses[mask5] = self.apply_linear(probs[mask5], self.coeffs[5])
        losses[mask6] = self.apply_quadratic(probs[mask6], self.coeffs[6])
        losses[mask7] = self.coeffs[7][0]

        # Apply reduction operation to tensor with sigmoid hill losses and return
        if reduction == 'none':
            return losses
        elif reduction == 'mean':
            loss = torch.mean(losses)[None]
            return loss
        elif reduction == 'sum':
            loss = torch.sum(losses)[None]
            return loss
        else:
            error_msg = f"Invalid reduction string '{reduction}'."
            raise ValueError(error_msg)
