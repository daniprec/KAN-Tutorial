import torch
import torch.nn as nn

from kan_tutorial.utils import eval_basis_functions


def initialize_KAN(KAN, k, grid_range, grid_size):
    """
    Returns the ingredients necessary to evaluate a KAN.
    Note: we use a single grid for all the activation functions.

    Args:
        KAN: List of integers corresponding to neurons in each layer
        k: spline order
        grid_range: range of grid. e.g, [-1, 1]
        grid_size: number of control points
    """
    grid = torch.linspace(grid_range[0], grid_range[1], steps=grid_size).unsqueeze(
        dim=0
    )

    # initialize the coefficients and symbolic functions
    coefs, masks, symbolic_functions = [], [], []
    for in_dim, out_dim in zip(KAN, KAN[1:]):
        coef = torch.zeros((in_dim, out_dim, grid_size + k - 1), requires_grad=True)
        mask = torch.ones((in_dim, out_dim))
        coefs.append(coef)
        masks.append(mask)

        symb_fns = [[lambda x: x for _ in range(out_dim)] for _ in range(in_dim)]
        symbolic_functions.append(symb_fns)

    return coefs, masks, symbolic_functions, grid


def eval_x_symbolically(x_eval, symbolic_fns):
    """
    Evaluates `x_eval` using functions specified in `symbolic_fns`.

    Args:
        x_eval: batch x number of dimensions
        symbolic_fns: list of list of symbolc functions.
    """
    in_dim = len(symbolic_fns)
    out_dim = len(symbolic_fns[0])
    postacts = []
    for i in range(in_dim):
        postacts_ = []
        for j in range(out_dim):
            symb_fn = symbolic_fns[i][j]
            postacts_.append(symb_fn(x_eval[:, i]))
        postacts.append(torch.stack(postacts_, dim=1))

    return torch.stack(postacts, dim=1)  # stack so that the input_dim is intact


def eval_KAN(x, coefs, masks, symbolic_functions, grid, k):
    """
    Acts as the model.forward to evaluate x according to the given KAN.

    Args:
        x: Batch x input_dimensions
        coefs: list of coefficients of size (in_dim, out_dim, h), where in_dim and out_dim are determined by the KAN structure, h is determined by the grid
        masks: list of masks to be used to combine spline activation and symbolic activations. Each mask is (in_dim, out_dim). A value of 1 means use spline.
        symbolic_functions: list of list symbolic functions. Each inner list represents symbolic function corresponding to a specific input dimension.
        grid: grid for spline estimation
        k: order of spline
    """
    x_in = x
    for coef, mask, symb_fns in zip(coefs, masks, symbolic_functions):
        # spline activations
        bases = eval_basis_functions(x_in, grid, k)
        y_sp = torch.einsum("ijk, bik -> bij", coef, bases)

        # symbolic activations
        y_symb = eval_x_symbolically(x_in, symb_fns)

        # combine the two outputs
        y = mask[None, ...] * y_sp + (1 - mask[None, ...]) * y_symb

        # add along the input dimensions
        x_in = y.sum(dim=1)  # B x output_dimension

    return x_in


def set_symbolic_fn(symbolic_functions, masks, i, j, k, fn, use_affine=False):
    """
    Sets the symbolic function in the layer i, neuron j's kth output.

    Args:
         symbolic_functions:  list of list symbolic functions. Each inner list represents symbolic function corresponding to a specific input dimension.
         masks: list of masks to be used to combine spline activation and symbolic activations. Each mask is (in_dim, out_dim). A value of 1 means use spline.
         i: layer index
         j: neuron index (input)
         k: output index
         fn: function to use
         use_affine: whether to use SymbolicKANLayer or not

    Returns:
        modified symbolic_functions and masks
    """
    if use_affine:
        symb_fn = SymbolicKANLayer(fn)
    else:
        symb_fn = fn

    symbolic_functions[i][j][k] = symb_fn
    masks[i].data[j][k] = 0

    return symbolic_functions, masks


class SymbolicKANLayer(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

        # learnable parameters
        self.affine = torch.nn.Parameter(torch.zeros(4), requires_grad=True)

    def forward(self, x_eval):
        return (
            self.affine[0] * self.fn(self.affine[1] * x_eval + self.affine[2])
            + self.affine[3]
        )

    def step(self, lr):
        """
        Performs a step of gradient descent using the learning rate `lr`
        """
        if self.affine.grad is not None:
            self.affine.data = self.affine.data - lr * self.affine.grad

    def zero_grad_(self):
        """
        Zeroes out the gradients in-place.
        """
        self.affine.grad.zero_()
