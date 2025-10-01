import torch

from kan_tutorial.utils import (
    eval_basis_functions,
    get_coeff,
)


def update_grid(prev_grid, new_grid_size, k, preacts, postacts):
    """
    Updates grid.

    Args:
        prev_grid: number of splines x number of control points
        new_grid_size: new number of control points
        k: spline-order
        preacts: inputs to the grid
        postacts: current outputs to the grid.

    Returns:
        new_coef: New coefficients to maintain the behavior of the current spline-activation
        new_grid: new grid with new control points
    """
    coarse_grid_size = prev_grid.shape[-1]
    finer_grid_size = new_grid_size

    # learn the spline to predict the control points in the existing grid
    x_pos = prev_grid.transpose(1, 0)
    temp_grid = torch.linspace(-1, 1, steps=coarse_grid_size).unsqueeze(dim=0)
    temp_bases = eval_basis_functions(
        temp_grid.transpose(1, 0), temp_grid, k=1
    )  # linear interpolate into a smaller grid
    temp_coef = get_coeff(temp_bases, x_pos)

    # let's predict more control points using the above spline
    percentiles = torch.linspace(-1, 1, steps=finer_grid_size).unsqueeze(dim=1)
    percentiles_basis = eval_basis_functions(percentiles, temp_grid, k=1)
    new_grid = torch.einsum("ijk, bij-> bk", temp_coef, percentiles_basis).transpose(
        1, 0
    )

    # find the coefficients (predicitng the same postacts from preacts but using the new control points)
    new_bases = eval_basis_functions(preacts, new_grid, k)
    new_coef = get_coeff(new_bases, postacts)

    return new_coef, new_grid


def model_predict(params_kan, x_eval, k):
    """
    Implements prediction functions using raw KANs.
    """
    grids = params_kan["grids"]
    coeffs = params_kan["coeffs"]
    scale_bases = params_kan["scale_bases"]
    scale_splines = params_kan["scale_splines"]
    base_fns = params_kan["base_fns"]

    for idx in range(len(grids)):
        grid, coeff = grids[idx], coeffs[idx]

        with torch.no_grad():
            x = torch.einsum(
                "ijk, bij -> bk", coeff, eval_basis_functions(x_eval, grid, k)
            )
            if len(base_fns) > 0:
                base_fn, scale_sp, scale_base = (
                    base_fns[idx],
                    scale_splines[idx],
                    scale_bases[idx],
                )
                x = scale_base * base_fn(x_eval) + scale_sp * x

        y_pred = x_eval = x

    return y_pred
