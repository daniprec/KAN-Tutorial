import numpy as np
import torch
import torch.nn.functional as F

from kan_tutorial.utils import eval_basis_functions


def single_stacked_kan_training(
    x_training,
    y_training,
    x_test,
    y_test,
    model_params=None,
    lr=0.1,
    k=2,
    n_layers=2,
    grid_sizes=[],
    grid_ranges=[],
    early_stopping_imrpovement_threshold=200,
    early_stopping_iterations=1e4,
    verbose=False,
    grid_range=[-1, 1],
    use_scales=False,
):
    """
    Trains a KAN of shape [1, 1, 1, ...1] with `n_layers` layers.
    Args:
        x_training: Training inputs; number of samples x number of input dimensions
        y_training: Training targets; number of samples x 1
        x_test: Test inputs; number of samples x number of input dimensions
        y_test: Test targets; number of samples x 1
        model_params: Parameters of the model. Used in the Part 3 of the tutorial to continue training from an existing set of parameters.
        lr: learning rate
        k: spline-order
        n_layers: number of layers in the KAN
        grid_sizes: Number of control points for each spline in the stack
        grid_ranges: Grid ranges for each spline in the stack
        early_stopping_improvement_threshold: Number of iterations after which we can stop if there is no improvement in the validation loss
        early_stopping_iterations: Maximum number of iterations
        verbose: Whether to print the intermediate losses or not
        grid_range: Range of grids
        use_scales: Whether to use the scaling parameters (see section 2. )

    """
    if grid_sizes == []:
        grid_sizes = [10] * n_layers

    if grid_ranges == []:
        grid_ranges = [[-1, 1]] * n_layers

    if not model_params:
        grids, coeffs, scale_bases, scale_splines, base_fns = [], [], [], [], []
        for idx in range(n_layers):
            grid = torch.linspace(
                grid_ranges[idx][0], grid_ranges[idx][1], steps=grid_sizes[idx]
            ).unsqueeze(dim=0)
            grids.append(grid)

            coeff = torch.zeros((1, grid_sizes[idx] + k - 1, 1), requires_grad=True)
            coeffs.append(coeff)

            if use_scales:
                base_fn = torch.nn.SiLU()
                scale_base = torch.nn.Parameter(
                    torch.ones(x_training.shape[-1])
                ).requires_grad_(True)
                scale_spline = torch.nn.Parameter(
                    torch.ones(x_training.shape[-1])
                ).requires_grad_(True)

                scale_bases.append(scale_base)
                scale_splines.append(scale_spline)
                base_fns.append(base_fn)
    else:
        grids = model_params["grids"]
        coeffs = model_params["coeffs"]
        scale_bases = model_params["scale_bases"]
        scale_splines = model_params["scale_splines"]
        base_fns = model_params["base_fns"]

    losses = {"train": [], "val": []}
    best_loss = np.inf
    n_no_improvements = 0
    i = 0
    all_xs = []
    while True:
        x = x_training
        xs = []
        for idx in range(n_layers):
            bases = eval_basis_functions(x, grids[idx], k)
            x_ = torch.einsum("ijk, bij->bk", coeffs[idx], bases)
            if use_scales:
                base_transformed_x = base_fns[idx](
                    x
                )  # transformation of the original x
                x = base_transformed_x * scale_bases[idx] + x_ * scale_splines[idx]
            else:
                x = x_

            xs.append(x.detach())

        all_xs.append(xs)

        y_pred = x
        loss = torch.mean(torch.pow(y_pred - y_training, 2))
        loss.backward()
        losses["train"].append(loss.item())

        # Gradient descent step
        for params in coeffs + scale_bases + scale_splines:
            params.data = params.data - lr * params.grad
            params.grad.zero_()

        # evaluate validation loss
        with torch.no_grad():
            x = x_test
            for idx in range(n_layers):
                bases = eval_basis_functions(x, grids[idx], k)
                x_ = torch.einsum("ijk, bij->bk", coeffs[idx], bases)
                if use_scales:
                    base_transformed_x = base_fns[idx](
                        x
                    )  # transformation of the original x
                    x = base_transformed_x * scale_bases[idx] + x_ * scale_splines[idx]
                else:
                    x = x_
            y_pred_test = x
            val_loss = torch.mean(torch.pow(x - y_test, 2))

            losses["val"].append(val_loss.item())

        if i % 100 == 0 and verbose:
            print(
                f"Val loss: {val_loss.item(): 0.5f}\tTrain loss: {loss.item(): 0.5f}\tBest Val loss:{best_loss: 0.5f}"
            )

        if best_loss > val_loss.item():
            best_loss = val_loss.item()
            best_model = (coeffs, base_fns, scale_bases, scale_splines)
            n_no_improvements = 0
        else:
            n_no_improvements += 1
            if n_no_improvements > early_stopping_imrpovement_threshold:
                print("Stopping: No further improvements...")
                break

        i += 1
        if i > early_stopping_iterations:
            print("Stopping: Iteration limit reached...")
            break

    model_params = {
        "grids": grids,
        "coeffs": best_model[0],
        "scale_bases": best_model[2],
        "scale_splines": best_model[3],
        "base_fns": best_model[1],
    }
    return model_params, y_pred_test, losses, all_xs


def single_stacked_mlp_training(
    x_training,
    y_training,
    x_test,
    y_test,
    lr,
    layer_sizes,
    early_stopping_imrpovement_threshold=200,
    early_stopping_iterations=1e4,
    verbose=True,
):
    """Trains MLP similar to the function above."""

    layer_sizes = [1] + layer_sizes + [1]
    weights, biases = [], []
    n_layers = len(layer_sizes)

    # Define MLP weights and biases
    for idx in range(n_layers - 1):
        w = torch.randn(layer_sizes[idx], layer_sizes[idx + 1], requires_grad=True)
        weights.append(w)

        b = torch.zeros(layer_sizes[idx + 1], requires_grad=True)
        biases.append(b)

    losses = {"train": [], "val": []}
    best_loss = np.inf
    n_no_improvements = 0
    i = 0
    while True:
        x = x_test
        for weight, bias in zip(weights, biases):
            x = F.linear(x, weight.t(), bias)
            x = F.silu(x)  # relu might not work better here

        y_pred = x
        loss = torch.mean(torch.pow(y_pred - y_test, 2))
        loss.backward()
        losses["train"].append(loss.item())

        # Perform gradient descent
        for weight, bias in zip(weights, biases):
            weight.data = weight.data - lr * weight.grad
            weight.grad.zero_()

            bias.data = bias.data - lr * bias.grad
            bias.grad.zero_()

        # evaluate validation loss
        with torch.no_grad():
            x = x_test
            for weight, bias in zip(weights, biases):
                x = F.linear(x, weight.t(), bias)
                x = F.silu(x)  # relu might not work better here
            y_pred_test = x
            val_loss = torch.mean(torch.pow(x - y_test, 2))
            losses["val"].append(val_loss.item())

        if i % 100 == 0 and verbose:
            print(
                f"Val loss: {val_loss.item(): 0.5f}\tTrain loss: {loss.item(): 0.5f}\tBest Val loss:{best_loss: 0.5f}"
            )

        if best_loss > val_loss.item():
            best_loss = val_loss.item()
            best_model = [weights, biases]
            n_no_improvements = 0
        else:
            n_no_improvements += 1
            if n_no_improvements > early_stopping_imrpovement_threshold:
                print("Stopping: No further improvements...")
                break

        i += 1
        if i > early_stopping_iterations:
            print("Stopping: Iteration limit reached...")
            break

    return best_model, y_pred_test, losses
