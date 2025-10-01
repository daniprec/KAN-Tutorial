import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.widgets import Slider

from kan_tutorial.utils import eval_basis_functions


def compute_curve(ctrl_x: np.ndarray, k: int, num_samples: int = 600):
    # Use control x-positions as the knot grid; fit coefficients to match control y-values
    grid = torch.tensor(ctrl_x, dtype=torch.float32).unsqueeze(0)

    x_dense = torch.linspace(0.0, 1.0, steps=num_samples).unsqueeze(1)
    bases_dense = eval_basis_functions(x_dense, grid, int(k), extend=False)  # [N, 1, H]
    y_dense = bases_dense.squeeze(1).mean(dim=1, keepdim=True)  # [N, 1]
    return x_dense.squeeze(1).numpy(), y_dense.numpy()


def main():
    # Initial settings
    num_ctrl = 10
    ctrl_x = np.linspace(0.0, 1.0, num_ctrl)
    ctrl_y = np.ones(num_ctrl) / 2
    k_init = 3

    fig, ax = plt.subplots(figsize=(9, 5))
    plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.20)

    x_curve, y_curve = compute_curve(ctrl_x, k_init)
    (curve_line,) = ax.plot(x_curve, y_curve, label="B-spline", lw=2)
    (curve_highlight,) = ax.plot(
        [], [], c="orange", lw=3, label="Affected segment", zorder=2
    )
    points = ax.scatter(
        ctrl_x, ctrl_y, c="crimson", s=50, zorder=3, label="Control points"
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 2)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    # Widgets: order (degree) slider and interval index slider
    ax_k = plt.axes([0.08, 0.10, 0.30, 0.03])
    s_k = Slider(
        ax=ax_k, label="Order k", valmin=0, valmax=5, valinit=k_init, valstep=1
    )

    # Drag handling
    state = {"drag_idx": None}

    def redraw(highlight_idx=None):
        k_val = int(s_k.val)
        x_c, y_c = compute_curve(ctrl_x, k_val)
        curve_line.set_data(x_c, y_c)
        if highlight_idx is not None:
            idx_left, idx_right = highlight_idx
            mask = (x_c >= ctrl_x[idx_left]) & (x_c <= ctrl_x[idx_right])
            curve_highlight.set_data(x_c[mask], y_c[mask])
        else:
            curve_highlight.set_data([], [])
        # update scatter offsets
        points.set_offsets(np.c_[ctrl_x, ctrl_y])
        fig.canvas.draw_idle()

    def on_press(event):
        if event.inaxes != ax:
            return
        # find nearest control point within a small distance
        xy = np.c_[ctrl_x, ctrl_y]
        dx = xy[:, 0] - event.xdata
        dy = xy[:, 1] - event.ydata
        d2 = dx * dx + dy * dy
        idx = int(np.argmin(d2))
        if np.sqrt(d2[idx]) < 0.1:  # pick threshold (data units)
            state["drag_idx"] = idx

    def on_release(event):
        state["drag_idx"] = None

    def on_move(event):
        idx = state.get("drag_idx")
        if (idx is None) or (idx == 0) or (idx == num_ctrl - 1):
            return
        if event.inaxes != ax:
            return
        # Move along X and Y. X is clamped between neighbors to avoid crossing
        x_new = float(event.xdata)
        # global bounds
        x_new = max(0.0, min(1.0, x_new))
        # neighbor bounds
        eps = 1e-2
        if idx > 0:
            x_new = max(x_new, float(ctrl_x[idx - 1]) + eps)
        if idx < len(ctrl_x) - 1:
            x_new = min(x_new, float(ctrl_x[idx + 1]) - eps)
        ctrl_x[idx] = x_new
        # highlight which part of the curve is being affected
        idx_left = max(0, idx - s_k.val)
        idx_right = min(num_ctrl - 1, idx + s_k.val)
        highlight_idx = (idx_left, idx_right)
        redraw(highlight_idx=highlight_idx)

    def on_k_change(val):
        redraw(highlight_idx=None)

    s_k.on_changed(on_k_change)

    cid_press = fig.canvas.mpl_connect("button_press_event", on_press)
    cid_release = fig.canvas.mpl_connect("button_release_event", on_release)
    cid_move = fig.canvas.mpl_connect("motion_notify_event", on_move)

    plt.show()


if __name__ == "__main__":
    main()
