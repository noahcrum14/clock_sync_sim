# --- Plotting ---
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List


def plot_results(history, coins):
    history = np.array(history)
    iters = history[:, 0]
    c_points = history[:, 1]
    f_c_vals = history[:, 2]
    d_points = history[:, 3]
    f_d_vals = history[:, 4]
    a_vals = history[:, 5]
    b_vals = history[:, 6]

    # Plot evolution of the internal points (c and d)
    plt.figure(figsize=(10, 6))
    plt.plot(iters, c_points, 'o-', label="c (internal point)")
    plt.plot(iters, d_points, 's-', label="d (internal point)")
    plt.xlabel("Iteration")
    plt.ylabel("Distance (m)")
    plt.title("Evolution of Internal Points in Golden Section Search")
    plt.legend()
    plt.grid(True)

    # Plot the corresponding coincidence probabilities at c and d
    plt.figure(figsize=(10, 6))
    plt.plot(iters, f_c_vals, 'o-', label="f(c) Coincidence Probability")
    plt.plot(iters, f_d_vals, 's-', label="f(d) Coincidence Probability")
    plt.xlabel("Iteration")
    plt.ylabel("Coincidence Probability")
    plt.title("Coincidence Probabilities at Internal Points")
    plt.legend()
    plt.grid(True)

    # Plot a histogram of the coincidence events from the final simulation
    plt.figure(figsize=(10, 6))
    plt.hist(coins, bins=2, edgecolor='black', rwidth=0.8)
    plt.xlabel("Coincidence Event (0 or 1)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Coincidence Events at Optimal Distance")
    plt.grid(True)


def plot_optimization_history(history: List[Tuple[float, float]], 
                             target: float = None,
                             param_name: str = "Parameter",
                             bounds: Tuple[float, float] = None,
                             save_path: str = None):
    """
    Visualize the optimization process with a futuristic space-age design.

    Args:
        history: List of (parameter_value, objective_value) tuples
        target: Target probability value to show as reference line
        param_name: Name of parameter being optimized for labels
        bounds: Initial search bounds for visualization
        save_path: Optional path to save the figure (shows plot if None)
    """
    plt.style.use("dark_background")  # Space-age dark mode

    fig, axs = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    fig.patch.set_facecolor('#121212')  # Deep black background
    
    # Extract values
    params, objectives = zip(*history)
    iterations = np.arange(len(history))

    # **Parameter value progression plot**
    axs[0].plot(iterations, params, marker='o', markersize=4, linestyle='-', lw=0.8, color='#00FFFF', alpha=0.85, label=param_name)
    axs[0].set_title(f"{param_name} Value Progression", color='white', fontsize=14, fontweight='bold')
    axs[0].set_xlabel("Iteration", color='white', fontsize=12)
    axs[0].set_ylabel(param_name, color='white', fontsize=12)
    axs[0].set_xlim(0, iterations[-1])
    axs[0].grid(True, linestyle='--', alpha=0.3)

    # Bounds visualization
    if bounds:
        axs[0].axhline(bounds[0], color='#FF00FF', linestyle='--', alpha=0.6, lw=1.5, label='Initial Bounds')
        axs[0].axhline(bounds[1], color='#FF00FF', linestyle='--', alpha=0.6, lw=1.5)

    axs[0].legend(frameon=False, fontsize=10, loc="best", facecolor='#121212')

    # **Objective value progression plot**
    axs[1].plot(iterations, objectives, marker='s', markersize=4, linestyle='-', lw=0.8, color='#FF00FF', alpha=0.85, label='Objective Value')
    if target is not None:
        axs[1].axhline(target, color='white', linestyle='--', lw=0.8, alpha=0.7, label='Target')

    axs[1].set_title("Objective Value Progression", color='white', fontsize=14, fontweight='bold')
    axs[1].set_xlabel("Iteration", color='white', fontsize=12)
    axs[1].set_ylabel("Coincidence Probability", color='white', fontsize=12)
    axs[1].set_xlim(0, iterations[-1])
    axs[1].grid(True, linestyle='--', alpha=0.3)
    
    axs[1].legend(frameon=False, fontsize=10, loc="best", facecolor='#121212')

    # Adjust layout & set a cyberpunk-inspired border
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3, top=0.92)

    # Save or show plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    else:
        plt.show()

