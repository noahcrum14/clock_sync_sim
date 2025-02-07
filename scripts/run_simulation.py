import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

# Add package root to Python path
package_root = Path(__file__).parent.parent
sys.path.append(str(package_root))

from clock_sync_sim.core import run_full_simulation
from clock_sync_sim.utils.visualization import plot_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    print("Running simulation.")
    optimal_distance, history, optimal_prob, coins, coin_index = run_full_simulation(args.config)

    print(f"Optimal MDL Distance: {optimal_distance:.2f} meters")
    print(f"Average Coincidence Probability: {optimal_prob:.5f}")
    print(f"Coincidence Index: {coin_index}")

    #plot_results(history, coins)
    plt.show()