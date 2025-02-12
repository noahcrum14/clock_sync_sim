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
from clock_sync_sim.config.settings import C_FIBER

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    print(f"Running simulation (Phase 1)")
    phase = 1
    optimal_distance_BB, history_BB, optimal_prob_BB, coins_BB, coin_index_BB, sch1, sch2 = run_full_simulation(args.config, phase)

    print(f"Optimal MDL Distance: {optimal_distance_BB:.2f} meters")
    print(f"Average Coincidence Probability: {optimal_prob_BB:.5f}")
    print(f"Coincidence Index: {coin_index_BB}")

    plt.show()

    print(f"Running simulation (Phase 2)")
    phase = 2
    optimal_distance_AA, history_AA, optimal_prob_AA, coins_AA, coin_index_AA, _, _ = run_full_simulation(args.config, phase)

    print(f"Optimal MDL Distance: {optimal_distance_AA:.2f} meters")
    print(f"Average Coincidence Probability: {optimal_prob_AA:.5f}")
    print(f"Coincidence Index: {coin_index_AA}")

    plt.show()

    f = 100 
    dT = 0.5*((coin_index_BB + coin_index_AA)*f + 1/C_FIBER*(optimal_distance_AA - optimal_distance_BB))

    print(f"Estimated time difference: {dT:.2f} ns")

    print(sch1, sch2)
    delta = sch2 - sch1 - dT
    print(f"Estimated offset: {np.mean(delta)} ns")
    np.save("data/delta.npy", delta)
    print("Simulation complete.")
