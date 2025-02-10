import numpy as np
import yaml
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Tuple, List, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from .models.party import Party
from .utils.calculations import get_coin_prob
from .config.settings import POL_DICT, DEFAULT_PARAMS
from .utils.optimization import coarse_search, GradientDescentOptimizer, GoldenSectionOptimizer, SPSAOptimizer
from .utils.visualization import plot_optimization_history
from .config.settings import SIGMA


def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from YAML file or use defaults"""
    default_config = DEFAULT_PARAMS

    if config_path:
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file) as f:
                return {**default_config, **yaml.safe_load(f)}

    return default_config


class SimulationEngine:
    def __init__(self, config: Dict = None):
        self.config = load_config() if config is None else config
        self.alice = self._create_party('Alice')
        self.bob = self._create_party('Bob')
        self.history = []

    def _create_party(self, party_id: str) -> Party:
        """Instantiate party with configuration parameters"""
        params = self.config['parties'][party_id.lower()]
        return Party(
            worldtime=0,
            party_id=party_id,
            source_type=params['source_type'],
            rep_rate=params['rep_rate'],
            delay=params['delay'],
            offset=params['offset']
        )

    def _propagate_schedules(self, bob_distance: float) -> Tuple[Dict, Dict]:
        """Generate and propagate emission schedules"""
        alice_schedule = self.alice.emission_schedule()
        bob_schedule = self.bob.emission_schedule()

        # Propagate through fibers
        alice_schedule = self.alice.propagate(
            alice_schedule,
            self.config['distance_alice_to_bob']
        )
        bob_schedule = self.bob.propagate(bob_schedule, bob_distance)

        return alice_schedule, bob_schedule

    def calculate_coincidence(self, bob_distance: float, n_runs: int = 1) -> float:
        """Calculate average coincidence probability over multiple runs"""
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self._single_run, bob_distance)
                       for _ in range(n_runs)]
            results = [f.result() for f in futures]

        # If results contain tuples, extract the first element from each tuple
        if isinstance(results[0], tuple):
            avg_coincidence = np.mean([r[0][0] for r in results])
            #print("Average Coincidence: ", avg_coincidence)
            return avg_coincidence

        return np.mean(results)

    def _single_run(self, bob_distance: float) -> float:
        """Single simulation run for a given distance"""
        alice_sched, bob_sched = self._propagate_schedules(bob_distance)
        coin_index, combined = self._coincidence_schedule(alice_sched, bob_sched)
        #combined = self._merge_schedules(alice_sched, bob_sched)
        paired_events = self._find_coincidences(combined)

        if not paired_events:
            return 0.0  # No coincidences detected

        return self._calculate_coin_rate(paired_events), coin_index

    def _merge_schedules(self, alice_sched: List, bob_sched: List) -> List:
        """Combine and sort schedules by time"""
        combined = []
        for event in alice_sched:
            combined.append({
                'party': 'Alice',
                't': event['t'],
                'pol': event['pol'],
                'photons': event['photons']
            })
        for event in bob_sched:
            combined.append({
                'party': 'Bob',
                't': event['t'],
                'pol': event['pol'],
                'photons': event['photons']
            })
        return sorted(combined, key=lambda x: x['t'])

    def _coincidence_schedule(self, alice_sched: List, bob_sched: List) -> List:
        window = self.config['processing_window']

        # Calculate the difference in first arrival times
        diff = alice_sched[0]['t'] - bob_sched[0]['t']
        
        if abs(diff) < window:
            coincident_index = 0
            good_events_A = alice_sched
            good_events_B = bob_sched
        elif diff > 0:
            for i in range(len(alice_sched)):
                diff_2 = alice_sched[0]['t'] - bob_sched[i]['t']
                if abs(diff_2) <= window:
                    coincident_index = i
                    good_events_A = alice_sched[i:]
                    good_events_B = bob_sched[:-i]
                    break
                else:
                    pass
        else:
            for i in range(len(bob_sched)):
                diff_2 = alice_sched[i]['t'] - bob_sched[0]['t']
                if abs(diff_2) <= window:
                    coincident_index = i
                    good_events_A = alice_sched[:-i]
                    good_events_B = bob_sched[i:]
                    break
                else:
                    pass
        merged_events = self._merge_schedules(good_events_A, good_events_B)
        return coincident_index, merged_events

    def _find_coincidences(self, schedule: List) -> List:
        """Pair events in a strict (0,1), (2,3), ... 
        manner within the processing window"""
        window = self.config['processing_window']
        pairs = []

        # Iterate with step size 2 to enforce strict pairing
        for i in range(0, len(schedule) - 1, 2):
            evt1, evt2 = schedule[i], schedule[i + 1]
            td = abs(evt1['t'] - evt2['t'])

            # Only pair if the events meet the time difference condition
            if td <= window:
                pairs.append((evt1, evt2))
        return pairs

    def _calculate_coin_rate(self, pairs: List) -> float:
        """Calculate coincidence rate for event pairs"""
        coincidents = []
        iter = 0
        for evt1, evt2 in pairs:
            td = evt1['t'] - evt2['t']
            mismatch = abs(np.dot(
                POL_DICT[evt1['pol']],
                POL_DICT[evt2['pol']]
            ))
            p_coin = get_coin_prob(1, 1, mismatch, td)
            #coin_count += np.random.uniform() < p_coin
            if np.random.uniform() < p_coin:
                coincidents.append(1)
            else:
                coincidents.append(0)
        coincidents = np.asarray(coincidents)
        coin_rate = p_coin#sum(coincidents) / len(pairs)
       
        return coin_rate, coincidents


def run_full_simulation(config_path: str = None) -> Tuple[float, List, float, List, float]:
    # Load config
    config = load_config(config_path)

    # Initialize simulation engine
    engine = SimulationEngine(config)

    # Target probability for optimization
    target_probability = 0#get_coin_prob(1, 1, np.cos(np.pi / 4), 0)
    post_selection_prob = get_coin_prob(1, 1, 0, 0)

    # Set bounds on search region
    bounds = (config['search_initial_low'], config['search_initial_high'])

    print(f"Searching for optimal MDL distance...")
    valley_bounds = coarse_search(engine.calculate_coincidence, bounds, 1/(2*SIGMA), num_points=20, samples_per_point=3)
    print(f"Valley Bounds: {valley_bounds}")
    # Select optimizer based on config
    optimizer_type = config.get('optimizer', 'golden_section')

    optimizers = {
        'golden_section': GoldenSectionOptimizer,
        'gradient_descent': GradientDescentOptimizer,
        'spsa': SPSAOptimizer
    }

    optimizer = optimizers[optimizer_type](
        target_prob=target_probability,
        objective_fn=engine.calculate_coincidence,  # Raw probability function
        bounds=valley_bounds,
        config=config
    )

    optimal_dist, history = optimizer.optimize()

    # Plot the results
    plot_optimization_history(
        history=history,
        target=target_probability,
        param_name="Bob's Distance",
        bounds=valley_bounds,
        save_path=f"Images/optimization_history_{round(time.time(), 4)}.png"
        )

    # Final verification run
    (final_prob, coins), coin_index = engine._single_run(optimal_dist)

    return optimal_dist, history, final_prob, coins, coin_index