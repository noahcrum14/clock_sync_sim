import numpy as np
from typing import Dict, Any, Tuple, List
from tqdm import tqdm
from joblib import Parallel, delayed
from ..config.settings import SIGMA, C_FIBER


"""
def coarse_search(objective_fn, bounds, valley_width, num_points=20, samples_per_point=5):
    x_samples = np.linspace(bounds[0], bounds[1], num_points)
    means = []
    for x in tqdm(x_samples, desc="Performing coarse search"):
        # Average multiple evaluations to mitigate noise
        f_vals = [objective_fn(x) for _ in range(samples_per_point)]
        means.append(np.mean(f_vals))
    valley_center = x_samples[np.argmin(means)]
    return [valley_center - valley_width/2, valley_center + valley_width/2]
"""

def coarse_search(objective_fn, bounds, valley_width, num_points=20, samples_per_point=5, n_jobs=-1):
    x_samples = np.linspace(bounds[0], bounds[1], num_points)
    means = []
    for x in tqdm(x_samples, desc="Performing coarse search"):
        # Parallelize the multiple evaluations per sample point
        f_vals = Parallel(n_jobs=n_jobs)(delayed(objective_fn)(x) for _ in range(samples_per_point))
        means.append(np.mean(f_vals))
    valley_center = x_samples[np.argmin(means)]
    return [valley_center - valley_width/2, valley_center + valley_width/2]


class GradientDescentOptimizer:
    """Gradient Descent optimizer with decaying learning rate for minimizing squared error to the target probability."""
    
    def __init__(self, target_prob: float, objective_fn: callable, bounds: Tuple[float, float], config: Dict[str, Any]):
        self.target = target_prob
        self.objective = objective_fn
        self.bounds = bounds
        self.config = config
        self.initial_learning_rate = config.get('learning_rate', 0.1)  # Initial learning rate
        self.tolerance = config.get('tolerance', 1e-5)
        self.max_iter = config.get('max_iterations', 25)
        self.h = config.get('gradient_step_size', 1e-5)
        self.decay_rate = config.get('decay_rate', 1e-2)  # Controls learning rate decay
        self.history = []
        self.momentum = 0  # Initialize momentum
        self.target_tolerance = 0.01 # Tolerance for target probability

    def _compute_learning_rate(self, iteration: int, fx: float) -> float:
        """Compute a more stable decaying learning rate."""
        decay_factor = 1 / (1 + self.decay_rate * iteration)  # Linear decay instead of exponential
        return max(self.initial_learning_rate * decay_factor, 1e-3)  # Ensure minimum step size

    def optimize(self) -> Tuple[float, List[Tuple[float, float]]]:
        """Perform optimization with gradient descent and decaying learning rate."""
        history = []
        x = np.mean(self.bounds)

        for iter_num in range(self.max_iter):
            fx = self.objective(x)
            print(f"Iteration {iter_num+1}: x = {x:.6f}, fx = {fx:.6f}")

            history.append((x, fx))
            
            # Compute numerical gradient
            #self.h = 10 #max(1e-5, abs(x) * 1e-3)  # Scale step with x

            x_plus = np.clip(x + self.h, *self.bounds)
            x_minus = np.clip(x - self.h, *self.bounds)
            f_plus = self.objective(x_plus)
            f_minus = self.objective(x_minus)
            
            gradient = (f_plus - f_minus) / (2 * self.h)
            print("Gradient:", {gradient})

            learning_rate = self._compute_learning_rate(iter_num, fx)

            if abs(gradient) < 1e-3:  # Flat gradient
                if abs(fx - self.target) < self.target_tolerance:
                    print(f"Converged to within {self.target_tolerance*100}% of target probability.")
                    break
                else:
                    self.momentum += 1  # Accumulate momentum
                    delta = self.momentum * np.sign(fx - self.target)
            else:
                self.momentum = 0  # Reset momentum when gradient is active
                delta = learning_rate * (fx - self.target) * gradient * 1/(SIGMA**2 * C_FIBER**2)


            # Update parameter
            x_new = np.clip(x - delta, *self.bounds)
            print(f"Update: Δx = {(x-x_new):.6f}, Learning rate: {learning_rate:.6f}\n")
            
            x = x_new

            # Check convergence
            if abs(fx - self.target) < self.target_tolerance:
                print(f"Converged to within {self.target_tolerance*100}% of target probability.")
                break
        
        # Final function evaluation
        fx = self.objective(x)
        history.append((x, fx))
        return x, history
    

class GoldenSectionOptimizer:
    """Golden Section Search optimizer to find the extremum (min/max) of the objective function."""
    def __init__(self, target_prob: float, objective_fn: callable, bounds: Tuple[float, float], config: Dict[str, Any]):
        self.objective = objective_fn
        self.target = target_prob
        self.bounds = bounds
        self.config = config
        self.tolerance = config.get('tolerance', 1e-5)
        self.target_tolerance = 0.001
        self.max_iter = config.get('max_iterations', 50)
        self.golden_ratio = (np.sqrt(5) - 1) / 2
        self.mode = config.get('golden_section_mode', 'min')
        self.history = []

    def optimize(self) -> Tuple[float, List[Tuple[float, float]]]:
        a, b = self.bounds
        gr = self.golden_ratio
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        
        fc = self.objective(c)
        fd = self.objective(d)
        self.history.extend([(c, fc), (d, fd)])

        for i in tqdm(range(self.max_iter), desc="Performing Golden Section Search"):
            if abs(b - a) < self.tolerance or abs((fc + fd) / 2 - self.target) < self.target_tolerance:
                break

            if self.mode == 'min':
                if fc < fd:
                    b, d, fd = d, c, fc
                    c = b - gr * (b - a)
                    fc = self.objective(c)
                else:
                    a, c, fc = c, d, fd
                    d = a + gr * (b - a)
                    fd = self.objective(d)
            else:
                if fc > fd:
                    b, d, fd = d, c, fc
                    c = b - gr * (b - a)
                    fc = self.objective(c)
                else:
                    a, c, fc = c, d, fd
                    d = a + gr * (b - a)
                    fd = self.objective(d)
            self.history.extend([(c, fc), (d, fd)])

        optimal_x = (a + b) / 2
        optimal_f = self.objective(optimal_x)
        self.history.append((optimal_x, optimal_f))
        return optimal_x, self.history


class SPSAOptimizer:
    """Simultaneous Perturbation Stochastic Approximation optimizer for noisy objective functions."""
    def __init__(self, target_prob: float, objective_fn: callable, bounds: Tuple[float, float], config: Dict[str, Any]):
        self.target = target_prob
        self.target_tolerance = 0.001
        self.objective = objective_fn
        self.bounds = bounds
        self.config = config
        self.tolerance = config.get('tolerance', 1e-5)
        self.max_iter = config.get('max_iterations', 100)
        self.a = config.get('spsa_a', 2.0)
        self.c = config.get('spsa_c', 0.1)
        self.alpha = config.get('spsa_alpha', 0.602)
        self.gamma = config.get('spsa_gamma', 0.101)
        self.A = config.get('spsa_A', 0.1 * self.max_iter)
        self.history = []

    def optimize(self) -> Tuple[float, List[Tuple[float, float]]]:
        x = np.mean(self.bounds)
        history = []

        for k in range(1, self.max_iter + 1):
            ak = self.a / (k + self.A) ** self.alpha
            ck = self.c / k ** self.gamma
            delta = np.random.choice([-1, 1])

            x_plus = np.clip(x + ck * delta, *self.bounds)
            f_plus = self.objective(x_plus)
            x_minus = np.clip(x - ck * delta, *self.bounds)
            f_minus = self.objective(x_minus)
            history.extend([(x_plus, f_plus), (x_minus, f_minus)])

            ghat = (f_plus - f_minus) / (2 * ck * delta)
            ghat = np.clip(ghat , *[-3, 3])

            if abs(ghat) < 1e-3:  # Flat gradient
                print("Flat gradient detected. Adjusting step size.")
                ghat = 2  # Add momentum
                x_new = np.clip(x - ak * ghat, *self.bounds)         
            else:   
                x_new = np.clip(x - ak * ghat, *self.bounds)
            
            if abs((f_plus + f_minus)/2 - self.target) < self.target_tolerance:
                    print(f"Converged to within {self.target_tolerance*100}% of target probability.")
                    break
            
            if abs(x_new - x) < self.tolerance:
                x = x_new
                break
            x = x_new

        final_f = self.objective(x)
        history.append((x, final_f))
        return x, history
