import numpy as np
from typing import Dict, Any, Tuple, List
from ..config.settings import SIGMA, C_FIBER


class GradientDescentOptimizer:
    """Gradient Descent optimizer with decaying learning rate for minimizing squared error to the target probability."""
    
    def __init__(self, target_prob: float, objective_fn: callable, bounds: Tuple[float, float], config: Dict[str, Any]):
        self.target = target_prob
        self.objective = objective_fn
        self.bounds = bounds
        self.config = config
        self.initial_learning_rate = config.get('learning_rate', 10)  # Initial learning rate
        self.tolerance = config.get('tolerance', 1e-5)
        self.max_iter = config.get('max_iterations', 80)
        self.h = config.get('gradient_step_size', 1e-5)
        self.decay_rate = config.get('decay_rate', 0.1)  # Controls learning rate decay
        self.history = []
        self.momentum = 0  # Initialize momentum

    def _compute_learning_rate(self, iteration: int, fx: float) -> float:
        """Compute a more stable decaying learning rate."""
        decay_factor = 1 / (1 + self.decay_rate * iteration)  # Linear decay instead of exponential
        return max(self.initial_learning_rate * decay_factor, 1e-3)  # Ensure minimum step size


    def optimize(self) -> Tuple[float, List[Tuple[float, float]]]:
        """Perform optimization with gradient descent and decaying learning rate."""
        history = []
        x = self.bounds[1]#np.mean(self.bounds)

        for iter_num in range(self.max_iter):
            fx = self.objective(x)
            print(f"Iteration {iter_num+1}: x = {x:.6f}, fx = {fx:.6f}")

            history.append((x, fx))
            
            # Compute numerical gradient
            self.h = 10 #max(1e-5, abs(x) * 1e-3)  # Scale step with x

            x_plus = np.clip(x + self.h, *self.bounds)
            x_minus = np.clip(x - self.h, *self.bounds)
            f_plus = self.objective(x_plus)
            f_minus = self.objective(x_minus)
            
            gradient = (f_plus - f_minus) / (2 * self.h)
            print("Gradient:", {gradient})

            learning_rate = self._compute_learning_rate(iter_num, fx)

            """
            if abs(gradient) < 1e-2:
                print("Flat gradient detected. Adjusting step size.")
                delta = 10  # Small step to move out of flat regions
            else:
                # Compute decaying learning rate
                delta = learning_rate * (fx - self.target) * gradient * 1/(SIGMA**2 * C_FIBER**2)
            """

            if abs(gradient) < 1e-3:  # Flat gradient
                self.momentum += 5  # Accumulate momentum
                delta = self.momentum * np.sign(fx - self.target)
            else:
                self.momentum = 0  # Reset momentum when gradient is active
                delta = learning_rate * (fx - self.target) * gradient * 1/(SIGMA**2 * C_FIBER**2)


            # Update parameter
            x_new = np.clip(x - delta, *self.bounds)
            print(f"Update: Δx = {(x-x_new):.6f}, Learning rate: {learning_rate:.6f}\n")
            
            # Check convergence
            if abs(x_new - x) < self.tolerance:
                x = x_new
                break
            
            x = x_new
        
        # Final function evaluation
        fx = self.objective(x)
        history.append((x, fx))
        return x, history
    

class GoldenSectionOptimizer:
    """Golden Section Search optimizer to find the extremum (min/max) of the objective function."""
    def __init__(self, target_prob: float, objective_fn: callable, bounds: Tuple[float, float], config: Dict[str, Any]):
        self.objective = objective_fn
        self.bounds = bounds
        self.config = config
        self.tolerance = config.get('tolerance', 1e-5)
        self.max_iter = config.get('max_iterations', 10)
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

        for _ in range(self.max_iter):
            if abs(b - a) < self.tolerance:
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
        self.objective = objective_fn
        self.bounds = bounds
        self.config = config
        self.tolerance = config.get('tolerance', 1e-5)
        self.max_iter = config.get('max_iterations', 100)
        self.a = config.get('spsa_a', 1.0)
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

            ghat = ((f_plus - self.target)**2 - (f_minus - self.target)**2) / (2 * ck * delta)
            x_new = np.clip(x - ak * ghat, *self.bounds)

            if abs(x_new - x) < self.tolerance:
                x = x_new
                break
            x = x_new

        final_f = self.objective(x)
        history.append((x, final_f))
        return x, history
