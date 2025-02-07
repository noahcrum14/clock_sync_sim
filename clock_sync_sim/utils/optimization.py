import numpy as np
from typing import Dict, Any, Tuple, List
from ..config.settings import SIGMA, C_FIBER

class GradientDescentOptimizer:
    """Gradient Descent optimizer to find the parameter that minimizes the squared error to the target probability."""
    def __init__(self, target_prob: float, objective_fn: callable, bounds: Tuple[float, float], config: Dict[str, Any]):
        self.target = target_prob
        self.objective = objective_fn
        self.bounds = bounds
        self.config = config
        self.learning_rate = config.get('learning_rate', 0.5)
        self.tolerance = config.get('tolerance', 1e-5)
        self.max_iter = config.get('max_iterations', 10)
        self.h = config.get('gradient_step_size', 1e-5)
        self.history = []

    def optimize(self) -> Tuple[float, List[Tuple[float, float]]]:
        history = []
        x = np.mean(self.bounds)
        iterations = 0
        for _ in range(self.max_iter):
            iterations += 1
            fx = self.objective(x)
            print("Value fx:", fx)
            history.append((x, fx))
            
            x_plus = np.clip(x + self.h, *self.bounds)
            x_minus = np.clip(x - self.h, *self.bounds)
            f_plus = self.objective(x_plus)
            f_minus = self.objective(x_minus)
            
            gradient = (f_plus - f_minus) / (2 * self.h)
            if abs(gradient) < 1e-5:
                print("Flat grad stepping further")
                delta = 1/SIGMA; 
            else:
                delta = 10*self.learning_rate * (fx - self.target) * gradient
            x_new = np.clip(x - delta, *self.bounds)
            print("Update to delay time: ", (x-x_new)/C_FIBER)
            
            if abs(x_new - x) < self.tolerance:
                x = x_new
                break
            x = x_new
        
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
