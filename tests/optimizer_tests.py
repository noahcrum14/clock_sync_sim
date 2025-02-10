from scipy.optimize import annealing
from ..clock_sync_sim.core import SimulationEngine

objective_function = engine.calculate 
result = annealing(objective_function, bounds=[(-10, 10), (-10, 10)])
print("Global minimum:", result.x)
