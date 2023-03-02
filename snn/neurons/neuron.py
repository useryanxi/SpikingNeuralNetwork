import numpy as np

"""
Basic neuron model with shared instantiation
"""
class Neuron:
  def __init__(self, weights=0):
    np.seterr(all='ignore')
    self.type = "Base"
    self.input = 0
    self.value = 0
    self.output = 0
    self.threshold = 0
    self.fired = False
    self.potential = 0
    self.weights = np.array([self.init_weight(weights) for _ in range(weights)], dtype='float64')

  def __str__(self) -> str:
    return f"Neuron {self.type}: [info missing]"

  def fire(self):
    self.fired = self.value > self.threshold
    if self.fired:
      self.value = 0
    return int(self.fired)

  def init_weight(self, num_weights):
    return np.random.uniform(-(2 / num_weights), (2 / num_weights))

  def solve(self):
    raise NotImplementedError("A neuron model needs a solve method")
