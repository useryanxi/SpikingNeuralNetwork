from snn.neurons.LeakyIntegrateAndFireNeuron import LeakyIntegrateAndFireNeuron
from snn.learning.stdp import STDP
from snn.network.snn import SNN

net = SNN(3, [2, 4, 5], 6, LeakyIntegrateAndFireNeuron, STDP())
counter = int(input("How many loops : "))
for _ in range(counter):
    print("Input", net.layers[0][0].weights)
    print("Final", net.solve([1, 1, 1]))
    print(5 * "*")
# Weights
print("Weights:")
for i, layer in enumerate(net.layers):
    if i == 0:
        print("Input", layer[0].weights)
    elif layer is net.layers[-1]:
        print("Output", layer[0].weights)
    else:
        print(f"Hidden layer {i}", layer[0].weights)
