from snn.neurons.LeakyIntegrateAndFireNeuron import LeakyIntegrateAndFireNeuron
from snn.learning.stdp import STDP
from snn.network.snn import SNN

net = SNN(3, [2, 4, 5], 6, LeakyIntegrateAndFireNeuron, STDP())
for _ in range(2):
    print('Input', net.layers[0][0].weights)
    print('Final', net.solve([ 1, 1, 1]))
    print(5 * "*")
print('Input', net.layers[0][0].weights)
print('Final', net.solve([ 1, 1, 1]))
