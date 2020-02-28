#!/usr/bin/python3

from numpy import exp, array, random, dot
training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
training_set_outputs = array([[0, 1, 1, 0]]).T
random.seed(1)
synaptic_weights = 2 * random.random((3, 1)) - 1
#print(synaptic_weights)
for iteration in xrange(10000):
    output = 1 / (1 + exp(-(dot(training_set_inputs, synaptic_weights))))
    #print('output:')
    #print(output)
    synaptic_weights += dot(training_set_inputs.T, (training_set_outputs - output) * output * (1 - output))
#print(synaptic_weights)
# print 1 / (1 + exp(-(dot(array([0, 0, 0]), synaptic_weights))))
result = 1 / (1 + exp(-(dot(array([0, 1, .8]), synaptic_weights))))
print("result:")
print(result)
