# Artificial-neural-networks-ANN-
Step-1: Randomly initialise the weights to small numbers close to 0 (but not 0)
Step-2: input the first observation of your dataset in the input layer, each feature in one input node
step-3: forward propagation from left to right, the neurons are activated in a way that the impact of each neuron's activation is limited by the weights. propagate the actions until getting the predicted result y.
step-4: compare the predicted result to the actual result. measure the generated error
step-5: back-propagation from right to left, the error is back propagated. Update the weights according to how much they are responsible for the error. The learning rate decides by how much we update the weights.
step-6: repeat steps 1 to 5 and pdate the weights after each observations (reinforcement learning) or:
repeat steps 1 to 5 but update the weights only after a batch of observations (Batch learning)
Step-7: when the whole training set passed through the ANN, that makes an epoch. redo more epochs
