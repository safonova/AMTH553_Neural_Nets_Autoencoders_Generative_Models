# ps3_functions.py
# CPSC 453 -- Problem Set 3
#

# Sasha Safonova

# This script contains pytorch shells for a Logistic regression model, a feed forward network, and an autoencoder.
#
from torch.nn.functional import softmax
from torch import optim, nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import torch

class LogisticRegression(nn.Module): # initialize a pytorch neural network module
    def __init__(self): # initialize the model
        super(LogisticRegression, self).__init__() # call for the parent class to initialize
        # you can define variables here that apply to the entire model (e.g. weights, biases, layers...)
        # this model only has two parameters: the weight, and the bias.
        # here's how you can initialize the weight:
        # W = nn.Parameter(torch.zeros(shape)) # this creates a model parameter out of a torch tensor of the specified shape
        # ... torch.zeros is much like numpy.zeros, except optimized for backpropogation.
        # We make it a model parameter and so it will be updated by gradient descent.
        self.Nfeatures = 784
        self.Noutcomes = 10
        weight_shape = [self.Nfeatures, self.Noutcomes]
        self.Nsamples = 1
        bias_shape = [self.Nsamples, self.Noutcomes]

        self.W = nn.Parameter(torch.zeros(weight_shape))
        # create a bias variable here
        self.bias = nn.Parameter(torch.zeros(bias_shape))

    def forward(self, x):
        """
        this is the function that will be executed when we call the logistic regression on data.
        INPUT:
            x, an MNIST image represented as a tensor of shape 784
        OUTPUT:
            softmax_probabilities, a tensor of shape 10
        """
        softmax_probabilities = torch.matmul(x, self.W)
        softmax_probabilities += self.bias
        softmax_probabilities = torch.reshape(softmax_probabilities, [self.Nsamples, 10])

        return softmax_probabilities

class FeedForwardNet(nn.Module):
    """ Simple feed forward network with one hidden layer."""
    # Here, you should place an exact copy of the code from the LogisticRegression class, with two modifications:
    # 1. Add another weight and bias vector to represent the hidden layer
    # 2. In the forward function, add some type of nonlinearity to the output of the first layer, then pass it onto the hidden layer.

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.lin1 = nn.Linear(784, 1000)
        # define additional layers here


    def encode(self, x):
        x = self.lin1(x)
        # ... additional layers, plus possible nonlinearities.
        return x

    def decode(self, z):
        # ditto, but in reverse
        return

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)




def train(model,loss_fn, optimizer, train_loader, test_loader):
    """
    This is a standard training loop, which leaves some parts to be filled in.
    INPUT:
    :param model: an untrained pytorch model
    :param loss_fn: e.g. Cross Entropy loss of Mean Squared Error.
    :param optimizer: the model optimizer, initialized with a learning rate.
    :param training_set: The training data, in a dataloader for easy iteration.
    :param test_loader: The testing data, in a dataloader for easy iteration.
    """
    num_epochs = 100 #10000000000000000 # obviously, this is too many. I don't know what this author was thinking.
    for epoch in range(num_epochs):
        # loop through each data point in the training set

        for data, targets in train_loader:

            # run the model on the data
            Nsample = 1
            model_input = torch.reshape(data, [Nsample, 784])#, 1])#  Turn the 28 by 28 image tensors into a 784 dimensional tensor.
            out = model(model_input)

            # Calculate the loss
            #targets = targets[:, None] # add an extra dimension to keep CrossEntropy happy.

            loss = loss_fn(out, targets)

            # Find the gradients of our loss via backpropogation
            loss.backward()

            # Adjust accordingly with the optimizer
            optimizer.step()

        # Give status reports every 100 epochs
        if epoch % 100==0:
            print(f" EPOCH {epoch}. Progress: {epoch/num_epochs*100}%. ")
            print(f" Train accuracy: {evaluate(model,train_loader)}. Test accuracy: {evaluate(model,test_loader)}")


def evaluate(model, evaluation_set):
    """
    Evaluates the given model on the given dataset.
    Returns the percentage of correct classifications out of total classifications.
    """
    correct = []
    with torch.no_grad(): # this disables backpropogation, which makes the model run much more quickly.
        for data, targets in evaluation_set:
            model_input = torch.reshape(data, [1, 784])
            result = model(model_input)
            softie = nn.Softmax(dim=1)
            result = softie(result)
            most_prob = np.argmax(result)
            if most_prob==targets:
                correct.append(1)
            else:
                correct.append(0)
    accuracy = np.sum(correct)/np.float(len(correct))
    return accuracy


# initialize the model (adapt this to each model)
model = LogisticRegression()
batch_size = 1
mnist_train = datasets.MNIST(root='data', train=True, download=True,
                             transform=transforms.ToTensor())
mnist_test  = datasets.MNIST(root='data', train=False, download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size)


# initialize the optimizer, and set the learning rate
# SGD === Stochastic Gradient Descent
SGD = torch.optim.SGD(model.parameters(), lr=0.5)
SGD.zero_grad()

loss_fn = nn.CrossEntropyLoss()

train(model, loss_fn, SGD, train_loader, test_loader)



# ----- Functions for Part 5 -----
def mmd(X,Y, kernel_fn):
    """
    Implementation of Maximum Mean Discrepancy.
    :param X: An n x 1 numpy vector containing the samples from distribution 1.
    :param Y: An n x 1 numpy vector containing the samples from distribution 2.
    :param kernel_fn: supply the kernel function to use.
    :return: the maximum mean discrepancy:
    MMD(X,Y) = Expected value of k(X,X) + Expected value of k(Y,Y) - Expected value of k(X,Y)
    where k is a kernel function
    """

    return mmd


def kernel(A, B):
    """
    A gaussian kernel on two arrays.
    :param A: An n x d numpy matrix containing the samples from distribution 1
    :param B: An n x d numpy matrix containing the samples from distribution 2.
    :return K:  An n x n numpy matrix k, in which k_{i,j} = e^{-||A_i - B_j||^2/(2*sigma^2)}
    """

    return K


