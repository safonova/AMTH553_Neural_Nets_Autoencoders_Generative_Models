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
import pandas as pd

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
        self.Nsamples = 128
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
            forward_result, a tensor of shape 10
        """
        forward_result = torch.matmul(x, self.W)
        forward_result += self.bias
        forward_result = torch.reshape(forward_result, [self.Nsamples, 10])

        return forward_result

class FeedForwardNet(nn.Module):
    """ Simple feed forward network with one hidden layer."""
    # Here, you should place an exact copy of the code from the LogisticRegression class, with two modifications:
    # 1. Add another weight and bias vector to represent the hidden layer
    # 2. In the forward function, add some type of nonlinearity to the output of the first layer, then pass it onto the hidden layer.
    def __init__(self): # initialize the model
        super(FeedForwardNet, self).__init__() # call for the parent class to initialize
        self.Nfeatures = 784
        self.Noutcomes = 10
        self.Nsamples = 128
        weight1_shape = [self.Nfeatures, self.Nsamples]
        weight2_shape = [self.Nsamples, self.Noutcomes]

        bias1_shape = [self.Nsamples, self.Nsamples]
        bias2_shape = [self.Nsamples, self.Noutcomes]

        random_tensor_W1 = torch.rand(weight1_shape)*2-1
        random_tensor_b1 = torch.rand(bias1_shape)*2-1
        random_tensor_W2 = torch.rand(weight2_shape)*2-1
        random_tensor_b2 = torch.rand(bias2_shape)*2-1

        self.W = nn.Parameter(random_tensor_W1)
        self.bias1 = nn.Parameter(random_tensor_b1)
        self.W2 = nn.Parameter(random_tensor_W2)
        self.bias2 = nn.Parameter(random_tensor_b2)


    def forward(self, x):
        """
        this is the function that will be executed when we call the logistic regression on data.
        INPUT:
            x, an MNIST image represented as a tensor of shape 784
        OUTPUT:
            forward_result, a tensor of shape 10
        """

        forward_result = torch.matmul(x, self.W)
        forward_result += self.bias1
        #forward_result = torch.sigmoid(forward_result)
        activation_fn = nn.ELU()
        forward_result = activation_fn(forward_result)
        forward_result = torch.matmul(forward_result, self.W2)
        forward_result += self.bias2
        #forward_result = torch.sigmoid(forward_result)

        forward_result = torch.reshape(forward_result, [self.Nsamples, 10])#, self.Nsamples])#, 10])

        return forward_result


class FeedForwardNet2(nn.Module):
    """ Simple feed forward network with two hidden layers."""
    # Here, you should place an exact copy of the code from the LogisticRegression class, with two modifications:
    # 1. Add another weight and bias vector to represent the hidden layer
    # 2. In the forward function, add some type of nonlinearity to the output of the first layer, then pass it onto the hidden layer.
    def __init__(self): # initialize the model
        super(FeedForwardNet2, self).__init__() # call for the parent class to initialize
        self.Nfeatures = 784
        self.Noutcomes = 10
        self.Nsamples = 128
        weight1_shape = [self.Nfeatures, self.Nsamples]
        weight2_shape = [self.Nsamples, self.Nsamples]
        weight3_shape = [self.Nsamples, self.Noutcomes]

        bias1_shape = [self.Nsamples, self.Nsamples]
        bias2_shape = [self.Nsamples, self.Nsamples]
        bias3_shape = [self.Nsamples, self.Noutcomes]


        random_tensor_W1 = torch.rand(weight1_shape)*2-1
        random_tensor_b1 = torch.rand(bias1_shape)*2-1
        random_tensor_W2 = torch.rand(weight2_shape)*2-1
        random_tensor_b2 = torch.rand(bias2_shape)*2-1
        random_tensor_W3 = torch.rand(weight3_shape)*2-1
        random_tensor_b3 = torch.rand(bias3_shape)*2-1

        self.W = nn.Parameter(random_tensor_W1)
        self.bias1 = nn.Parameter(random_tensor_b1)
        self.W2 = nn.Parameter(random_tensor_W2)
        self.bias2 = nn.Parameter(random_tensor_b2)
        self.W3 = nn.Parameter(random_tensor_W3)
        self.bias3 = nn.Parameter(random_tensor_b3)


    def forward(self, x):
        """
        this is the function that will be executed when we call the logistic regression on data.
        INPUT:
            x, an MNIST image represented as a tensor of shape 784
        OUTPUT:
            forward_result, a tensor of shape 10
        """

        forward_result = torch.matmul(x, self.W)
        forward_result += self.bias1
        # forward_result = torch.sigmoid(forward_result)
        activation_fn = nn.ELU()
        forward_result = activation_fn(forward_result)
        forward_result = torch.matmul(forward_result, self.W2)
        forward_result += self.bias2
        forward_result = activation_fn(forward_result)
        forward_result = torch.matmul(forward_result, self.W3)
        forward_result += self.bias3
        forward_result = torch.reshape(forward_result, [self.Nsamples, 10])#, self.Nsamples])#, 10])

        return forward_result

class FeedForwardNet3(nn.Module):
    """ Simple feed forward network with two hidden layers."""
    # Here, you should place an exact copy of the code from the LogisticRegression class, with two modifications:
    # 1. Add another weight and bias vector to represent the hidden layer
    # 2. In the forward function, add some type of nonlinearity to the output of the first layer, then pass it onto the hidden layer.
    def __init__(self): # initialize the model
        super(FeedForwardNet3, self).__init__() # call for the parent class to initialize
        self.Nfeatures = 784
        self.Noutcomes = 10
        self.Nsamples = 128
        weight1_shape = [self.Nfeatures, self.Nsamples]
        weight2_shape = [self.Nsamples, self.Nsamples]
        weight3_shape = [self.Nsamples, self.Nsamples]
        weight4_shape = [self.Nsamples, self.Noutcomes]


        self.W1 = nn.Linear(*weight1_shape)
        self.W2 = nn.Linear(*weight2_shape)
        self.W3 = nn.Linear(*weight3_shape)
        self.W4 = nn.Linear(*weight4_shape)


    def forward(self, x):
        """
        this is the function that will be executed when we call the logistic regression on data.
        INPUT:
            x, an MNIST image represented as a tensor of shape 784
        OUTPUT:
            forward_result, a tensor of shape 10
        """
        activation_fn = nn.ELU()

        forward_result = self.W1(x)
        forward_result = activation_fn(forward_result)


        forward_result = self.W2(forward_result)
        forward_result = activation_fn(forward_result)

        forward_result = self.W3(forward_result)
        forward_result = activation_fn(forward_result)

        forward_result = self.W4(forward_result)

        forward_result = torch.reshape(forward_result, [self.Nsamples, 10])

        return forward_result

class FeedForwardNet4(nn.Module):
    """ Simple feed forward network with two hidden layers."""
    # Here, you should place an exact copy of the code from the LogisticRegression class, with two modifications:
    # 1. Add another weight and bias vector to represent the hidden layer
    # 2. In the forward function, add some type of nonlinearity to the output of the first layer, then pass it onto the hidden layer.
    def __init__(self): # initialize the model
        super(FeedForwardNet4, self).__init__() # call for the parent class to initialize
        self.Nfeatures = 784
        self.Noutcomes = 10
        self.Nsamples = 128
        weight1_shape = [self.Nfeatures, self.Nsamples]
        weight2_shape = [self.Nsamples, self.Nsamples*2]
        weight3_shape = [self.Nsamples*2, self.Nsamples*3]
        weight4_shape = [self.Nsamples*3, self.Nsamples*2]
        weight5_shape = [self.Nsamples*2, self.Noutcomes]


        self.W1 = nn.Linear(*weight1_shape)
        self.W2 = nn.Linear(*weight2_shape)
        self.W3 = nn.Linear(*weight3_shape)
        self.W4 = nn.Linear(*weight4_shape)
        self.W5 = nn.Linear(*weight5_shape)


    def forward(self, x):
        """
        this is the function that will be executed when we call the logistic regression on data.
        INPUT:
            x, an MNIST image represented as a tensor of shape 784
        OUTPUT:
            forward_result, a tensor of shape 10
        """
        activation_fn = nn.ELU()

        forward_result = self.W1(x)
        forward_result = activation_fn(forward_result)

        forward_result = self.W2(forward_result)
        forward_result = activation_fn(forward_result)

        forward_result = self.W3(forward_result)
        forward_result = activation_fn(forward_result)

        forward_result = self.W4(forward_result)
        forward_result = activation_fn(forward_result)

        forward_result = self.W5(forward_result)

        forward_result = torch.reshape(forward_result, [self.Nsamples, 10])

        return forward_result


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.lin1 = nn.Linear(784, 1000)
        # define additional layers here
        # 784-1000-500-250-2-250-500-1000-784
        self.lin2 = nn.Linear(1000, 500)
        self.lin3 = nn.Linear(500, 250)
        self.lin4 = nn.Linear(250, 2)
        self.lin5 = nn.Linear(2, 250)
        self.lin6 = nn.Linear(250, 500)
        self.lin7 = nn.Linear(500, 1000)
        self.lin8 = nn.Linear(1000, 784)

    def encode(self, x):
        x = self.lin1(x)
        # ... additional layers, plus possible nonlinearities.
        for layer in [self.lin2,
                      self.lin3,
                      self.lin4,
                      ]:
            x = torch.tanh(x)
            x = layer(x)
        return x

    def decode(self, z):
        # ditto, but in reverse
        z = self.lin5(z)
        z = torch.tanh(z)
        z = self.lin6(z)
        z = torch.tanh(z)
        z = self.lin7(z)
        z = torch.sigmoid(z)
        z = self.lin8(z)
        return z

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


class retinal_Bipolar_Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.lin1 = nn.Linear(784, 1000)
        # define additional layers here
        # 784-1000-500-250-2-250-500-1000-784
        self.lin2 = nn.Linear(1000, 500)
        self.lin3 = nn.Linear(500, 250)
        self.lin4 = nn.Linear(250, 2)
        self.lin5 = nn.Linear(2, 250)
        self.lin6 = nn.Linear(250, 500)
        self.lin7 = nn.Linear(500, 1000)
        self.lin8 = nn.Linear(1000, 784)

    def encode(self, x):
        x = self.lin1(x)
        # ... additional layers, plus possible nonlinearities.
        for layer in [self.lin2,
                      self.lin3,
                      self.lin4,
                      ]:
            x = torch.tanh(x)
            x = layer(x)
        return x

    def decode(self, z):
        # ditto, but in reverse
        z = self.lin5(z)
        z = torch.tanh(z)
        z = self.lin6(z)
        z = torch.tanh(z)
        z = self.lin7(z)
        # z = torch.sigmoid(z)
        z = self.lin8(z)
        return z

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
    num_epochs = 201
    print("number of epochs", num_epochs)
    for epoch in range(num_epochs):
        print("epoch", epoch)
        # loop through each data point in the training set
        for data, targets in train_loader:

            # run the model on the data
            Nsample = 128
            try:
                model_input = torch.reshape(data, [Nsample, 784]) #  Turn the 28 by 28 image tensors into a 784 dimensional tensor.
            except RuntimeError:
                break

            out = model(model_input)

            # Calculate the loss
            """if torch.isnan(sum(sum(out))) or torch.isinf(sum(sum(out))):
                print("sum(sum(model_input))", sum(sum(model_input)))
                print("model_input", model_input)
                print("sum(sum(out))", sum(sum(out)))
                print('invalid input detected at iteration ', epoch)
            """

            loss = loss_fn(out, targets)

            # Find the gradients of our loss via backpropogation
            loss.backward()

            # Adjust accordingly with the optimizer
            optimizer.step()
            optimizer.zero_grad()
        print("loss", loss.item())
        # Give status reports every 100 epochs
        if epoch % 5==0:
            evaluate_train = np.round(evaluate(model,train_loader, Nsample=Nsample), 4)
            # evaluate_test = np.round(evaluate(model,test_loader, Nsample=Nsample), 4)

            evaluate_test, correct_digit, mislabeled_digit = evaluate(model, test_loader, Nsample=Nsample, return_confusion=True)
            evaluate_test = np.round(evaluate_test, 4)
            dataseries = pd.Series({"epoch":int(epoch),
                                    "Train accuracy": evaluate_train,
                                    "Test accuracy": evaluate_test,
                                    "Loss" : np.round(loss.item(), 4)}
                                   )
            if epoch==0:
                df_accuracy = pd.DataFrame()

            df_accuracy = df_accuracy.append(dataseries, ignore_index=True)
            print(f" EPOCH {epoch}. Progress: {epoch/num_epochs*100}%. ")
            print(f" Train accuracy: {evaluate_train}. Test accuracy: {evaluate_test}")
            df_accuracy.to_csv("dataframe_accuracy.csv")

            confusionmatrix = {"Correct digit": correct_digit,
                                         "Mislabeled digit": mislabeled_digit}

            if epoch == 0:
                df_confusion = pd.DataFrame()

            df_confusion = df_confusion.append(confusionmatrix, ignore_index=True)
            df_confusion.to_csv("dataframe_confusion.csv")




def evaluate(model, evaluation_set, Nsample=128, return_confusion=False):
    """
    Evaluates the given model on the given dataset.
    Returns the percentage of correct classifications out of total classifications.
    """
    correct = []
    correct_digit = []
    mislabeled_digit = []
    with torch.no_grad(): # this disables backpropogation, which makes the model run much more quickly.
        for data, targets in evaluation_set:
            try:
                model_input = torch.reshape(data, [Nsample, 784])
            except RuntimeError:
                break
            result = model(model_input)
            softie = nn.Softmax(dim=1)
            result = softie(result)
            for ii, result_line in enumerate(result):
                most_prob = np.argmax(result_line)
                if most_prob==targets[ii]:
                    correct.append(1)
                else:
                    correct.append(0)
                    correct_digit.append(targets[ii].item())
                    mislabeled_digit.append(most_prob.item())
    accuracy = np.sum(correct)/np.float(len(correct))
    if return_confusion:
        return accuracy, correct_digit, mislabeled_digit
    return accuracy






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
    kxx = kernel_fn(X, X)
    kyy = kernel_fn(Y, Y)
    kxy = kernel_fn(X, Y)
    mmd = np.mean(kxx) + np.mean(kyy) + np.mean(kxy)
    return mmd


def kernel(A, B):
    """
    A gaussian kernel on two arrays.
    :param A: An n x d numpy matrix containing the samples from distribution 1
    :param B: An n x d numpy matrix containing the samples from distribution 2.
    :return K:  An n x n numpy matrix k, in which k_{i,j} = e^{-||A_i - B_j||^2/(2*sigma^2)}
    """
    sigma = np.std(A.flatten())
    K = np.exp(-(abs(A-B)**2)/(0.5*sigma**2))
    return K

def main():
    batch_size = 128
    mnist_train = datasets.MNIST(root='data', train=True, download=True,
                                 transform=transforms.ToTensor())
    mnist_test = datasets.MNIST(root='data', train=False, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size)

    #======2.2 Logistic Regression======
    '''# initialize the model (adapt this to each model)
    model = LogisticRegression()
    # initialize the optimizer, and set the learning rate
    # SGD === Stochastic Gradient Descent
    SGD = torch.optim.SGD(model.parameters(), lr=0.5)
    SGD.zero_grad()
    loss_fn = nn.CrossEntropyLoss()
    train(model, loss_fn, SGD, train_loader, test_loader)'''

    #======2.3 Feed-Forward Neural Network======
    '''
    model = FeedForwardNet()

    SGD = torch.optim.SGD(model.parameters(), lr=0.005)
    SGD.zero_grad()
    loss_fn = nn.CrossEntropyLoss()

    train(model, loss_fn, SGD, train_loader, test_loader)'''
    #======2.3 varying width======
    '''model = FeedForwardNet2()

    SGD = torch.optim.SGD(model.parameters(), lr=0.005)
    SGD.zero_grad()
    loss_fn = nn.CrossEntropyLoss()

    train(model, loss_fn, SGD, train_loader, test_loader)'''
    '''model = FeedForwardNet3()

    SGD = torch.optim.SGD(model.parameters(), lr=0.001)
    SGD.zero_grad()
    loss_fn = nn.CrossEntropyLoss()

    train(model, loss_fn, SGD, train_loader, test_loader)'''
    model = FeedForwardNet4()

    SGD = torch.optim.SGD(model.parameters(), lr=0.001)
    SGD.zero_grad()
    loss_fn = nn.CrossEntropyLoss()

    train(model, loss_fn, SGD, train_loader, test_loader)

    # ======3.1 Autoencoder======
    '''model = Autoencoder()

    #
    Adam = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    train(model, loss_fn, Adam, train_loader, test_loader)'''


if __name__=="__main__":
    main()
