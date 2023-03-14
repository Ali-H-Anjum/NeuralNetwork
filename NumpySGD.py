import pandas as pd
import numpy as np
import pickle
from keras.datasets import mnist
import matplotlib.pyplot as plt

def ReLU(Z):
    return np.maximum(Z,0)

def der_ReLU(Z):
    return Z > 0

def Leaky_ReLU(Z):
    return Z * 0.01 if Z.any() < 0 else Z
    
def der_Leaky_ReLU(Z):
    return 0.01 if Z.any() < 0 else 1

def Swish(Z):
    return Z * 1 / (1 + np.exp(-Z))

def der_Swish(Z):
    return Z / (1. + np.exp(-Z)) + (1. / (1. + np.exp(-Z))) * (1. - Z * (1. / (1. + np.exp(-Z))))
    
def softmax(Z):
    exp = np.exp(Z - np.max(Z)) 
    return exp / exp.sum(axis=0)

def one_hot(Y):
    one_hot_Y = np.zeros((Y.max()+1,Y.size)) 
    one_hot_Y[Y,np.arange(Y.size)] = 1 
    return one_hot_Y

def init_params(size):
    W1 = np.random.rand(10,size) * np.sqrt(1./(784))
    b1 = 0
    W2 = np.random.rand(10,10) * np.sqrt(1./20)
    b2 = 0
    return W1,b1,W2,b2

def forward_propagation(X,W1,b1,W2,b2):
    Z1 = W1.dot(X) + b1 
    A1 = Leaky_ReLU(Z1) 
    Z2 = W2.dot(A1) + b2 
    A2 = softmax(Z2) 
    return Z1, A1, Z2, A2

def backward_propagation(X, Y, A1, A2, W2, Z1, m):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y #10,m
    dW2 = 1/m * (dZ2.dot(A1.T)) # 10 , 10
    db2 = 1/m * np.sum(dZ2,1) # 10, 1
    dZ1 = W2.T.dot(dZ2)*der_Leaky_ReLU(Z1) # 10, m
    dW1 = 1/m * (dZ1.dot(X.T)) #10, 784
    db1 = 1/m * np.sum(dZ1,1) # 10, 1
    return dW1, db1, dW2, db2

def update_params(alpha, W1, b1, W2, b2, dW1, db1, dW2, db2):
    W1 -= alpha * dW1 #10, 784
    b1 -= alpha * np.reshape(db1, (10,1)) #10, 1
    W2 -= alpha * dW2 #10,10
    b2 -= alpha * np.reshape(db2, (10,1)) #10, 1
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y)/Y.size

def make_predictions(X, W1 ,b1, W2, b2):    
    _, _, _, A2 = forward_propagation(X, W1, b1, W2, b2)
    predictions = get_predictions(A2)
    return predictions

def show_prediction(index,X, Y, W1, b1, W2, b2):
    vect_X = X[:, index,None]
    prediction = make_predictions(vect_X, W1, b1, W2, b2)
    label = Y[index]
    print("Prediction: ", prediction)
    print("Index: ", index, " Label: ", label)
    current_image = vect_X.reshape((width, height)) * scale
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

def compute_loss(y, y_pred):
    loss = 1 / 2 * np.mean((y_pred - y)**2)
    return loss

def plot_loss(x, y):
    plt.plot(x, y)
    plt.xlim(0, )
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.title('Loss')
    plt.savefig('Models/LossSGD')
    plt.show()

def gradient_descent(X, Y, alpha, iterations):
    size , m = X.shape
    W1, b1, W2, b2 = init_params(size)
    xArray = []
    yLoss = []
    for i in range(iterations):

        Z1, A1, Z2, A2 = forward_propagation(X, W1, b1, W2, b2)
        dW1, db1, dW2, db2 = backward_propagation(X, Y, A1, A2, W2, Z1, m)
        W1, b1, W2, b2 = update_params(alpha, W1, b1, W2, b2, dW1, db1, dW2, db2)
        xArray.append(i)

        if (i+1) % int(iterations/iterations) == 0:
            prediction = get_predictions(A2)
            if (i+1) % int(iterations/10) == 0:
                print(f"Iteration: {i+1} / {iterations}")
                print(f'{get_accuracy(prediction, Y):.3%}')
            loss = compute_loss(Y_train, prediction)
            yLoss.append(loss)
    plot_loss(xArray, yLoss)
    return W1, b1, W2, b2

############## MAIN ##############

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()                 
scale = 255 
width = X_train.shape[1]
height = X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], width * height).T / scale
X_test = X_test.reshape(X_test.shape[0], width * height).T  / scale

iterations = 200
alpha = 0.15

W1, b1, W2, b2 = gradient_descent(X_train, Y_train, alpha, iterations)
with open("trained_params.pkl","wb") as dump_file:
    pickle.dump((W1, b1, W2, b2),dump_file)

with open("trained_params.pkl","rb") as dump_file:
    W1, b1, W2, b2 = pickle.load(dump_file)

for x in range(1, 11):
    random_Index = np.random.randint(10000)
    show_prediction(random_Index, X_test, Y_test, W1, b1, W2, b2)

