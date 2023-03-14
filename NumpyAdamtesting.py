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

def initialize_adam(size, m):
    V_dw1 = 0
    S_dw1 = 0
    V_db1 = 0
    S_db1 = 0
    V_dw2 = 0
    S_dw2 = 0
    V_db2 = 0
    S_db2 = 0

    W1 = np.random.rand(10,size) * np.sqrt(1./(784))
    b1 = 0
    W2 = np.random.rand(10,10) * np.sqrt(1./20)
    b2 = 0

    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    alpha = 0.0000001
    return V_dw1, S_dw1, V_db1, S_db1, V_dw2, S_dw2, V_db2, S_db2, W1, b1, W2, b2, beta1, beta2, epsilon, alpha

def forward_propagation(X,W1,b1,W2,b2):
    Z1 = W1.dot(X) + b1 #10, m
    A1 = Swish(Z1) # 10,m
    Z2 = W2.dot(A1) + b2 #10,m
    A2 = softmax(Z2) #10,m
    return Z1, A1, Z2, A2

def backward_propagation(X, Y, A1, A2, W2, Z1, m):
    one_hot_Y = one_hot(Y)
    dZ2 = 2*(A2 - one_hot_Y) #10,m
    dW2 = 1/m * (dZ2.dot(A1.T)) # 10 , 10
    db2 = 1/m * np.sum(dZ2,1) # 10, 1
    dZ1 = W2.T.dot(dZ2) * der_Swish(Z1) # 10, m
    dW1 = 1/m * (dZ1.dot(X.T)) #10, 784
    db1 = 1/m * np.sum(dZ1,1) # 10, 1
    return dW1, db1, dW2, db2

def Momentum(X, Y, beta1):
    Y = beta1 * Y + (1 - beta1) * X
    return Y

def RMSprop(X, Y, beta2):
    Y = beta2 * Y + (1 - beta2) * np.square(X)
    return Y

def Correction(Y, beta, iteration):
    Y = Y / (1 - (beta ** iteration))
    return Y

def Compute(V_dw1, S_dw1, V_db1, S_db1, V_dw2, S_dw2, V_db2, S_db2, dW1, db1, dW2, db2, beta1, beta2, iteration):
    db1 = np.reshape(db1, (10,1))
    db2 = np.reshape(db2, (10,1))

    V_dw1 = Momentum(dW1, V_dw1, beta1)
    V_db1 = Momentum(db1, V_db1, beta1)
    V_dw2 = Momentum(dW2, V_dw2, beta1)
    V_db2 = Momentum(db2, V_db2, beta1)

    S_dw1 = RMSprop(dW1, S_dw1, beta2)
    S_db1 = RMSprop(db1, S_db1, beta2)
    S_dw2 = RMSprop(dW2, S_dw2, beta2)
    S_db2 = RMSprop(db2, S_db2, beta2)

    V_dw1 = Correction(V_dw1, beta1, iteration)
    V_db1 = Correction(V_db1, beta1, iteration)
    V_dw2 = Correction(V_dw2, beta1, iteration)
    V_db2 = Correction(V_db2, beta1, iteration)
    S_dw1 = Correction(S_dw1, beta2, iteration)
    S_db1 = Correction(S_db1, beta2, iteration)
    S_dw2 = Correction(S_dw2, beta2, iteration)
    S_db2 = Correction(S_db2, beta2, iteration)

    return V_dw1, S_dw1, V_db1, S_db1, V_dw2, S_dw2, V_db2, S_db2

def Update(Y, momentum, rms, alpha, epsilon):
    Y -= alpha * (momentum / (np.sqrt(rms) + epsilon))
    return Y

def UpdateParameters(W1, b1, W2, b2, V_dw1, S_dw1, V_db1, S_db1, V_dw2, S_dw2, V_db2, S_db2, alpha, epsilon):
    W1 = Update(W1, V_dw1, S_dw1, alpha, epsilon)
    W2 = Update(W2, V_dw2, S_dw2, alpha, epsilon) 
    b1 = Update(b1, V_db1, S_db1, alpha, epsilon)
    b2 = Update(b2, V_db2, S_db2, alpha, epsilon)
    
    print(W1.shape, b1.shape, W2.shape, b2.shape)
    return W1, W2, b1, b2

def Adam(X, Y, iterations):
    size , m = X.shape
    V_dw1, S_dw1, V_db1, S_db1, V_dw2, S_dw2, V_db2, S_db2, W1, b1, W2, b2, beta1, beta2, epsilon, alpha = initialize_adam(size, m)
    xArray = []
    yLoss = []

    for i in range(1, iterations):
        Z1, A1, Z2, A2 = forward_propagation(X,W1,b1,W2,b2)
        dW1, db1, dW2, db2 = backward_propagation(X, Y, A1, A2, W2, Z1, m)
        V_dw1, S_dw1, V_db1, S_db1, V_dw2, S_dw2, V_db2, S_db2 = Compute(V_dw1, S_dw1, V_db1, S_db1, V_dw2, S_dw2, V_db2, S_db2, dW1, db1, dW2, db2, beta1, beta2, i)
        W1, W2, b1, b2 = UpdateParameters(W1, b1, W2, b2, V_dw1, S_dw1, V_db1, S_db1, V_dw2, S_dw2, V_db2, S_db2, alpha, epsilon)

        
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
    plt.savefig('Models/LossAdam')
    plt.show()


############## MAIN ##############

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
scale = 255 
width = X_train.shape[1]
height = X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], width * height).T / scale
X_test = X_test.reshape(X_test.shape[0], width * height).T  / scale
iterations = 100

W1, b1, W2, b2 = Adam(X_train, Y_train, iterations)

"""""
for x in range(1, 11):
    random_Index = np.random.randint(10000)
    show_prediction(random_Index, X_test, Y_test, W1, b1, W2, b2)
    """