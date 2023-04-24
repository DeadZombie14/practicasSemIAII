import numpy as np
import matplotlib.pyplot as plt

### Funciones de activacion
def linear(z, derivative=False):
    a = z
    if derivative:
        da = np.ones(z.shape)
        return a, da
    return a

def logistic(z, derivative=False):
    a = 1 / (1 + np.exp(-z))
    if(derivative):
        da = a * (1 - a) # vectorizado
        return a, da
    return a

def softmax(z, derivative=False):
    e = np.exp(z - np.exp(z, axis=0))
    a = e / np.sum(e, axis=0)
    if(derivative):
        da = np.ones(z.shape)
        return a, da
    return a

### De las capas ocultas
def tanh(z, derivative=False):
    a = np.tanh(z)
    if(derivative):
        da = (1 - a) * (1 - a) # vectorizado
        return a, da
    return a

def relu(z, derivative=False):
    a = z * (z >= 0) # elementor mayor entre 0 y mayor
    if(derivative):
        da = np.array(z >= 0, dtype= float)
        return a, da
    return a

def logistic_hidden(z, derivative=False):
    a = 1 / (1 + np.exp(-z))
    if(derivative):
        da = a * (1 - a) # vectorizado
        return a, da
    return a

### Fin

class DenseNetwork:
    def __init__(self, layers_dim, hidden_activation=tanh, output_activation=logistic):
        # Atributes
        self.L = len(layers_dim)-1
        self.w = [None] * (self.L+1) # Crear contenedor vacio
        self.b = [None] * (self.L+1)
        self.f = [None] * (self.L+1)

        # Initialize weights and biases
        for l in range(1, self.L + 1):
            self.w[l] = -1 + 2 * np.random.rand(layers_dim[l], layers_dim[l-1])
            self.b[l] = -1 + 2 * np.random.rand(layers_dim[l], 1)
            if l == self.L:
                self.f[l] = output_activation
            else:
                self.f[l] = hidden_activation
        pass
    
    def predict(self, X):
        a = X
        for l in range(1, self.L+1):
            z = self.w[l] @ a + self.b[l]
            a = self.f[l](z)
        return a
    
    def fit(self, X, Y, epochs=500, lr=0.1):
        p = X.shape[1]
        # SGD
        for _ in range(epochs):
            # Initiliaze activations and gradients
            a = [None] * (self.L + 1)
            da =  [None] * (self.L + 1)
            lg =  [None] * (self.L + 1)

            # Propagation
            a[0] = X
            for l in range(1, self.L+1):
                z = self.w[l] @ a[l-1] + self.b[l]
                a[l], da[l] = self.f[l](z, derivative=True)
            
            # Backpropagation
            for l in range(self.L, 0, -1):
                if l == self.L:
                    lg[l] = - (Y - a[l]) * da[l]
                else:
                    lg[l] = (self.w[l+1].T @ lg[l+1]) * da[l]
            
            # Gradient Descent
            for l in range(1, self.L+1):
                self.w[l] -= (lr/p) * (lg[l] @ a[l-1].T)
                self.b[l] -= (lr/p) * np.sum(lg[l]) # Añadir axis=0 si falla

# net = DenseNetwork((2,10,1))

def MLP_binary_classification_2d(X,Y,net):
    plt.figure()
    for i in range(X.shape[1]):
        if Y[0,i]==0:
            plt.plot(X[0,i], X[1,i], 'ro', markersize=9)
        else:
            plt.plot(X[0,i], X[1,i], 'bo',markersize=9)
    xmin, ymin=np.min(X[0,:])-0.5, np.min(X[1,:])-0.5
    xmax, ymax=np.max(X[0,:])+0.5, np.max(X[1,:])+0.5
    xx, yy = np.meshgrid(np.linspace(xmin,xmax, 100), 
                         np.linspace(ymin,ymax, 100))
    data = [xx.ravel(), yy.ravel()]
    zz = net.predict(data)
    zz = zz.reshape(xx.shape)
    plt.contour(xx,yy,zz,[0.5], colors='k',  linestyles='--', linewidths=2)
    plt.contourf(xx,yy,zz, alpha=0.8, 
                 cmap=plt.cm.RdBu)
    plt.xlim([xmin,xmax])
    plt.ylim([ymin,ymax])
    plt.grid()
    plt.show()
    

# X = np.array([[0, 0, 1, 1],
#               [0, 1, 0, 1]])
# Y = np.array([[1, 0, 0, 1]]) 

# Prueba con datos propios
X = np.loadtxt(fname= 'X.csv',delimiter=',').T
Y = np.loadtxt(fname= 'Y.csv',delimiter=',').T
Y = np.array([Y]) # Se necesita convertir a matriz

net = DenseNetwork((2,2,1))
net.fit(X, Y, epochs=10000)
print(net.predict(X))
MLP_binary_classification_2d(X,Y,net)
