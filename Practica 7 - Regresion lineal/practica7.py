# Mendoza Morelos Martin Mathier Practica 5
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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
        da = np.ones(z.shape)
        return a, da
    return a

def softmax(z, derivative=False):
    e = np.exp(z - np.exp(z))
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

class Ventana:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title('Practica 7 Regresion Lineal')
        self.window.resizable(False,False)
        self.window.protocol("WM_DELETE_WINDOW", self.cerrar)

        apartadoPrincipal = tk.Frame(self.window)
        apartadoPrincipal.pack(side='left') 

        acciones = tk.Frame(apartadoPrincipal)
        acciones.pack(side='left', fill='both') 

        acciones2 = tk.Frame(self.window)
        acciones2.pack(side='right', fill='both') 

        # Crear grafica
        self.grafica = tk.Frame(apartadoPrincipal, width=600, height=600, background="#000")
        self.grafica.pack(side='right')
        fig = plt.figure(figsize = (600/100, 600/100), dpi = 100)
        fig.suptitle("Regresion Lineal")
        self.plot = fig.add_subplot()
        self.canvas = FigureCanvasTkAgg(figure= fig,master = self.grafica)
        self.canvas.get_tk_widget().pack()

        # Crear grafica de error cuadratico medio
        self.graficaError = tk.Frame(acciones, width=300, height=300, background="#000")
        self.graficaError.pack(side='bottom')
        figError = plt.figure(figsize = (300/50, 300/50), dpi = 50)
        figError.suptitle("MSE")
        self.plotError = figError.add_subplot()
        self.plotError.clear()
        self.plotError.grid('on')
        self.plotError.set_xlim([0,1000])
        self.plotError.set_ylim([0,1])
        self.plotError.set_xlabel(r'$Iteracion$')
        self.plotError.set_ylabel(r'$Error$')
        self.canvasError = FigureCanvasTkAgg(figure= figError,master = self.graficaError)
        self.canvasError.get_tk_widget().pack()

        # Crear grafica de representacion de red neuronal
        self.graficaNetGraph = tk.Frame(acciones2, width=300, height=300, background="#000")
        self.graficaNetGraph.pack(side='bottom')
        figNetGraph = plt.figure(figsize = (300/50, 300/50), dpi = 50)
        figNetGraph.suptitle("Modelo de la red")
        self.plotNetGraph = figNetGraph.add_subplot()
        self.plotNetGraph.clear()
        self.plotNetGraph.axis('scaled')
        self.plotNetGraph.axis('off')
        self.canvasNetGraph = FigureCanvasTkAgg(figure= figNetGraph,master = self.graficaNetGraph)
        self.canvasNetGraph.get_tk_widget().pack()

        # Modificador de neuronas por capa
        cargarFrame = tk.Frame(acciones2)
        cargarFrame.pack(side='top')
        self.archivo = tk.StringVar(value=str('datos.csv'))
        datosArchivoEntryLabel = tk.Label(cargarFrame, text="Archivo: ")
        datosArchivoEntryLabel.pack(side='left')
        datosArchivoEntry = tk.Entry(master=cargarFrame, textvariable=self.archivo)
        datosArchivoEntry.pack(side='left')
        cargarBtn = tk.Button(
            master = acciones2,
            command = self.cargarDatos,
            height = 2,
            width = 20,
            text = "Cargar archivo")
        cargarBtn.pack(padx=10, pady=10, side='top')
        
        agregarCapaBtn = tk.Button(
            master = acciones2,
            command = self.agregarCapaOculta,
            height = 2,
            width = 20,
            text = "Agregar capa oculta")
        agregarCapaBtn.pack(padx=10, pady=(0,10))

        self.entradas = tk.StringVar(value=str(2))
        entradasFrame = tk.Frame(acciones2)
        entradasFrame.pack(side='top')
        entradasLabel = tk.Label(entradasFrame, text="Entradas detectadas:")
        entradasLabel.pack(side='top')
        entradasEntry = tk.Entry(master=entradasFrame, textvariable=self.entradas, state='disabled')
        entradasEntry.pack(side='bottom')

        # Capas ocultas
        self.capasOcultasFrame = tk.Frame(acciones2)
        self.capasOcultasFrame.pack(side='top')

        self.salidas = tk.StringVar(value=str(21))
        salidasFrame = tk.Frame(acciones2)
        salidasFrame.pack(side='top')
        salidasLabel = tk.Label(salidasFrame, text="Salidas detectadas:")
        salidasLabel.pack(side='top')
        salidasEntry = tk.Entry(master=salidasFrame, textvariable=self.salidas, state='disabled')
        salidasEntry.pack(side='bottom')
        
        from tkinter import ttk
        frameFunciones = tk.Frame(acciones)
        frameFunciones.pack()
        self.funcionOculta = tk.StringVar(value='logistic')
        funcionOcultaLabel = tk.Label(frameFunciones, text="Funcion capas ocultas")
        funcionOcultaLabel.pack(padx=10, pady=(0,10))
        funcionOcultaList = ttk.Combobox(frameFunciones, textvariable=self.funcionOculta)
        funcionOcultaList['values'] = ('linear','tanh','relu','logistic')
        funcionOcultaList['state'] = 'readonly'
        funcionOcultaList.pack(padx=10, pady=(0,10))
        self.funcionSalida = tk.StringVar(value='linear')
        funcionSalidaLabel = tk.Label(frameFunciones, text="Funcion capa salida")
        funcionSalidaLabel.pack(padx=10, pady=(0,10))
        funcionOcultaList = ttk.Combobox(frameFunciones, textvariable=self.funcionOculta)
        funcionSalidaList = ttk.Combobox(frameFunciones, textvariable=self.funcionSalida)
        funcionSalidaList['values'] = ('linear','logistic','softmax')
        funcionSalidaList['state'] = 'readonly'
        funcionSalidaList.pack(padx=10, pady=(0,10))

        self.epocas = tk.StringVar(value=str(50000))
        epocasFrame = tk.Frame(acciones)
        epocasFrame.pack(padx=10, pady=10)
        epocasFrameLabel = tk.Label(master=epocasFrame, text="Epocas maximas:")
        epocasFrameLabel.pack(side='top')
        epocasFrameEntry = tk.Entry(master=epocasFrame, textvariable=self.epocas)
        epocasFrameEntry.pack(side='bottom')
        
        self.errorMax = tk.StringVar(value=str(0.001))
        errorMaxFrame = tk.Frame(acciones)
        errorMaxFrame.pack(padx=10, pady=10)
        errorMaxFrameLabel = tk.Label(master=errorMaxFrame, text="MSE maximo:")
        errorMaxFrameLabel.pack(side='top')
        errorMaxFrameEntry = tk.Entry(master=errorMaxFrame, textvariable=self.errorMax)
        errorMaxFrameEntry.pack(side='bottom')

        procesarBtn = tk.Button(
            master = acciones,
            command = self.entrenarRedNeuronal,
            height = 2,
            width = 20,
            text = "Entrenar")
        procesarBtn.pack(padx=10, pady=10)

        self.error = tk.StringVar(value=str(0))
        errorFrame = tk.Frame(acciones)
        errorLabel = tk.Label(master=errorFrame, text="MSE actual:")
        errorLabel.pack(side='top')
        errorEntry = tk.Entry(master=errorFrame, state='disabled', textvariable=self.error)
        errorEntry.pack(side='bottom')
        errorFrame.pack(padx=10, pady=10)

        self.epoca = tk.StringVar(value=str(0))
        epocaFrame = tk.Frame(acciones)
        epocaLabel = tk.Label(master=epocaFrame, text="Epoca actual:")
        epocaLabel.pack(side='top')
        epocaEntry = tk.Entry(master=epocaFrame, state='disabled', textvariable=self.epoca)
        epocaEntry.pack(side='bottom')
        epocaFrame.pack(padx=10, pady=10)

        self.capasOcultas = []

        self.animacion = True
        self.animacionBtn = tk.Button(
            master = acciones,
            command = self.quitarAnimacion,
            height = 2,
            width = 20,
            text = "Deshabilitar animacion")
        self.animacionBtn.pack(padx=10, pady=(0,10))

        self.window.mainloop()
    def quitarAnimacion(self):
        if(self.animacion):
            self.animacion = False
            self.animacionBtn.config(text="Habilitar animacion")
        else:
            self.animacion = True
            self.animacionBtn.config(text="Deshabilitar animacion")
    def cerrar(self):
        self.window.destroy()
        self.window.update()
    def dibujarError(self, errorActual, iteracionActual):
        self.plotError.plot(iteracionActual,errorActual, marker=',', color='#000')
        self.canvasError.draw()

    def dibujarResultados(self, X, Y, Y_est, net, plot, n_neuronas):
        plot.clear()
        plot.grid('on')
        plot.set_xlim([np.min(X)-1,np.max(Y)+1])
        plot.set_ylim([np.min(Y)-1,np.max(Y)+1])
        plot.set_xlabel(r'$X$')
        plot.set_ylabel(r'$Y$')  

        # Dibujar puntos
        plot.scatter(X, Y, marker='o', color='blue')

        # Dibujar linea
        lineaX = np.linspace(np.min(X), np.max(X), 100).reshape(1, -1)
        lineaY = net.predict(lineaX)
        plot.plot(lineaX.ravel(), lineaY.ravel(), color='red')
        
        self.canvas.draw()

    def agregarCapaOculta(self, n_neuronas = 1):
        capaOculta = tk.StringVar(value=str(n_neuronas))
        capaOcultaId = len(self.capas)-1
        capaOcultaFrame = tk.Frame(self.capasOcultasFrame)
        capaOcultaFrame.pack(padx=10, pady=10)
        capaOcultaLabel = tk.Label(capaOcultaFrame, text="Capa oculta #1:")
        capaOcultaLabel.pack(side='top')
        capaOcultaInput = tk.Frame(capaOcultaFrame)
        capaOcultaInput.pack(fill='both')
        capaOcultaContador = tk.Frame(capaOcultaInput)
        capaOcultaContador.pack(side='bottom')
        capaOcultaEntry = tk.Entry(master=capaOcultaContador, textvariable=capaOculta, state='disabled')
        capaOcultaEntry.pack(side='left')
        aumentarNeuronasBtn = tk.Button(
            master=capaOcultaContador,
            command=lambda: self.modificarCapaOculta(capaOcultaId, self.capas[capaOcultaId] + 1, capaOculta),
            height=1,width=1,text="+")
        aumentarNeuronasBtn.pack(side='left')
        reducirNeuronasBtn = tk.Button(
            master=capaOcultaContador,
            command=lambda: self.modificarCapaOculta(capaOcultaId, self.capas[capaOcultaId] - 1, capaOculta),
            height=1,width=1,text="-")
        reducirNeuronasBtn.pack(side='left')
        quitarCapaBtn = tk.Button(
            master=capaOcultaContador,
            command=lambda: self.eliminarCapaOculta(capaOcultaId, capaOcultaFrame),
            height=1,text="Eliminar")
        quitarCapaBtn.pack(side='right', padx=10)
        self.capas.insert(len(self.capas)-1, n_neuronas)
        self.dibujarRedNeuronal()
        
    def eliminarCapaOculta(self, i, frame):
        self.capas.pop(i)
        frame.pack_forget()
        frame.destroy()
        self.dibujarRedNeuronal()
    
    def modificarCapaOculta(self, i, n_neuronas, variable):
        self.capas[i] = n_neuronas
        variable.set(str( n_neuronas ))
        self.dibujarRedNeuronal()

    def dibujarRedNeuronal(self):
        # Redibujar la red
        self.plotNetGraph.clear()
        dibujarRed = DibujarRedNeuronal()
        for capa in self.capas:
            dibujarRed.add_layer(capa)
        dibujarRed.draw(plot=self.plotNetGraph)
        self.canvasNetGraph.draw()
    
    def cargarDatos(self):
        x, y = np.loadtxt(fname=self.archivo.get(), delimiter=',', unpack=True, skiprows=1)
        self.X = np.atleast_2d(x)
        self.Y = np.atleast_2d(y)
        self.n_entradas = self.X.shape[0]
        self.n_salidas = self.Y.shape[0]
        self.entradas.set(str(self.n_entradas))
        self.salidas.set(str(self.n_salidas))
        self.capas = [self.n_entradas, self.n_salidas]
        self.dibujarRedNeuronal()

    def entrenarRedNeuronal(self):
        X = self.X 
        Y = self.Y
        n_entradas = self.n_entradas
        n_salidas = self.n_salidas

        capa_oculta = tanh
        capa_salida = linear
        if(self.funcionOculta.get() == 'linear'):
            capa_oculta = linear
        elif(self.funcionOculta.get() == 'tanh'):
            capa_oculta = tanh
        elif(self.funcionOculta.get() == 'relu'):
            capa_oculta = relu
        elif(self.funcionOculta.get() == 'logistic'):
            capa_oculta = logistic_hidden
        if(self.funcionSalida.get() == 'linear'):
            capa_salida = linear
        elif(self.funcionSalida.get() == 'logistic'):
            capa_salida = logistic
        elif(self.funcionSalida.get() == 'softmax'):
            capa_salida = softmax

        net = RedNeuronalMulticapa(tuple(self.capas),hidden_activation=capa_oculta, output_activation=capa_salida)

        epocas = int(self.epocas.get())
        self.plotError.clear()
        def animar():
            from time import sleep
            Y_est = net.predict(X)
            self.dibujarResultados(X, Y, Y_est, net, self.plot, n_salidas)
            self.error.set(str(net.error))
            self.epoca.set(str(net.epocaActual))
            self.dibujarError(net.error, net.epocaActual)
            #sleep(0.01)
            self.window.update()
        if(self.animacion):
            net.fit(X,Y, epocas, margen_error=float(self.errorMax.get()),callback=animar)
        else:
            net.fit(X,Y, epocas, margen_error=float(self.errorMax.get()),callback=None)
        Y_est = net.predict(X)
        self.dibujarResultados(X, Y, Y_est, net, self.plot, n_salidas)
        print('Resultados Originales\n',Y)
        for neurona in range(n_salidas):
            print('Resultados Predecidos Neurona #',(neurona+1),': ',Y_est[neurona])

class RedNeuronalMulticapa:
    def __init__(self, layers_dim, hidden_activation=tanh, output_activation=logistic):
        # Atributes
        self.L = len(layers_dim)-1
        # Initialize matrices
        self.w = [None] * (self.L+1)
        self.b = [None] * (self.L+1)
        self.f = [None] * (self.L+1)
        # Initialize weights and biases
        for l in range(1, self.L + 1):
            self.w[l] = np.random.rand(layers_dim[l], layers_dim[l-1])
            self.b[l] = np.random.rand(layers_dim[l], 1)
            if l == self.L:
                self.f[l] = output_activation
            else:
                self.f[l] = hidden_activation

    def predict(self, X):
        a = X
        for l in range(1, self.L+1):
            z = self.w[l] @ a + self.b[l]
            a = self.f[l](z)
        return a
    
    def fit(self, X, Y, epochs=500, eta=0.01, margen_error=0.1, callback=None):
        p=X.shape[1]
        self.epocaActual=0
        while (True):
            # Initiliaze activations and gradients
            a = [None] * (self.L + 1)
            da =  [None] * (self.L + 1)
            lg =  [None] * (self.L + 1)

            # Propagation
            a[0] = X # Layer 0 are inputs
            for l in range(1, self.L+1):
                z = self.w[l] @ a[l-1] + self.b[l]
                a[l], da[l] = self.f[l](z, True)
            
            # Backpropagation, calculate local gradients
            for l in range(self.L, 0, -1):
                if l == self.L:
                    # Output layer
                    self.error = np.average((Y - a[l])**2)
                    lg[l] = - (Y - a[l]) * da[l]
                else:
                    # Hidden layer
                    lg[l] = (self.w[l+1].T @ lg[l+1]) * da[l]
            
            # Ajust weights with Gradient Descent
            for l in range(1, self.L+1):
                self.w[l] = self.w[l] - (eta/p) * (lg[l] @ a[l-1].T)
                self.b[l] = self.b[l] - (eta/p) * np.sum(lg[l], axis=1, keepdims=True) #, axis=1, keepdims=True
            
            # Animation
            if(callback and self.epocaActual % 60 == 0):
                callback()

            # Break condition
            self.epocaActual = self.epocaActual + 1
            if( self.error < margen_error or self.epocaActual > epochs ):
                break

from math import cos, sin, atan

### Para dibujar la red
class Neuron():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self, plot):
        from matplotlib import patches as patch
        circle = patch.Circle((self.x, self.y), radius=10, fill=True, zorder=3, color="#F00")
        plot.add_patch(circle)


class Layer():
    def __init__(self, network, number_of_neurons):
        self.distance_between_neurons = 30
        self.distance_between_layers = 200
        self.number_of_neurons_in_widest_layer = 20
        self.previous_layer = self.__get_previous_layer(network)
        self.x = self.__calculate_layer_x_position()
        self.neurons = self.__intialise_neurons(number_of_neurons)

    def __intialise_neurons(self, number_of_neurons):
        neurons = []
        y = self.__calculate_margin_so_layer_is_centered(number_of_neurons)
        for iteration in range(number_of_neurons):
            neuron = Neuron(self.x, y)
            neurons.append(neuron)
            y += self.distance_between_neurons
        return neurons

    def __calculate_margin_so_layer_is_centered(self, number_of_neurons):
        return self.distance_between_neurons * (self.number_of_neurons_in_widest_layer - number_of_neurons) / 2

    def __calculate_layer_x_position(self):
        if self.previous_layer:
            return self.previous_layer.x + self.distance_between_layers
        else:
            return 0

    def __get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def __line_between_two_neurons(self, neuron1, neuron2, plot):
        x_vals = np.linspace(neuron1.x, neuron2.x, 100)
        y_vals = np.linspace(neuron1.y, neuron2.y, 100)
        plot.plot(x_vals, y_vals, color="#000", marker=',', zorder=1)

    def draw(self, plot, layerType=0):
        for neuron in self.neurons:
            neuron.draw(plot)
            if self.previous_layer:
                for previous_layer_neuron in self.previous_layer.neurons:
                    self.__line_between_two_neurons(neuron, previous_layer_neuron, plot)
        # write Text
        distance = self.number_of_neurons_in_widest_layer * self.distance_between_neurons + 10
        if layerType == 0:
            plot.text(self.x - 60, distance, 'Entradas', fontsize = 12)
        elif layerType == -1:
            plot.text(self.x - 60, distance, 'Salidas', fontsize = 12)
        else:
            plot.text(self.x - 60, distance, 'Oculta #'+str(layerType), fontsize = 12)


class DibujarRedNeuronal():
    def __init__(self):
        self.layers = []
        self.layersInNumber = []

    def add_layer(self, number_of_neurons):
        layer = Layer(self, number_of_neurons)
        self.layers.append(layer)
        self.layersInNumber.append(number_of_neurons)

    def draw(self, plot):
        widest_layer = max( self.layersInNumber )
        for layer in range(0,len(self.layers)):
            self.layers[layer].number_of_neurons_in_widest_layer = widest_layer
            if(layer == len(self.layers)-1):
                self.layers[layer].draw(plot, layerType=-1)
            else:
                self.layers[layer].draw(plot, layer)
        plot.axis('scaled')

app = Ventana()