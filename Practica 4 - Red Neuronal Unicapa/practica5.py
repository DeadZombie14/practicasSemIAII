# Mendoza Morelos Martin Mathier Practica 5
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class Ventana:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title('Practica 5 Red Neuronal Multicapa')
        self.window.resizable(False,False)
        self.window.protocol("WM_DELETE_WINDOW", self.cerrar)

        acciones = tk.Frame(self.window)
        acciones.pack(side='left', fill='both') 

        # Crear grafica
        self.grafica = tk.Frame(self.window, width=600, height=600, background="#000")
        self.grafica.pack(side='right')
        fig = plt.figure(figsize = (600/100, 600/100), dpi = 100)
        self.plot = fig.add_subplot()
        self.canvas = FigureCanvasTkAgg(figure= fig,master = self.grafica)
        self.canvas.get_tk_widget().pack()

        self.numeroNeuronas = tk.StringVar(value=str(2))
        numeroNeuronasFrame = tk.Frame(acciones)
        numeroNeuronasLabel = tk.Label(numeroNeuronasFrame, text="Numero de neuronas:")
        numeroNeuronasLabel.pack(side='top')
        numeroNeuronasInput = tk.Frame(numeroNeuronasFrame)
        numeroNeuronasInput.pack(fill='both')
        numeroNeuronasContador = tk.Frame(numeroNeuronasInput)
        numeroNeuronasContador.pack(side='bottom')
        numeroNeuronasEntry = tk.Entry(master=numeroNeuronasContador, textvariable=self.numeroNeuronas)
        numeroNeuronasEntry.pack(side='left')
        aumentarNeuronasBtn = tk.Button(
            master=numeroNeuronasContador,
            command=lambda: self.numeroNeuronas.set(str( int(self.numeroNeuronas.get()) + 1 )),
            height=1,width=1,text="+")
        aumentarNeuronasBtn.pack(side='left')
        reducirNeuronasBtn = tk.Button(
            master=numeroNeuronasContador,
            command=lambda: self.numeroNeuronas.set(str( int(self.numeroNeuronas.get()) - 1 )),
            height=1,width=1,text="-")
        reducirNeuronasBtn.pack(side='left')
        numeroNeuronasFrame.pack(padx=10, pady=10)

        self.epocas = tk.StringVar(value=str(500))
        epocasFrame = tk.Frame(acciones)
        epocasFrame.pack(padx=10, pady=10)
        epocasFrameLabel = tk.Label(master=epocasFrame, text="Epocas:")
        epocasFrameLabel.pack(side='top')
        epocasFrameEntry = tk.Entry(master=epocasFrame, textvariable=self.epocas)
        epocasFrameEntry.pack(side='bottom')

        creardatosBtn = tk.Button(
            master = acciones,
            command = self.generarDataset,
            height = 2,
            width = 20,
            text = "Generar dataset")
        creardatosBtn.pack(padx=10, pady=(0,10))

        procesarBtn = tk.Button(
            master = acciones,
            command = self.entrenarRedNeuronal,
            height = 2,
            width = 20,
            text = "Entrenar")
        procesarBtn.pack(padx=10, pady=(0,10))

        self.error = tk.StringVar(value=str(0))
        errorFrame = tk.Frame(acciones)
        errorLabel = tk.Label(master=errorFrame, text="Error actual:")
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

        self.window.mainloop()
    def leerDatos(self, archivo):
        datos = np.loadtxt(
            fname= archivo,
            delimiter=',')
        return datos
    def generarDataset(self):
        self.ventana2 = VentanaGenerarDataset(self.window)
    def cerrar(self):
        self.window.quit()
    def dibujarResultados(self, X, Y, Y_est, net, plot, n_neuronas):
        plot.clear()
        plot.set_title('Red Neuronal Unicapa')
        plot.grid('on')
        plot.set_xlim([-2,2])
        plot.set_ylim([-2,2])
        plot.set_xlabel(r'$x_1$')
        plot.set_ylabel(r'$x_2$')

        colores = [[0,0,0],
                   [1,0,0],
                   [0,1,0],
                   [1,1,0],
                   [0,0,1],
                   [1,0,1],
                   [0,1,1]]
        #y_c = np.sum(Y, axis=0, dtype=int)
        #y_c = np.argmax(Y, axis=0)
        
        # Crear combinaciones
        combinaciones = Y.T

        # Crear arreglo de unicos
        combUnicas = np.unique(combinaciones, axis=0)
        
        # Darle un ID a cada combinacion
        y_c = []
        for i in combinaciones:
            id = 0
            for c in range(combUnicas.shape[0]):
                if(i == combUnicas[c]).all():
                    id = c
            y_c.append(id)

        # Dibujar puntos
        for p in range(X.shape[1]):
            plot.plot(X[0,p], X[1,p], marker='o', c=colores[y_c[p]])

        # Dibujar lineas para cada neurona de la primera capa
        # capa = 2
        # for i in range(net.b[capa].shape[0]):
        #     # Encontrar que clase predice esta neurona (en base a su distancia con punto mas cercano)
        #     w1, w2 = net.w[capa][i][0],net.w[capa][i][1]  
        #     b = net.b[capa][i]
        #     plot.plot([-2,2], [(1/w2)*(-w1*(-2)-b), (1/w2)*(-w1*2-b)], linewidth=1, marker='.')

        # Dibujar contornos de resultados, mapeando puntos
        xmin, ymin=np.min(X[0,:])-2, np.min(X[1,:])-2
        xmax, ymax=np.max(X[0,:])+2, np.max(X[1,:])+2
        xx, yy = np.meshgrid(np.linspace(xmin,xmax, 100),np.linspace(ymin,ymax, 100))
        data = np.array([xx.ravel(), yy.ravel()])
        zz = net.predict(data)
        zz = zz.reshape(xx.shape)
        plot.contourf(xx,yy,zz, alpha=0.8, cmap=plt.cm.binary)
        
        self.canvas.draw()
    def entrenarRedNeuronal(self):
        X = self.leerDatos('X.csv').T
        Y = self.leerDatos('Y.csv').T
        n_entradas = X.shape[0] # 2 porque solo hay X1 y X2
        
        # Calcular numero de neuronas en base a Y
        #n_neuronas = int(self.numeroNeuronas.get())
        n_neuronas = 1
        self.numeroNeuronas.set(str(n_neuronas))

        epocas = int(self.epocas.get())
        net = DenseNetwork((n_entradas, 20, n_neuronas))
        def animar():
            from time import sleep
            Y_est = net.predict(X)
            self.dibujarResultados(X, Y, Y_est, net, self.plot, n_neuronas)
            self.error.set(str(net.error))
            self.epoca.set(str(net.epocaActual))
            sleep(0.0001)
            self.window.update()
        net.fit(X,Y, epocas, margen_error=0.01,callback=animar)
        Y_est = net.predict(X)
        self.dibujarResultados(X, Y, Y_est, net, self.plot, n_neuronas)
        print('Resultados Originales\n',Y)
        for neurona in range(n_neuronas):
            print('Resultados Predecidos Neurona #',(neurona+1),': ',Y_est[neurona])

### Funciones de activacion
def linear(z, derivada=False):
    a = z
    if derivada:
        da = np.ones(z.shape)
        return a, da
    return a

def logistic(z, derivada=False):
    a = 1 / (1 + np.exp(-z))
    if derivada:
        da = np.ones(z.shape)
        return a, da
    return a

def tanh(z, derivada=False):
    a = np.tanh(z)
    if derivada:
        da = (1 - a) * (1 + a)
        return a, da
    return a

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
    
    def fit(self, X, Y, epochs=500, lr=0.1, margen_error=0.1, callback=None):
        p = X.shape[1]
        # SGD
        self.epocaActual=0
        while (True):
            # Initiliaze activations and gradients
            a = [None] * (self.L + 1)
            da =  [None] * (self.L + 1)
            lg =  [None] * (self.L + 1)

            # Propagation
            a[0] = X
            for l in range(1, self.L+1):
                z = self.w[l] @ a[l-1] + self.b[l]
                a[l], da[l] = self.f[l](z, derivada=True)
            
            # Backpropagation
            for l in range(self.L, 0, -1):
                if l == self.L:
                    self.error = np.average(abs((Y - a[l])))
                    lg[l] = - (Y - a[l]) * da[l]
                else:
                    lg[l] = (self.w[l+1].T @ lg[l+1]) * da[l]
            
            # Gradient Descent
            for l in range(1, self.L+1):
                self.w[l] -= (lr/p) * (lg[l] @ a[l-1].T)
                self.b[l] -= (lr/p) * np.sum(lg[l]) # Añadir axis=0 si falla
            
            # Animation
            if(callback):
                callback()

            # Break condition
            self.epocaActual = self.epocaActual + 1
            if( self.error < margen_error or self.epocaActual > epochs ):
                break

class RedNeuronalUnicapa:
    def __init__(self, n_inputs, n_outputs, funcionActivacion=linear):
        self.w = -1 + 2 * np.random.rand(n_outputs, n_inputs)
        self.b = -1 + 2 * np.random.rand(n_outputs, 1)
        self.f = funcionActivacion

    def predict(self, X):
        Z = self.w @ X + self.b
        return self.f(Z)
    
    def fit(self, X, Y, epocas=500, factorAprendizaje=0.1, callback=None):
        p = X.shape[1]
        self.epocaActual=0
        while (True):
            # Propagar la red
            Z = self.w @ X + self.b
            Yest, dY = self.f(Z, derivada=True)
            
            # Calcular el gradiente
            self.error = abs(np.average((Y - Yest)))
            lg = (Y - Yest) * dY

            # Actualización de parámetros
            self.w += (factorAprendizaje/p) * lg @ X.T
            self.b += (factorAprendizaje/p) * np.sum(lg)

            if(callback):
                callback()
            self.epocaActual = self.epocaActual + 1
            if( self.error < 0.0001 or self.epocaActual > epocas ):
                break

class VentanaGenerarDataset:
    def __init__(self, ventanaPadre=None):
        self.X = []
        self.Y = []
        self.window = tk.Toplevel(ventanaPadre)
        self.window.title('Generar datos')
        self.window.resizable(False,False)
        self.window.protocol("WM_DELETE_WINDOW", self.cerrar)

        acciones = tk.Frame(self.window)
        acciones.pack(side='left', fill='both') 
        grafica = tk.Frame(self.window, width=600, height=600, background="#000")
        grafica.pack(side='right')

        guardarBtn = tk.Button(
            master = acciones,
            command = self.guardar,
            height = 2,
            width = 10,
            text = "Guardar")
        guardarBtn.pack()

        valorYFrame = tk.Frame(acciones)
        valorYLabel = tk.Label(master=valorYFrame, text="Valor Y:")
        valorYLabel.pack(side='top')
        valorYEntry = tk.Entry(master=valorYFrame, state='disabled')
        valorYEntry.pack(side='bottom')
        valorYFrame.pack(padx=10, pady=10)

        # Generar el primer dato al azar
        colorRandomRGB = tuple(np.random.choice(range(256), size=3))
        valor = np.asarray(colorRandomRGB)/255
        self.colorY = '#%02x%02x%02x' % colorRandomRGB
        self.valorY = valor

        seleccionarColorYBtn = tk.Button(
            master = acciones,
            command = self.generarColor,
            height = 2,
            width = 10,
            text = "Seleccionar")
        seleccionarColorYBtn.pack()
        
        # Dibujar la grafica
        self.fig = plt.figure(figsize = (5, 5), dpi = 100)
        # Conectar evento click
        self.fig.canvas.mpl_connect('button_press_event', self.generarDato)

        self.plot = self.fig.add_subplot()
        self.plot.set_title('Haz clic para colocar punto')
        self.plot.grid('on')
        self.plot.set_xlim([-2,2])
        self.plot.set_ylim([-2,2])
        self.plot.set_xlabel(r'$x_1$')
        self.plot.set_ylabel(r'$x_2$')
        
        canvas = FigureCanvasTkAgg(
            figure= self.fig,
            master = grafica)  
        canvas.draw()
        canvas.get_tk_widget().pack()
        canvas.get_tk_widget().pack()

        self.window.mainloop()

    def cerrar(self):
        self.window.destroy()
    
    def generarColor(self):
        from tkinter.colorchooser import askcolor
        colorRGB = askcolor(title="Selecciona un color para dar valor a Y:", parent=self.window)
        valor = np.asarray(colorRGB[0])/255
        self.colorY = colorRGB[1]
        self.valorY = valor

    def generarDato(self, event):
        valorDeseado = self.valorY
        self.plot.plot(event.xdata, event.ydata, marker='o', color=self.colorY)
        self.X.append([event.xdata, event.ydata])
        self.Y.append(valorDeseado)
        self.fig.canvas.draw()
    
    def guardar(self):
        self.crearDatos('X.csv', self.X)
        self.crearDatos('Y.csv', self.Y)
        self.window.destroy()

    def crearDatos(self, archivo, datos):
        datos = np.array(datos)
        np.savetxt(
            fname= archivo,
            X= datos,
            fmt='%.5f',
            delimiter=',')

app = Ventana()