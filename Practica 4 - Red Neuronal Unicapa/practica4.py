import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class Ventana:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title('Practica 4 Red Neuronal Unicapa')
        self.window.resizable(False,False)
        self.window.protocol("WM_DELETE_WINDOW", self.cerrar)

        acciones = tk.Frame(self.window)
        acciones.pack(side='left', fill='both') 

        self.grafica = tk.Frame(self.window, width=600, height=600, background="#000")
        self.grafica.pack(side='right')

        self.numeroNeuronas = tk.StringVar(value=str(1))
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

        creardatosBtn = tk.Button(
            master = acciones,
            command = self.generarDataset,
            height = 2,
            width = 20,
            text = "Generar dataset")
        creardatosBtn.pack(padx=10, pady=(0,10))

        procesarBtn = tk.Button(
            master = acciones,
            command = self.entrenarPerceptron,
            height = 2,
            width = 20,
            text = "Comenzar entrenamiento")
        procesarBtn.pack(padx=10, pady=(0,10))

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
    def dibujarResultados(self, X, Y, Y_est, net):
        fig = plt.figure(figsize = (600/100, 600/100), dpi = 100)
        plot = fig.add_subplot()
        canvas = FigureCanvasTkAgg(
            figure= fig,
            master = self.grafica)  
        canvas.draw()
        canvas.get_tk_widget().pack()

        plot.clear()
        plot.set_title('Red Neuronal Unicapa')
        plot.grid('on')
        plot.set_xlim([-2,2])
        plot.set_ylim([-2,2])
        plot.set_xlabel(r'$x_1$')
        plot.set_ylabel(r'$x_2$')
        # Dibujar puntos
        for i in range(X.shape[1]):
            plot.plot(X[0,i], X[1,i], 'or')
        # Dibujar contornos de resultados
        xmin, ymin=np.min(X[0,:])-0.5, np.min(X[1,:])-0.5
        xmax, ymax=np.max(X[0,:])+0.5, np.max(X[1,:])+0.5
        xx, yy = np.meshgrid(np.linspace(xmin,xmax, 100),np.linspace(ymin,ymax, 100))
        data = [xx.ravel(), yy.ravel()]
        zz = net.predict(data)
        zz = zz.reshape(xx.shape)
        plt.contour(xx,yy,zz,[0.5], colors='k',  linestyles='--', linewidths=2)
        plt.contourf(xx,yy,zz, alpha=0.8, cmap=plt.cm.RdBu)
    def entrenarPerceptron(self):
        X = self.leerDatos('X.csv').T
        Y = self.leerDatos('Y.csv')
        n_entradas = X.shape[0] # 2 porque solo hay X1 y X2
        n_neuronas = int(self.numeroNeuronas.get())
        net = RedNeuronalUnicapa(n_entradas, n_neuronas, logistic)
        net.fit(X,Y)
        Y_est = net.predict(X)
        print('Resultados Originales\n',Y)
        print('Resultados Predecidos\n',Y_est)
        self.dibujarResultados(X, Y, Y_est, net)

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

class RedNeuronalUnicapa:
    def __init__(self, n_inputs, n_outputs, funcionActivacion=linear):
        self.w = -1 + 2 * np.random.rand(n_outputs, n_inputs)
        self.b = -1 + 2 * np.random.rand(n_outputs, 1)
        self.f = funcionActivacion

    def predict(self, X):
        Z = self.w @ X + self.b
        return self.f(Z)
    
    def fit(self, X, Y, epocas=500, factorAprendizaje=0.1):
        p = X.shape[1]
        for _ in range(epocas):
            # Propagar la red
            Z = self.w @ X + self.b
            Yest, dY = self.f(Z, derivada=True)
            
            # Calcular el gradiente
            lg = (Y - Yest) * dY

            # Actualización de parámetros
            self.w += (factorAprendizaje/p) * lg @ X.T
            self.b += (factorAprendizaje/p) * np.sum(lg)

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

        self.valorY = tk.StringVar(value=str(0))
        valorYFrame = tk.Frame(acciones)
        valorYLabel = tk.Label(master=valorYFrame, text="Valor Y:")
        valorYLabel.pack(side='top')
        valorYEntry = tk.Entry(master=valorYFrame, textvariable=self.valorY, state='disabled')
        valorYEntry.pack(side='bottom')
        valorYFrame.pack(padx=10, pady=10)

        # Generar el primer dato al azar
        colorRandomRGB = tuple(np.random.choice(range(256), size=3))
        valor = np.average(np.asarray(colorRandomRGB)/255)
        self.colorY = '#%02x%02x%02x' % colorRandomRGB
        self.valorY.set(str(valor))

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
        valor = np.average(np.asarray(colorRGB[0])/255)
        self.colorY = colorRGB[1]
        self.valorY.set(str(valor))

    def generarDato(self, event):
        valorDeseado = float(self.valorY.get())
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