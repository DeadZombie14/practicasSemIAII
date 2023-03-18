# Mendoza Morelos Martin Mathier 214798285
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,NavigationToolbar2Tk)
from time import sleep

epocas = 100
tiempo_espera = epocas / (epocas * 60 * 60)
funcion = 'logistica'
factorAprendizaje = 0.4
margenError = 0.4
archivoDatos = 'datos.csv'
archivoValoresDeseados = 'valoresDeseados.csv'

class App:
    def __init__(self) -> None:
        self.window = tk.Tk()
        self.window.title('Neurona Lineal Adaline')
        self.window.resizable(False,False)

        self.grafica = tk.Frame(self.window, width=600, height=600, background="#000")
        self.grafica.pack(side='left')

        self.acciones = tk.Frame(self.window)
        creardatosBtn = tk.Button(
            master = self.acciones,
            command = self.generarDatos,
            height = 2,
            width = 10,
            text = "Regenerar datos")
        creardatosBtn.pack(padx=3, pady=6, fill='both')

        procesarBtn = tk.Button(
            master = self.acciones,
            command = self.entrenarPerceptron,
            height = 2,
            width = 10,
            text = "Entrenar")
        procesarBtn.pack(padx=3, pady=(0,6), fill='both')

        actualizarBtn = tk.Button(
            master = self.acciones,
            command = self.predecirResultados,
            height = 2,
            width = 10,
            text = "Predecir")
        actualizarBtn.pack(padx=3, pady=(0,6), fill='both')

        self.acciones.pack(side='right', fill='both') 

        self.fig = plt.figure(figsize = (600/100, 600/100), dpi = 100)
        self.plot = self.fig.add_subplot()

        self.canvas = FigureCanvasTkAgg(
            figure= self.fig,
            master = self.grafica)  
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

        X = self.leerDatos(archivoDatos).T
        Y = self.leerDatos(archivoValoresDeseados)
        self.model = self.Perceptron(X.shape[0], funcion, factorAprendizaje)
        self.dibujar(X, Y, self.plot, self.model)

        # Input de b
        biasInput = tk.Frame(self.acciones)
        biasLabel = tk.Label(biasInput, text="Bias")
        biasLabel.pack(side='left')
        sv = tk.StringVar(value=str(self.model.b))
        self.inputBias = sv
        def callbackb(sv):
            self.model.b = float(sv.get())
        sv.trace_add('write', lambda name, index, mode, sv=sv: callbackb(sv))
        biasEntry = tk.Entry(biasInput, textvariable=sv)
        biasEntry.pack(side='right')
        biasInput.pack(fill='both')

        # Inputs de pesos
        self.inputsPesos = []
        for i in range(self.model.w.shape[0]):
            wInput = tk.Frame(self.acciones)
            wLabel = tk.Label(wInput, text="Peso "+str(i)+")")
            wLabel.pack(side='left')
            sv = tk.StringVar(value=str(self.model.w[i]))
            self.inputsPesos.append(sv)
            def callback(sv, i):
                self.model.w[i] = float(sv.get())
            sv.trace_add('write', lambda name, index, mode, sv=sv, i=i: callback(sv, i))
            wEntry = tk.Entry(wInput, textvariable=sv)
            wEntry.pack(side='right')
            wInput.pack()

        self.window.protocol("WM_DELETE_WINDOW", self.cerrar)
        self.window.mainloop()

    def cerrar(self):
        self.window.quit()

    def generarDatos(self):
        self.VentanaGenerarDatos(archivoValoresDeseados)
    
    def leerDatos(self, archivo):
        X = np.loadtxt(
            fname= archivo,
            delimiter=',')
        return X
        
    def entrenarPerceptron(self):
        X = self.leerDatos(archivoDatos).T
        Y = self.leerDatos(archivoValoresDeseados)
        def animar():
            self.dibujar(X, Y, self.plot, self.model)
            sleep(tiempo_espera)
            self.inputBias.set(self.model.b)
            for w in range(len(self.inputsPesos)):
                self.inputsPesos[w].set(self.model.w[w])
            self.window.update()
        self.model.entrenar(X, Y, max_epocas=epocas, animarCallback=animar)
        self.dibujar(X, Y,self.plot, self.model)
        
        print('Resultados Originales',Y)
        Y_est = []
        for i in range(X.shape[1]):
            Y_est.append( self.model.f(self.model.predecir(X[:,i])) )
        print('Resultados Predecidos',Y_est)
            

    def predecirResultados(self):
        X = self.leerDatos(archivoDatos).T
        Y = self.leerDatos(archivoValoresDeseados)
        self.dibujar(X, Y, self.plot, self.model)
    
    def dibujar(self, X, Y, plot, perceptron):
        plot.clear()
        plot.set_title('Adaline')
        plot.grid('on')
        plot.set_xlim([-2,2])
        plot.set_ylim([-2,2])
        plot.set_xlabel(r'$x_1$')
        plot.set_ylabel(r'$x_2$')
        # Dibujar puntos
        for i in range(X.shape[1]):
            if perceptron.predecir(X[:,i]) >= margenError: # Funcion f(v)
                plot.plot(X[0,i], X[1,i], 'or')
            else:
                plot.plot(X[0,i], X[1,i], 'ob')
        # Dibujar linea
        w1, w2, b = perceptron.w[0], perceptron.w[1], perceptron.b - margenError
        plot.plot([-2,2], [(1/w2)*(-w1*(-2)-b), (1/w2)*(-w1*2-b)], '--k')
        self.canvas.draw()

    class VentanaGenerarDatos:
        def __init__(self, archivo='valoresDeseados.csv') -> None:
            self.archivo = archivo
            self.datos = []
            self.valoresDeseados = []
            self.window = tk.Tk()
            self.window.title('Generar datos')
            self.window.resizable(False,False)
            guardarBtn = tk.Button(
                master = self.window,
                command = self.guardar,
                height = 2,
                width = 10,
                text = "Guardar")
            guardarBtn.pack()
            self.generarFigura() # Dibujar la grafica
            self.window.mainloop()
    
        def generarFigura(self):
            self.fig = plt.figure(figsize = (5, 5), dpi = 100)
            self.plot = self.fig.add_subplot()
            self.plot.set_title('Haz clic para colocar punto')
            self.plot.grid('on')
            self.plot.set_xlim([-2,2])
            self.plot.set_ylim([-2,2])
            self.plot.set_xlabel(r'$x_1$')
            self.plot.set_ylabel(r'$x_2$')
            
            # Conectar evento
            self.fig.canvas.mpl_connect('button_press_event', self.generarDato)
            
            canvas = FigureCanvasTkAgg(
                figure= self.fig,
                master = self.window)  
            canvas.draw()
            canvas.get_tk_widget().pack()
            # Sin barra de herramientas
            #toolbar = NavigationToolbar2Tk(canvas,self.window)
            #toolbar.update()
            canvas.get_tk_widget().pack()

        def generarDato(self, event):
            valorDeseado = tk.messagebox.askyesno(message="Valor de salida deseado", title="Dar valor a punto", parent=self.window)
            if(valorDeseado):
                self.plot.plot(event.xdata, event.ydata, 'or')
            else:
                self.plot.plot(event.xdata, event.ydata, 'ob')
            self.datos.append([event.xdata, event.ydata])
            self.valoresDeseados.append(1 if valorDeseado else 0)
            self.fig.canvas.draw()
        
        def guardar(self):
            self.crearDatos('datos.csv', self.datos)
            self.crearDatos('valoresDeseados.csv', self.valoresDeseados)
            self.window.destroy()

        def crearDatos(self, archivo, datos):
            X = np.array(datos)
            np.savetxt(
                fname= archivo,
                X= X,
                fmt='%.5f',
                delimiter=',')

    class Perceptron:
        def __init__(self, n_inputs, funcionActivacion='linear', factorAprendizaje=0.1):
            self.w = -1 + 2 * np.random.rand(n_inputs)
            self.b = -1 + 2 * np.random.rand()
            self.factorAprendizaje = factorAprendizaje
            if funcionActivacion == 'linear':
                self.f = self.linear
            elif funcionActivacion == 'logistica':
                self.f = self.logistica
            elif funcionActivacion == 'tanh':
                self.f = self.tanh
        def predecir(self, X):
            Z = np.dot(self.w, X) + self.b
            return self.f(Z)
        def entrenar(self, X, Y, max_epocas=50, animarCallback=None):
            p = X.shape[1]
            for _ in range(max_epocas):
                Z = np.dot(self.w, X) + self.b
                Y_est, dY = self.f(Z, True)
                lg = (Y - Y_est) * dY # gradiente local en variable

                self.w += (self.factorAprendizaje/p) * np.dot(lg, X.T).ravel()
                self.b += (self.factorAprendizaje/p) * np.sum(lg)
                if(animarCallback != None):
                    animarCallback()
        def linear(self, z, derivada=False):
            a = z
            if derivada:
                da = np.ones(z.shape)
                return a, da
            return a
        def logistica(self, z, derivada=False):
            a = 1 / (1 + np.exp(-z))
            if(derivada):
                da = a * (1 - a)
                return a, da
            return a
        def tanh(self, z, derivada=False):
            a = np.tanh(z)
            if(derivada):
                da = (1 - a) * (1 - a)
                return a, da
            return a
app = App()