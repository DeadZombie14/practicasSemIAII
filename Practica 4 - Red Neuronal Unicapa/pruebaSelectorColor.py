import tkinter as tk
from tkinter import ttk
from tkinter.colorchooser import askcolor
import numpy as np


root = tk.Tk()
root.title('Tkinter Color Chooser')
root.geometry('300x150')


def change_color():
    colors = askcolor(title="Tkinter Color Chooser")
    print(np.average(np.asarray(colors[0])/255)) # Media entre 1 y 0 del RGB
    root.configure(bg=colors[1])


ttk.Button(
    root,
    text='Select a Color',
    command=change_color).pack(expand=True)


root.mainloop()