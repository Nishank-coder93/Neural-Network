# Author: Nishank Bhatnagar
# Description: A Python built in GUI application that shows the Boundary for seprable input values
# on the graph using Perceptron Neural Network and Unified Learning Rule 

import tkinter as tk
from tkinter import simpledialog
from tkinter import filedialog
import perceptron as k02  # This module is for plotting components


class WidgetsWindow(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.menu_bar = MenuBar(self, self, background='orange')  # MenuBar

        self.status_bar = StatusBar(self, self,bg='white',bd=1, relief=tk.RIDGE) # Status Bar

        self.center_frame = tk.Frame(self)
        # Create a frame for plotting graphs
        self.left_frame = PlotsDisplayFrame(self, self.center_frame, bg='blue')
        self.display_activation_functions = k02.DisplayActivationFunctions(self, self.center_frame)

        self.center_frame.grid(row=1, column=0,sticky=tk.N + tk.E + tk.S + tk.W)
        self.center_frame.grid_propagate(True)
        self.center_frame.rowconfigure(1, weight=1, uniform='xx')
        self.center_frame.columnconfigure(0, weight=1, uniform='xx')
        self.center_frame.columnconfigure(1, weight=1, uniform='xx')
        self.status_bar.grid(row=2, column=0,sticky=tk.N + tk.E + tk.S + tk.W)
        self.status_bar.rowconfigure(2, minsize=20)
        self.left_frame.grid(row=0,columnspan=2 ,sticky=tk.N + tk.E + tk.S + tk.W)
        # self.right_frame.grid(row=0, column=1,sticky=tk.N + tk.E + tk.S + tk.W)

class MenuBar(tk.Frame):
    def __init__(self, root, master, *args, **kwargs):
        tk.Frame.__init__(self, master, *args, **kwargs)
        self.root = root
        self.menu = tk.Menu(self.root)
        root.config(menu=self.menu)
        self.filemenu = tk.Menu(self.menu)
        self.menu.add_cascade(label="File", menu=self.filemenu)
        self.filemenu.add_command(label="New", command=self.menu_callback)
        self.filemenu.add_command(label="Open...", command=self.menu_callback)
        self.filemenu.add_separator()
        self.filemenu.add_command(label="Exit", command=self.menu_callback)
        self.dummymenu = tk.Menu(self.menu)
        self.menu.add_cascade(label="Dummy", menu=self.dummymenu)
        self.dummymenu.add_command(label="Item1", command=self.menu_item1_callback)
        self.dummymenu.add_command(label="Item2", command=self.menu_item2_callback)
        self.helpmenu = tk.Menu(self.menu)
        self.menu.add_cascade(label="Help", menu=self.helpmenu)
        self.helpmenu.add_command(label="About...", command=self.menu_help_callback)

    def menu_callback(self):
        self.root.status_bar.set('%s', "called the menu callback!")

    def menu_help_callback(self):
        self.root.status_bar.set('%s', "called the help menu callback!")

    def menu_item1_callback(self):
        self.root.status_bar.set('%s', "called item1 callback!")

    def menu_item2_callback(self):
        self.root.status_bar.set('%s', "called item2 callback!")


class StatusBar(tk.Frame):
    def __init__(self, root, master, *args, **kwargs):
        tk.Frame.__init__(self, master, *args, **kwargs)
        self.label = tk.Label(self)
        self.label.grid(row=0,sticky=tk.N + tk.E + tk.S + tk.W)

    def set(self, format, *args):
        self.label.config(text=format % args)
        self.label.update_idletasks()

    def clear(self):
        self.label.config(text="")
        self.label.update_idletasks()


class PlotsDisplayFrame(tk.Frame):
    def __init__(self, root, master, *args, **kwargs):
        tk.Frame.__init__(self, master, *args, **kwargs)
        self.root = root
        # self.configure(width=500, height=500)
        self.bind("<ButtonPress-1>", self.left_mouse_click_callback)
        self.bind("<ButtonRelease-1>", self.left_mouse_release_callback)
        self.bind("<B1-Motion>", self.left_mouse_down_motion_callback)
        self.bind("<ButtonPress-3>", self.right_mouse_click_callback)
        self.bind("<ButtonRelease-3>", self.right_mouse_release_callback)
        self.bind("<B3-Motion>", self.right_mouse_down_motion_callback)
        self.bind("<Key>", self.key_pressed_callback)
        self.bind("<Up>", self.up_arrow_pressed_callback)
        self.bind("<Down>", self.down_arrow_pressed_callback)
        self.bind("<Right>", self.right_arrow_pressed_callback)
        self.bind("<Left>", self.left_arrow_pressed_callback)
        self.bind("<Shift-Up>", self.shift_up_arrow_pressed_callback)
        self.bind("<Shift-Down>", self.shift_down_arrow_pressed_callback)
        self.bind("<Shift-Right>", self.shift_right_arrow_pressed_callback)
        self.bind("<Shift-Left>", self.shift_left_arrow_pressed_callback)
        self.bind("f", self.f_key_pressed_callback)
        self.bind("b", self.b_key_pressed_callback)


    def key_pressed_callback(self, event):
        self.root.status_bar.set('%s', 'Key pressed')

    def up_arrow_pressed_callback(self, event):
        self.root.status_bar.set('%s', "Up arrow was pressed")

    def down_arrow_pressed_callback(self, event):
        self.root.status_bar.set('%s', "Down arrow was pressed")

    def right_arrow_pressed_callback(self, event):
        self.root.status_bar.set('%s', "Right arrow was pressed")

    def left_arrow_pressed_callback(self, event):
        self.root.status_bar.set('%s', "Left arrow was pressed")

    def shift_up_arrow_pressed_callback(self, event):
        self.root.status_bar.set('%s', "Shift up arrow was pressed")

    def shift_down_arrow_pressed_callback(self, event):
        self.root.status_bar.set('%s', "Shift down arrow was pressed")

    def shift_right_arrow_pressed_callback(self, event):
        self.root.status_bar.set('%s', "Shift right arrow was pressed")

    def shift_left_arrow_pressed_callback(self, event):
        self.root.status_bar.set('%s', "Shift left arrow was pressed")

    def f_key_pressed_callback(self, event):
        self.root.status_bar.set('%s', "f key was pressed")

    def b_key_pressed_callback(self, event):
        self.root.status_bar.set('%s', "b key was pressed")

    def left_mouse_click_callback(self, event):
        self.root.status_bar.set('%s', 'Left mouse button was clicked. ' + \
                                'x=' + str(event.x) + '   y=' + str(event.y))
        self.x = event.x
        self.y = event.y
        self.canvas.focus_set()

    def left_mouse_release_callback(self, event):
        self.root.status_bar.set('%s', 'Left mouse button was released. ' + \
                                'x=' + str(event.x) + '   y=' + str(event.y))
        self.x = None
        self.y = None

    def left_mouse_down_motion_callback(self, event):
        self.root.status_bar.set('%s', 'Left mouse down motion. ' + \
                                'x=' + str(event.x) + '   y=' + str(event.y))
        self.x = event.x
        self.y = event.y

    def right_mouse_click_callback(self, event):
        self.root.status_bar.set('%s', 'Right mouse down motion. ' + \
                                'x=' + str(event.x) + '   y=' + str(event.y))
        self.x = event.x
        self.y = event.y

    def right_mouse_release_callback(self, event):
        self.root.status_bar.set('%s', 'Right mouse button was released. ' + \
                                'x=' + str(event.x) + '   y=' + str(event.y))
        self.x = None
        self.y = None

    def right_mouse_down_motion_callback(self, event):
        self.root.status_bar.set('%s', 'Right mouse down motion. ' + \
                                'x=' + str(event.x) + '   y=' + str(event.y))
        self.x = event.x
        self.y = event.y

    def left_mouse_click_callback(self, event):
        self.root.status_bar.set('%s', 'Left mouse button was clicked. ' + \
                                 'x=' + str(event.x) + '   y=' + str(event.y))
        self.x = event.x
        self.y = event.y
        self.focus_set()



def close_window_callback(root):
    if tk.messagebox.askokcancel("Quit", "Do you really wish to quit?"):
        root.destroy()

widgets_window = WidgetsWindow()
widgets_window.title('Assignment_01 --  Bhatnagar')
widgets_window.minsize(600,300)
widgets_window.protocol("WM_DELETE_WINDOW", lambda root_window=widgets_window: close_window_callback(root_window))
widgets_window.mainloop()