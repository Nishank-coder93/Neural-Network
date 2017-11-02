# Author: Nishank Bhatnagar

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import tkinter as tk


class DisplayActivationFunctions:
    """
    This class is for displaying Boundary functions for NN.
    It calculates the Boundary function using the Perceptron Unified Learning Rule 

    A = f(WX + b)

    e = t - A
    Wnew = Wold + e(X)
    Bnew = Bold + e

    A <- Is the Predicted value
    t <- Is the target Value
    W <- Is the Weight matrix
    X <- Is the randomly generated Input Matrix
    b/B <- Is the Bias value
    """

    def __init__(self, root, master, *args, **kwargs):
        self.master = master
        self.root = root
        #########################################################################
        #  Set up the constants and default values
        #########################################################################
        self.xmin = -10
        self.xmax = 10
        self.ymin = -10
        self.ymax = 10
        self.input_weight_1 = 1  # Default Weight 1 Value
        self.input_weight_2 = 1  # Default Weight 2 Value
        self.bias = 0  # Default Bias Value

        self.weight_vector = np.array([self.input_weight_1,self.input_weight_2])
        self.input_vector,self.input_vector_class = self.randomDataPoints()
        self.activation_function = "Symmetrical Hard Limit"
        self.predicted_value = np.ones(4)

        #########################################################################
        #  Set up the plotting area
        #########################################################################

        self.plot_frame = tk.Frame(self.master)
        self.plot_frame.grid(row=0, column=0, columnspan=2, sticky=tk.N + tk.E + tk.S + tk.W)
        self.plot_frame.rowconfigure(0, weight=1)
        self.plot_frame.columnconfigure(0, weight=1)
        self.figure = plt.figure("")
        self.axes = self.figure.gca()
        self.axes.set_xlabel('Input X ')
        self.axes.set_ylabel('Input Y ')
        self.axes.set_title("")
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)

        self.plot_widget = self.canvas.get_tk_widget()
        self.plot_widget.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        #########################################################################
        #  Set up the frame for sliders (scales)
        #########################################################################
        self.sliders_frame = tk.Frame(self.master, highlightbackground="black", highlightthickness=1,bd=1)
        self.sliders_frame.grid(row=1, column=0, sticky=tk.N + tk.E + tk.S + tk.W)
        self.sliders_frame.rowconfigure(0, weight=10)
        self.sliders_frame.rowconfigure(1, weight=2)
        self.sliders_frame.columnconfigure(0, weight=5, uniform='xx')
        self.sliders_frame.columnconfigure(1, weight=1, uniform='xx')
        # set up the sliders

        ## Setting up Weight One Slider
        self.input_weight_1_slider = tk.Scale(self.sliders_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                            from_=-10.0, to_=10.0, resolution=0.01, bg="#DDDDDD",
                                            activebackground="#FF0000",
                                            highlightcolor="#00FFFF",
                                            label="Input Weight 1",
                                            command=lambda event: self.input_weight_1_slider_callback())
        self.input_weight_1_slider.set(self.input_weight_1)
        self.input_weight_1_slider.bind("<ButtonRelease-1>", lambda event: self.input_weight_1_slider_callback())
        self.input_weight_1_slider.grid(row=0, column=0, columnspan=2,sticky=tk.N + tk.E + tk.S + tk.W)

        ## Setting Up Second Weight Slider
        self.input_weight_2_slider = tk.Scale(self.sliders_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                            from_=-10.0, to_=10.0, resolution=0.01, bg="#DDDDDD",
                                            activebackground="#FF0000",
                                            highlightcolor="#00FFFF",
                                            label="Input Weight 2",
                                            command=lambda event: self.input_weight_2_slider_callback())
        self.input_weight_2_slider.set(self.input_weight_2)
        self.input_weight_2_slider.bind("<ButtonRelease-1>", lambda event: self.input_weight_2_slider_callback())
        self.input_weight_2_slider.grid(row=1, column=0, columnspan=2, sticky=tk.N + tk.E + tk.S + tk.W)

        ## Bias Slider
        self.bias_slider = tk.Scale(self.sliders_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                    from_=-10.0, to_=10.0, resolution=0.01, bg="#DDDDDD",
                                    activebackground="#FF0000",
                                    highlightcolor="#00FFFF",
                                    label="Bias",
                                    command=lambda event: self.bias_slider_callback())
        self.bias_slider.set(self.bias)
        self.bias_slider.bind("<ButtonRelease-1>", lambda event: self.bias_slider_callback())
        self.bias_slider.grid(row=2, column=0, columnspan=2,sticky=tk.N + tk.E + tk.S + tk.W)

        #########################################################################
        #  Set up the frame for button(s)
        #########################################################################

        self.buttons_frame = tk.Frame(self.master, highlightbackground="black", highlightthickness=1,bd=1)
        self.buttons_frame.grid(row=1, column=1, sticky=tk.N + tk.E + tk.S + tk.W)

        self.buttons_frame.columnconfigure(0, weight=1, uniform='xx')
        self.train_button = tk.Button(self.buttons_frame , text="Train", command=lambda: self.train_data())

        self.train_button.grid(row=0,column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        self.gen_data_button = tk.Button(self.buttons_frame,
                                         text="Create Data",
                                         command=lambda: self.generate_random_data())
        self.gen_data_button.grid(row=1, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        self.activation_function_variable = tk.StringVar()
        self.activation_function_dropdown = tk.OptionMenu(self.buttons_frame, self.activation_function_variable,
                                                          "Symmetrical Hard Limit", "Hyperbolic Tangent", "Linear",
                                                          command=lambda
                                                              event: self.activation_function_dropdown_callback())
        self.activation_function_variable.set("Symmetrical Hard Limit")
        self.activation_function_dropdown.grid(row=2, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        # self.display_boundary_function()
        print("Window size:", self.master.winfo_width(), self.master.winfo_height())

        #########################################################################
        #  Displaying Boundary Decision
        #########################################################################


    def calculate_activation_function(self,net_value):

        activation = 0

        if self.activation_function == "Symmetrical Hard Limit":
            print("Symmetrical Hard Limit")
            activation = np.where(net_value >= 0.0, 1, -1)
        elif self.activation_function == "Hyperbolic Tangent":
            activation = (np.exp(net_value) - np.exp(-net_value))/(np.exp(net_value) + np.exp(-net_value))
        elif self.activation_function == "Linear":
            activation = net_value

        return activation



    def display_boundary_function(self):

        p1 = -self.bias / self.input_weight_1
        p2 = -self.bias / self.input_weight_2

        x = [p1, 0]
        y = [0, p2]


        self.axes.cla()
        self.axes.plot(x, y)
        self.axes.xaxis.set_visible(True)

        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)

        for i in range(4):
            if self.input_vector_class[i][0] == 1:
                plt.scatter(self.input_vector[i][0], self.input_vector[i][1], marker='x', color='b')
            else:
                plt.scatter(self.input_vector[i][0], self.input_vector[i][1], marker='o', color='r')




        plt.title("Perceptron")
        self.canvas.draw()

    def input_weight_1_slider_callback(self):
        self.input_weight_1 = self.input_weight_1_slider.get()
        self.weight_vector = np.array([self.input_weight_1, self.input_weight_2])
        self.display_boundary_function()

    def input_weight_2_slider_callback(self):
        self.input_weight_2 = self.input_weight_2_slider.get()
        self.weight_vector = np.array([self.input_weight_1, self.input_weight_2])
        self.display_boundary_function()

    def bias_slider_callback(self):
        self.bias = self.bias_slider.get()
        print(self.bias)
        self.display_boundary_function()

    def activation_function_dropdown_callback(self):
        self.activation_function = self.activation_function_variable.get()
        # print(self.calculate_activation_function())
        self.display_boundary_function()

#############################################################################################
##
    # Generates Random Data inputs and sets the Global Input variable as that
##
    def randomDataPoints(self):
        input_dimensions = np.zeros(shape=(4, 2))
        input_dimensions_class = np.zeros(shape=(4, 3))
        classValue = [-1, 1]
        class_counter = 0
        for i in range(0, 4):
            X = np.random.randint(-10, 10)
            Y = np.random.randint(-10, 10)
            input_dimensions[i] = [X, Y]
            class_val = np.random.choice(classValue)
            if class_val == 1:
                class_counter += 1

            if class_counter > 2:
                class_val = -1

            input_dimensions_class[i] = [class_val, X, Y]

        return (input_dimensions,input_dimensions_class)

    def generate_random_data(self):
        self.input_vector,self.input_vector_class = self.randomDataPoints()
        self.display_boundary_function()
        print(self.input_vector_class)


###############################################################################################
##
    # Calculates the Net Value (WP + b)
##
    def calculate_net_value(self,w,p,b):
        wp = np.dot(w.T, p)
        return (wp + b)
##
    # Trains the Data for next 100 steps
##
    def train_data(self):

        show = 1
        for j in range(100):
            for i in range(len(self.input_vector)):
                input_data = self.input_vector[i]
                print("Input Data", input_data)
                input_class = self.input_vector_class[i][0]
                print("Input Class", input_class)
                n_val = self.calculate_net_value(self.weight_vector, input_data, self.bias)
                activation_value = self.calculate_activation_function(n_val)
                print("Activation Value",activation_value)
                e = input_class - (activation_value)
                print("Error Calculation",e)
                self.weight_vector += e * input_data
                self.bias += e
                print("Bias: ", self.bias)
                print("weight_vector: ", self.weight_vector)

            if 10*show <= 100:
                self.display_boundary_function()
                show += 1


###############################################################################################