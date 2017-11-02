# Author: Nishank Bhatnagar
""" 
About: Using TensorFlow to create two layer Neural Network to read and Train MNIST Dataset from Google
and to calculate the Error rate of test dataset and Loss Function using Cross Entropy or L2 loss function
 This was implemented using Pyhton's built-in GUI TKinter 
 The Nodes for Hidden Layer can be selected via slider 
 The Activation function for both Hidden Layer and Output layer can be selected from Dropdown menu
 """


import os
import matplotlib
import itertools
import scipy.misc
import pandas as pd
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import tkinter as tk
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data



class DisplayActivationFunctions:
    """
    This class is calculating the Error Rate and Loss Function for each Epoch and 
    Displaying it on the Graph using MatplotLib.
    """

    def __init__(self, root, master, *args, **kwargs):
        self.master = master
        self.root = root
        #########################################################################
        #  Set up the constants and default values
        #########################################################################
        ## X Axis 1 -100 Epochs
        self.xmin = 1
        self.xmax = 100

        ## Y Axis 0 - 100 percentage
        self.ymin = 0
        self.ymax = 100

        self.input_learing_rate = 0.1 # Default Learning Rule
        self.lambda_rate = 0.01 # Default Lambda rate
        self.num_of_nodes = 100 # Default Number of Nodes Value
        self.batch_size_number = 64 # Default Batch size
        self.percent_training_data = 10  # Default Percent training size

        ## Training and Testing data
        self.X_train = 0
        self.x_test = 0
        self.Y_train = 0
        self.y_test = 0

        ## Initial Weights
        self.initial_weights = []
        self.updated_weight = []

        self.hidden_layer = "Relu" # Default Hidden layer transfer function
        self.output_layer = "Softmax" # Default Output layer transfer function
        self.cost_function = "Cross Entropy"  # Default Cost Function

        self.predicted_value = np.ones(4)

        ## Image matrix default
        self.images_matrix = np.empty(1)
        self.image_labels = np.empty(1)
        self.input_label_encoded = []
        self.image_files = []
        self.image_names = []

        ## Error rate array
        self.error_rate = []

        ## MNIST Data
        self.mnist  = input_data.read_data_sets("MNIST_data/", one_hot=True)


        #########################################################################
        #  Setting up TensorFlow Placeholders and Variables
        #########################################################################
        self.avg_cost_lst = []
        self.class_names = np.arange(0, 10)

        self.sess = tf.Session()
        self.init_op_flag = False

        self.init_op = ""

        self.W1 = ''
        self.b1 = ''
        self.W2 = ''
        self.b2 = ''


        #########################################################################
        #  Set up the plotting area
        #########################################################################

        self.plot_frame = tk.Frame(self.master)
        self.plot_frame.grid(row=0, column=0, columnspan=2, sticky=tk.N + tk.E + tk.S + tk.W)
        self.plot_frame.rowconfigure(0, weight=1)
        self.plot_frame.columnconfigure(0, weight=1)
        self.figure = plt.figure(1)
        self.axes = self.figure.gca()
        self.axes.set_xlabel(' Epoch ')
        self.axes.set_ylabel('Percentage ')
        self.axes.set_title("Error Rate")
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)

        self.plot_widget = self.canvas.get_tk_widget()
        self.plot_widget.grid(row=0, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        self.epoch_iteration = 1

        #########################################################################
        #  Set up the another plot area
        #########################################################################

        self.plot_frame_2 = tk.Frame(self.master)
        self.plot_frame_2.grid(row=0, column=3, columnspan=1, sticky=tk.N + tk.E + tk.S + tk.W)
        self.plot_frame_2.rowconfigure(0, weight=1)
        self.plot_frame_2.columnconfigure(0, weight=1)
        self.figure_2 = plt.figure(2)
        self.axes_2 = self.figure_2.gca()
        self.axes_2.set_xlabel("Epoch")
        self.axes_2.set_ylabel("Cost Rate")
        self.axes_2.set_title("Loss Function")
        plt.xlim(1, 100)
        plt.ylim(0, 1)
        self.canvas_2 = FigureCanvasTkAgg(self.figure_2, master=self.plot_frame_2)

        self.plot_widget_2 = self.canvas_2.get_tk_widget()
        self.plot_widget_2.grid(row=0,column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        #########################################################################
        #  Set up the another plot area
        #########################################################################

        self.plot_frame_3 = tk.Frame(self.master)
        self.plot_frame_3.grid(row=1, column=0, columnspan=2, sticky=tk.N + tk.E + tk.S + tk.W)
        self.plot_frame_3.rowconfigure(0, weight=1)
        self.plot_frame_3.columnconfigure(0, weight=1)
        self.figure_3 = plt.figure(3)
        # self.axes_3 = self.figure_3.gca()
        self.canvas_3 = FigureCanvasTkAgg(self.figure_3, master=self.plot_frame_3)

        self.plot_widget_3 = self.canvas_3.get_tk_widget()
        self.plot_widget_3.grid(row=0,column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        #########################################################################
        #  Set up the frame for sliders (scales)
        #########################################################################
        self.sliders_frame = tk.Frame(self.master, highlightbackground="black", highlightthickness=1,bd=1)
        self.sliders_frame.grid(row=0, column=4, columnspan=2, sticky=tk.N + tk.E + tk.S + tk.W)
        self.sliders_frame.rowconfigure(0, weight=1)
        self.sliders_frame.rowconfigure(1, weight=1)
        self.sliders_frame.rowconfigure(2, weight=1)
        self.sliders_frame.rowconfigure(3, weight=1)
        self.sliders_frame.rowconfigure(4, weight=1)
        self.sliders_frame.columnconfigure(0, weight=2, uniform='xx')
        # self.sliders_frame.columnconfigure(1, weight=5, uniform='xx')
        # set up the sliders

        ## Setting up Learning Rule Slider
        self.learning_rule_slider = tk.Scale(self.sliders_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                            from_=0.001, to_=1.0, resolution=0.001, bg="#DDDDDD",
                                            activebackground="#FF0000",
                                            highlightcolor="#00FFFF",
                                            label="Alpha learning Rate",
                                            command=lambda event: self.learning_rule_slider_callback())
        self.learning_rule_slider.set(self.input_learing_rate)
        self.learning_rule_slider.bind("<ButtonRelease-1>", lambda event: self.learning_rule_slider_callback())
        self.learning_rule_slider.grid(row=0, column=0, columnspan=2 ,sticky=tk.N + tk.E + tk.S + tk.W)

        ## Setting up Lambda Slider
        self.lambda_slider = tk.Scale(self.sliders_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                             from_=0.0, to_=1.0, resolution=0.01, bg="#DDDDDD",
                                             activebackground="#FF0000",
                                             highlightcolor="#00FFFF",
                                             label="Lambda Rate",
                                             command=lambda event: self.lambda_slider_callback())
        self.lambda_slider.set(self.lambda_rate)
        self.lambda_slider.bind("<ButtonRelease-1>", lambda event: self.lambda_slider_callback())
        self.lambda_slider.grid(row=1, column=0, columnspan=2, sticky=tk.N + tk.E + tk.S + tk.W)

        ## Setting up Number of Nodes Slider
        self.num_of_nodes_slider = tk.Scale(self.sliders_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                      from_=1, to_=500, resolution=1, bg="#DDDDDD",
                                      activebackground="#FF0000",
                                      highlightcolor="#00FFFF",
                                      label="Number of Nodes",
                                      command=lambda event: self.num_of_nodes_slider_callback())
        self.num_of_nodes_slider.set(self.num_of_nodes)
        self.num_of_nodes_slider.bind("<ButtonRelease-1>", lambda event: self.num_of_nodes_slider_callback())
        self.num_of_nodes_slider.grid(row=2, column=0, columnspan=2, sticky=tk.N + tk.E + tk.S + tk.W)

        ## Setting up Batch size
        self.batch_size_slider = tk.Scale(self.sliders_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                            from_=1, to_=256, resolution=1, bg="#DDDDDD",
                                            activebackground="#FF0000",
                                            highlightcolor="#00FFFF",
                                            label="Batch Size",
                                            command=lambda event: self.batch_size_slider_callback())
        self.batch_size_slider.set(self.batch_size_number)
        self.batch_size_slider.bind("<ButtonRelease-1>", lambda event: self.batch_size_slider_callback())
        self.batch_size_slider.grid(row=3, column=0, columnspan=2, sticky=tk.N + tk.E + tk.S + tk.W)

        ## Setting up percent for training data
        self.training_percent_slider = tk.Scale(self.sliders_frame, variable=tk.DoubleVar(), orient=tk.HORIZONTAL,
                                            from_=1, to_=100, resolution=1, bg="#DDDDDD",
                                            activebackground="#FF0000",
                                            highlightcolor="#00FFFF",
                                            label="Percent training dataset",
                                            command=lambda event: self.training_percent_slider_callback())
        self.training_percent_slider.set(self.percent_training_data)
        self.training_percent_slider.bind("<ButtonRelease-1>", lambda event: self.training_percent_slider_callback())
        self.training_percent_slider.grid(row=4, column=0, columnspan=2, sticky=tk.N + tk.E + tk.S + tk.W)

        #########################################################################
        #  Set up the frame for button(s)
        #########################################################################

        self.buttons_frame = tk.Frame(self.master, highlightbackground="black", highlightthickness=1,bd=1)
        self.buttons_frame.grid(row=1, column=4, columnspan=2, sticky=tk.N + tk.E + tk.S + tk.W)

        self.buttons_frame.columnconfigure(1, weight=1, uniform='xx')

        ## Training Button Data by Adjusting the weights
        self.train_button = tk.Button(self.buttons_frame , text="Adjust Weights", command=lambda: self.train_data())

        self.train_button.grid(row=0,column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        ## Generating Randomized weights and biases data
        self.gen_data_button = tk.Button(self.buttons_frame,
                                         text="Randomize Data",
                                         command=lambda: self.randomize_weights())
        self.gen_data_button.grid(row=1, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        #########################################################################
        #  Sliders for Transfer function
        #########################################################################

        ## Drop Down activation function
        self.hidden_layer_variable = tk.StringVar()
        self.hidden_layer_dropdown = tk.OptionMenu(self.buttons_frame, self.hidden_layer_variable,
                                                          "Relu", "Sigmoid",
                                                          command=lambda
                                                              event: self.hidden_layer_dropdown_callback())
        self.hidden_layer_variable.set(self.hidden_layer)
        self.hidden_layer_dropdown.grid(row=2, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        ## Drop Down Selection of Learning Rule
        self.output_layer_variable = tk.StringVar()
        self.output_layer_dropdown = tk.OptionMenu(self.buttons_frame, self.output_layer_variable,
                                                          "Softmax", "Sigmoid",
                                                          command=lambda
                                                              event: self.output_layer_dropdown_callback())
        self.output_layer_variable.set(self.output_layer)
        self.output_layer_dropdown.grid(row=3, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        ## Drop Down Selection of Cost Function
        self.cost_function_variable = tk.StringVar()
        self.cost_function_dropdown = tk.OptionMenu(self.buttons_frame, self.cost_function_variable,
                                                    "Cross Entropy", "MSE",
                                                    command=lambda
                                                        event: self.cost_function_dropdown_callback())
        self.cost_function_variable.set(self.cost_function)
        self.cost_function_dropdown.grid(row=4, column=0, sticky=tk.N + tk.E + tk.S + tk.W)

        # self.display_boundary_function()
        # self.load_image_data()
        print("Window size:", self.master.winfo_width(), self.master.winfo_height())


    #########################################################################
    #  Plot_confusion_matrix
    #########################################################################

    def plot_confusion_matrix(self,cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.figure(3)
        plt.clf()
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.gcf().canvas.draw()

    #########################################################################
    #  Split Data for Training and Testing
    #########################################################################

    def split_data(self,x,y):
        # Returns the 20% test value and 80% training value

        return train_test_split(x,y,test_size=0.2)


    #########################################################################
    #       Plotting Error Rate
    #########################################################################

    def plot_error_rate(self):

        plt.figure(1)
        plt.clf()

        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)

        epochs_val = np.arange(1,len(self.error_rate) + 1)

        print("Error Rate",self.error_rate)
        print("Epoch Val",epochs_val)
        plt.plot(epochs_val,self.error_rate)
        plt.title("Assignment_02_Bhatnagar")
        plt.gcf().canvas.draw()

    def plot_loss_function(self):

        plt.figure(2)
        plt.clf()

        plt.xlim(1, 100)
        plt.ylim(0,1)

        epoch_val = np.arange(1,len(self.avg_cost_lst) + 1)
        plt.plot(epoch_val,self.avg_cost_lst)
        plt.gcf().canvas.draw()

#########################################################################
#           Slider Callback Functions
#########################################################################

## Learning Rule Slider Callback
    def learning_rule_slider_callback(self):
        self.input_learing_rule = self.learning_rule_slider.get()
        print("Input Learning Rule ",self.input_learing_rule)

## Learning Rule Slider Callback
    def lambda_slider_callback(self):
        self.lambda_rate = self.lambda_slider.get()
        print("Lambda ", self.lambda_rate)

 ## Number of Nodes Slider Callback
    def num_of_nodes_slider_callback(self):
        self.num_of_nodes = self.num_of_nodes_slider.get()
        print("Number of Nodes ", self.num_of_nodes)
        # self.change_hidden_layer_nodes()

## Batch Size Slider Callback
    def batch_size_slider_callback(self):
        self.batch_size_number = self.batch_size_slider.get()
        print("Batch Size ", self.batch_size_number)

## Percent training data Slider Callback
    def training_percent_slider_callback(self):
        self.percent_training_data = self.training_percent_slider.get()
        print("Percent ", self.percent_training_data,"%")


 #########################################################################
 #          Drop down selection function
#########################################################################

## Hidden Layer selection function Callback
    def hidden_layer_dropdown_callback(self):
        self.hidden_layer = self.hidden_layer_variable.get()
        print("Hidden Layer transfer function ", self.hidden_layer)

## Learning Rule seletion Callback
    def output_layer_dropdown_callback(self):
        self.output_layer_selection = self.output_layer_variable.get()
        # self.error_rate = []
        # self.updated_weight = self.initial_weights
        print("Output layer transfer function",self.output_layer)

## Cost Function Callback
    def cost_function_dropdown_callback(self):
        self.cost_function = self.cost_function_variable.get()
        print("Cost Function ", self.cost_function)


    #########################################################################
    #       Randomize Weight function on Button press
    #########################################################################

    def randomize_weights(self):

        # tf.Session.reset(self.sess)
        self.avg_cost_lst = []
        self.error_rate = []
        self.init_op_flag = False

        plt.figure(1)
        plt.clf()
        plt.gcf().canvas.draw()

        plt.figure(2)
        plt.clf()
        plt.gcf().canvas.draw()

        plt.figure(3)
        plt.clf()
        plt.gcf().canvas.draw()


    #########################################################################
    #       Calculate the Error rate using the Test value
    #########################################################################

    def calculate_error_rate(self,pred_label, true_label):
        error_arg = true_label - pred_label
        print("Error cal: ", error_arg)

        cnt_nonzeroes = np.count_nonzero(error_arg)
        cnt_zeroes = len(true_label) - cnt_nonzeroes
        error_percent = (cnt_nonzeroes / (cnt_zeroes + cnt_nonzeroes)) * 100
        print("Percentage error for one epoch :", error_percent)

        self.error_rate.append(error_percent)

###############################################################################################

##
    # Trains the Data for next epoch_value steps
##
    def train_data(self):
        epochs = 10

        # Declaring the training data placeholder
        # input x - for 28 X 28 pixels = 784

        x = tf.placeholder(tf.float32, shape=[None, 784])

        # Declare the output data placeholder - 10 digits

        y = tf.placeholder(tf.float32, [None, 10])

        # Setting up Weights and Bias variable for Three layer Neural Network
        # There are always L-1 number of weights and Bias tensors where L is the number of Layers

        if not self.init_op_flag:
        # Declaring weights and Bias connecting the input to the Hidden Layer
            self.W1 = tf.Variable(tf.random_normal([784, self.num_of_nodes], stddev=0.03), name='W1')
            self.b1 = tf.Variable(tf.random_normal([self.num_of_nodes]), name='b1')

        # and the weights connecting the Hidden Layer to the ouput layer
            self.W2 = tf.Variable(tf.random_normal([self.num_of_nodes, 10], stddev=0.03), name='W2')
            self.b2 = tf.Variable(tf.random_normal([10]), name='b2')

        hidden_out_net = tf.add(tf.matmul(x, self.W1), self.b1)
        hidden_out = ''
        if self.hidden_layer == "Relu":
            hidden_out = tf.nn.relu(hidden_out_net)
        elif self.hidden_layer == "Sigmoid":
            hidden_out = tf.nn.sigmoid(hidden_out_net)

        output_layer_net = tf.add(tf.matmul(hidden_out, self.W2), self.b2)
        y_output = ''

        if self.output_layer == "Softmax":
            y_output = tf.nn.softmax(output_layer_net)
        elif self.output_layer == "Sigmoid":
            y_output = tf.nn.sigmoid(output_layer_net)

        cost_func = ''

        # regularizers = tf.nn.l2_loss(self.W1) + tf.nn.l2_loss(self.W2)
        if self.cost_function == "Cross Entropy":
            # Better accuracy without Regularization
            y_clipped = tf.clip_by_value(y_output, 1e-10, 0.9999999)
            cost_func = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped) + (1 - y) * tf.log(1 - y_clipped), axis=1))
            # cost_func = tf.reduce_mean(cost_func + self.lambda_rate * regularizers)
        elif self.cost_function == "MSE":
            regularizers = tf.nn.l2_loss(self.W1) + tf.nn.l2_loss(self.W2)
            cost_func = tf.reduce_mean(tf.square(tf.subtract(y, y_output)))
            cost_func = tf.reduce_mean(cost_func + self.lambda_rate * regularizers)



        # Setting up an optimizer

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.input_learing_rate).minimize(cost_func)

        # # setting up initialization operator

        init_op = tf.global_variables_initializer()

        # Define an accuracy assesment model
        # tf.equal returns True or False tf.argmax same as np.argmax returns the index of maximum value in a vector/tensor
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_output, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        y_output_arg = tf.argmax(y_output, 1)

        # with tf.Session() as sess:
            # initialize the variables
        if not self.init_op_flag:
            self.sess.run(init_op)
            self.init_op_flag = True

        total_batch = int(len(self.mnist.train.labels) / self.batch_size_number)
        y_t_cm_arg = self.sess.run(tf.argmax(self.mnist.test.labels, 1))

        for epoch in range(epochs):
            avg_cost = 0

            for i in range(total_batch):
                batch_x, batch_y = self.mnist.train.next_batch(batch_size=self.batch_size_number)
                _, c = self.sess.run([optimizer, cost_func], feed_dict={x: batch_x, y: batch_y})

                avg_cost += c / total_batch

            self.avg_cost_lst.append(avg_cost)
            print("Epoch:", (epoch + 1), "cost=", "{:.3f}".format(avg_cost))
            y_out_epoch = self.sess.run(y_output_arg, feed_dict={x: self.mnist.test.images, y: self.mnist.test.labels})
            self.calculate_error_rate(y_out_epoch, y_t_cm_arg)

            self.plot_error_rate()
            self.plot_loss_function()
                #         print("Accuracy for Epoch:", (epoch + 1), "is ", sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}) )
        print("Accuracy (in percent)", self.sess.run(accuracy, feed_dict={x: self.mnist.test.images, y: self.mnist.test.labels}))
        y_out_cm_arg = self.sess.run(y_output_arg, feed_dict={x: self.mnist.test.images, y: self.mnist.test.labels})
        print(y_out_cm_arg)
        print(y_t_cm_arg)
        cmf_val = confusion_matrix(y_out_cm_arg, y_t_cm_arg)
        print(cmf_val)
        self.plot_confusion_matrix(cmf_val,classes=self.class_names,title="Confusion Matrix")


###############################################################################################