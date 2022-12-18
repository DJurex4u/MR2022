# -*- coding: utf-8 -*-

import errno
import os
import os.path
import sys

# Make sure that we are using QT5
import matplotlib
import networkx as nx
from PyQt5 import QtCore, QtWidgets
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor

matplotlib.use('Qt5Agg')

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import seaborn as sb

import numpy as np

# Change current working directory to this script path
import pathlib

pathlib.Path(__file__).parent.absolute()
os.chdir(pathlib.Path(__file__).parent.absolute())

####Global NN parameters######
N_PER_LAYER = (10, 10)  # Neurons per layer
ALPHA = 0.00001  # Alpha value, regularization term
ACTIVATION_F = 1  # Neuron activation function, 1 = "logistic"
SOLVER = 0  # Solver algorithm, 1 = "lbfgs"
MAX_ITER = 1000  # Max number of training iterations
##############################
activationKey = ['identity', 'logistic', 'tanh', 'relu']
learningAlgorithm = ['lbfgs', 'sgd', 'adam']
####Other global variables####
# Load learn and test data
learn_data_10 = np.genfromtxt("func_10.csv", delimiter=',', dtype='float32')
learn_data_30 = np.genfromtxt("func_30.csv", delimiter=',', dtype='float32')
learn_data_60 = np.genfromtxt("func_60.csv", delimiter=',', dtype='float32')
test_data = np.genfromtxt("func_75.csv", delimiter=',', dtype='float32')

activationKey = ['identity', 'logistic', 'tanh', 'relu']
learningAlgorithm = ['lbfgs', 'sgd', 'adam']
filename = ""
LEARN_DATA = learn_data_10
logger = open('neural_network_regression.txt', 'w')
learn_data_vector = [learn_data_10, learn_data_30, learn_data_60]
neurons_in_layer = [5, 10, 30]
num_layers = [1, 2, 3]

mse_best = 100.0
combination_best = ""
combs_layers1 = []
combs_layers2 = []
combs_layers3 = []

for i in neurons_in_layer:
    combs_layers1.append((i,))
    for j in neurons_in_layer:
        combs_layers2.append((i, j,))
        for k in neurons_in_layer:
            combs_layers3.append((i, j, k))


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def createFilenamePathAndName():
    path_dir = ""

    path_dir += "layers_" + str(len(N_PER_LAYER)) + "/"

    path_dir += SOLVER + "/"

    path_file = path_dir
    for layer in N_PER_LAYER:
        path_file += str(layer) + "_"

    path_file += ACTIVATION_F + "_"

    if np.array_equal(LEARN_DATA, learn_data_10):
        path_file += "func10"
    if np.array_equal(LEARN_DATA, learn_data_30):
        path_file += "func30"
    if np.array_equal(LEARN_DATA, learn_data_60):
        path_file += "func60"

    return path_dir, path_file


# Function for visualizing scikit-learn neural networks
def VisualizeNN(mpl_axes, mpl_figure, mlp, use_weighted_edges=True, show_neuron_labels=True, show_bias=False):
    # Determine number of layers. Needed for width of the graph
    no_layers = mlp.n_layers_

    g_height = 0
    layer_sizes = ()
    try:
        # Determine max number of neurons per layer. It will represent the height of the graph
        g_height = max(mlp.hidden_layer_sizes)
        # Determining number of neurons per layer
        layer_sizes = (mlp.coefs_[0].shape[0],) + mlp.hidden_layer_sizes + (
            mlp.n_outputs_,)  # Imput layer + hidden layers + output layer
    except:
        # Determine max number of neurons per layer. It will represent the height of the graph
        g_height = mlp.hidden_layer_sizes
        # Determining number of neurons per layer
        layer_sizes = (mlp.coefs_[0].shape[0],) + (mlp.hidden_layer_sizes,) + (
            mlp.n_outputs_,)  # Imput layer + hidden layers + output layer

    # Crating graph
    g = nx.Graph()

    # Adding neurons per layer
    curr_n = 0
    for l in range(no_layers):
        for n in range(layer_sizes[l]):
            g.add_node(curr_n, pos=(l * 3, (-1.0) * (g_height / 2.0 - layer_sizes[
                l] / 2.0 + n)))  # (-1.0 * ...) in order to be inverted on Y axes, othervise (0,0) is lower left corner
            curr_n += 1

    # Adding edges
    curr_n = 0
    start_next_layer_idx = 0
    for l in range(no_layers - 1):
        start_next_layer_idx += layer_sizes[l]
        for n in range(layer_sizes[l]):
            for n_next in range(layer_sizes[l + 1]):
                g.add_edge(curr_n, start_next_layer_idx + n_next, weight=mlp.coefs_[l][n][n_next])
            curr_n += 1

    # Add bias nodes if requested
    if show_bias:
        # Adding bias nodes
        for l in range(1, no_layers):  # Start from first hidden layer
            g.add_node('b' + str(l), pos=(l * 3 + 1.5, 1.0))

            # Adding bias edges
        curr_n = layer_sizes[0]
        for l in range(1, no_layers):  # Start from first hidden layer
            for n in range(layer_sizes[l]):
                g.add_edge(curr_n, 'b' + str(l), weight=mlp.intercepts_[l - 1][n])
                curr_n += 1

    # Drawing
    pos = nx.get_node_attributes(g, 'pos')
    if use_weighted_edges:
        weights = nx.get_edge_attributes(g, "weight")
        weights_v = list(weights.values())
        nx.draw(g, pos, ax=mpl_axes, edge_color=weights_v, edge_cmap=plt.cm.Spectral, with_labels=show_neuron_labels,
                font_weight='bold')
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Spectral, norm=plt.Normalize(vmin=min(weights_v), vmax=max(weights_v)))
        sm._A = []
        mpl_figure.colorbar(sm)

    else:
        nx.draw(g, pos, ax=mpl_axes, with_labels=show_neuron_labels, font_weight='bold')


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)

        FigureCanvas.__init__(self, self.fig)

        self.setParent(parent)

        FigureCanvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def Reset(self):
        self.fig.clf()
        self.axes = self.fig.add_subplot(111)

    def saveFig(self, figName):
        self.fig.savefig(figName)


class Ui_MainWindow(QtWidgets.QMainWindow):
    def setupUi(self):
        self.setObjectName("MainWindow")
        self.resize(1010, 915)
        self.setWindowTitle("NN - Approximation")
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")

        self.lineChartWidget = QtWidgets.QWidget(self.centralwidget)
        self.lineChartWidget.setGeometry(QtCore.QRect(10, 10, 600, 450))

        self.lineChart = MplCanvas(self.lineChartWidget, 600, 450)
        self.lineChart.setObjectName("lineChart")

        self.lineChartWidgetlayout = QtWidgets.QVBoxLayout()
        self.lineChartWidgetlayout.addWidget(self.lineChart)
        self.lineChartWidgetlayout.addWidget(NavigationToolbar(self.lineChart, self.lineChartWidget))
        self.lineChartWidget.setLayout(self.lineChartWidgetlayout)

        self.NNChartWidget = QtWidgets.QWidget(self.centralwidget)
        self.NNChartWidget.setGeometry(QtCore.QRect(10, 470, 1000, 450))

        self.NNChart = MplCanvas(self.NNChartWidget, 1000, 450)
        self.NNChart.setObjectName("NNChart")

        self.NNChartWidgetlayout = QtWidgets.QVBoxLayout()
        self.NNChartWidgetlayout.addWidget(self.NNChart)
        self.NNChartWidgetlayout.addWidget(NavigationToolbar(self.NNChart, self.NNChartWidget))
        self.NNChartWidget.setLayout(self.NNChartWidgetlayout)

        self.nnParams = QtWidgets.QGroupBox(self.centralwidget)
        self.nnParams.setGeometry(QtCore.QRect(630, 10, 200, 165))
        self.nnParams.setObjectName("nnParams")
        self.nnParams.setTitle("NN parameters")

        self.label1 = QtWidgets.QLabel(self.nnParams)
        self.label1.setGeometry(QtCore.QRect(10, 20, 91, 16))
        self.label1.setObjectName("label1")
        self.label1.setText("Neurons per layer:")

        self.label2 = QtWidgets.QLabel(self.nnParams)
        self.label2.setGeometry(QtCore.QRect(10, 50, 91, 16))
        self.label2.setObjectName("label2")
        self.label2.setText("Alpha:")

        self.label3 = QtWidgets.QLabel(self.nnParams)
        self.label3.setGeometry(QtCore.QRect(10, 80, 91, 16))
        self.label3.setObjectName("label3")
        self.label3.setText("Activation function:")

        self.label4 = QtWidgets.QLabel(self.nnParams)
        self.label4.setGeometry(QtCore.QRect(10, 110, 91, 16))
        self.label4.setObjectName("label4")
        self.label4.setText("Solver:")

        self.label5 = QtWidgets.QLabel(self.nnParams)
        self.label5.setGeometry(QtCore.QRect(10, 140, 91, 16))
        self.label5.setObjectName("label5")
        self.label5.setText("Max iterations:")

        self.tbxNeuroPerLayer = QtWidgets.QLineEdit(self.nnParams)
        self.tbxNeuroPerLayer.setGeometry(QtCore.QRect(120, 20, 71, 20))
        self.tbxNeuroPerLayer.setObjectName("tbxNeuroPerLayer")

        self.tbxAlpha = QtWidgets.QLineEdit(self.nnParams)
        self.tbxAlpha.setGeometry(QtCore.QRect(120, 50, 71, 20))
        self.tbxAlpha.setObjectName("tbxAlpha")

        self.comboActivation = QtWidgets.QComboBox(self.nnParams)
        self.comboActivation.setGeometry(QtCore.QRect(120, 80, 71, 20))
        self.comboActivation.setObjectName("comboActivation")
        self.comboActivation.addItem("identity")
        self.comboActivation.addItem("logistic")
        self.comboActivation.addItem("tanh")
        self.comboActivation.addItem("relu")

        self.comboSolver = QtWidgets.QComboBox(self.nnParams)
        self.comboSolver.setGeometry(QtCore.QRect(120, 110, 71, 20))
        self.comboSolver.setObjectName("comboSolver")
        self.comboSolver.addItem("lbfgs")
        self.comboSolver.addItem("sgd")
        self.comboSolver.addItem("adam")

        self.tbxMaxIter = QtWidgets.QLineEdit(self.nnParams)
        self.tbxMaxIter.setGeometry(QtCore.QRect(120, 140, 71, 20))
        self.tbxMaxIter.setObjectName("tbxMaxIter")

        self.label6 = QtWidgets.QLabel(self.centralwidget)
        self.label6.setGeometry(QtCore.QRect(620, 230, 91, 16))
        self.label6.setObjectName("label6")
        self.label6.setText("Training sample:")

        self.comboSample = QtWidgets.QComboBox(self.centralwidget)
        self.comboSample.setGeometry(QtCore.QRect(720, 230, 100, 20))
        self.comboSample.setObjectName("comboSample")
        self.comboSample.addItem("func_10.csv")
        self.comboSample.addItem("func_30.csv")
        self.comboSample.addItem("func_60.csv")

        self.btnStart = QtWidgets.QPushButton(self.centralwidget)
        self.btnStart.setGeometry(QtCore.QRect(650, 270, 150, 50))
        self.btnStart.setObjectName("btnStart")
        self.btnStart.setText("Start")

        self.cbxShowNeuronLabels = QtWidgets.QCheckBox(self.centralwidget)
        self.cbxShowNeuronLabels.setGeometry(QtCore.QRect(300, 460, 100, 17))
        self.cbxShowNeuronLabels.setObjectName("cbxShowNeuronLabels")
        self.cbxShowNeuronLabels.setText("Show neuron labels")

        self.cbxShowWeights = QtWidgets.QCheckBox(self.centralwidget)
        self.cbxShowWeights.setGeometry(QtCore.QRect(500, 460, 100, 17))
        self.cbxShowWeights.setObjectName("cbxShowWeights")
        self.cbxShowWeights.setText("Show weights")

        self.cbxShowBias = QtWidgets.QCheckBox(self.centralwidget)
        self.cbxShowBias.setGeometry(QtCore.QRect(700, 460, 100, 17))
        self.cbxShowBias.setObjectName("cbxShowBias")
        self.cbxShowBias.setText("Show bias")

        self.setCentralWidget(self.centralwidget)
        QtCore.QMetaObject.connectSlotsByName(self)

        # Connect events
        self.btnStart.clicked.connect(self.btnStart_Click)

        # Set default GA variables
        self.tbxNeuroPerLayer.insert(str(N_PER_LAYER))
        self.tbxAlpha.insert(str(ALPHA))
        self.comboActivation.setCurrentIndex(ACTIVATION_F)
        self.comboSolver.setCurrentIndex(SOLVER)
        self.tbxMaxIter.insert(str(MAX_ITER))

    def btnStart_Click(self):
        # Set global variables from information on UI
        global N_PER_LAYER
        global ALPHA
        global ACTIVATION_F
        global SOLVER
        global MAX_ITER
        N_PER_LAYER = eval(self.tbxNeuroPerLayer.text())
        ALPHA = float(self.tbxAlpha.text())
        ACTIVATION_F = self.comboActivation.currentText()
        SOLVER = self.comboSolver.currentText()
        MAX_ITER = int(self.tbxMaxIter.text())

        # Create neural network
        mlp = MLPRegressor(hidden_layer_sizes=N_PER_LAYER,
                           activation=ACTIVATION_F,
                           solver=SOLVER,
                           alpha=ALPHA,
                           max_iter=MAX_ITER)
        # Ispravno instancirati odgovarajuci tip neurnonskih mreza, uzimajuci u obzir podatke zadane preko sucelja koje
        # su pohranjene u iznad navedene globalne varijable

        # Learn neural network
        if self.comboSample.currentText() == "func_10.csv":
            mlp.fit(learn_data_10[:, 0].reshape(learn_data_10.shape[0], 1), learn_data_10[:, 1])
        elif self.comboSample.currentText() == "func_30.csv":
            mlp.fit(learn_data_30[:, 0].reshape(learn_data_30.shape[0], 1), learn_data_30[:, 1])
        elif self.comboSample.currentText() == "func_60.csv":
            mlp.fit(learn_data_60[:, 0].reshape(learn_data_60.shape[0], 1), learn_data_60[:, 1])

        output = mlp.predict(test_data[:, 0].reshape(test_data.shape[0], 1))

        # Draw function
        self.lineChart.Reset()
        self.lineChart.axes.plot(test_data[:, 1])
        self.lineChart.axes.plot(output)
        # self.lineChart.draw()
        # Draw NN chart
        # self.NNChart.Reset()
        # VisualizeNN(self.NNChart.axes, self.NNChart.figure, mlp, use_weighted_edges=self.cbxShowWeights.isChecked(),
        #             show_neuron_labels=self.cbxShowNeuronLabels.isChecked(), show_bias=self.cbxShowBias.isChecked())
        # plt.close()  # Close anomalous popping figure?!?
        # self.NNChart.draw()

        # Calculate MSE
        mse = np.mean((test_data[:, 1] - output) ** 2)
        print("MSE = ", mse)

    def startCombination(self):
        # Set global variables from information on UI
        global N_PER_LAYER
        global ALPHA
        global ACTIVATION_F
        global SOLVER
        global MAX_ITER
        global mse_best
        global combination_best

        # Create neural network
        mlp = MLPRegressor(hidden_layer_sizes=N_PER_LAYER,
                           activation=ACTIVATION_F,
                           solver=SOLVER,
                           alpha=ALPHA,
                           max_iter=MAX_ITER)

        # Ispravno instancirati odgovarajuci tip neurnonskih mreza, uzimajuci u obzir podatke zadane preko sucelja koje
        # su pohranjene u iznad navedene globalne varijable

        # Learn neural network
        mlp.fit(LEARN_DATA[:, 0].reshape(LEARN_DATA.shape[0], 1), LEARN_DATA[:, 1])

        output = mlp.predict(test_data[:, 0].reshape(test_data.shape[0], 1))

        # create filepath and filename
        combination_dirpath, combination_filepath = createFilenamePathAndName()
        mkdir_p(combination_dirpath)

        # plot test data and NN output on the same graph and save
        plt.figure()
        plt.plot(test_data[:, 1])
        plt.plot(output, '.')
        plt.title(combination_filepath)
        plt.savefig(combination_filepath + ".png")
        plt.close()

        # Calculate MSE
        mse = mean_squared_error(test_data[:, 1], output)

        # Write MSE to file
        logger.write(combination_filepath + ": MSE = " + str(mse) + "\n")

        print("Current comb: " + combination_filepath)
        print("Best comb: " + combination_best)

        # Check for the best combination based on MSE
        if mse < mse_best:
            mse_best = mse
            combination_best = combination_filepath
        return mse


def getHeatmapPath(num_layer):
    learn_data_str = ""
    if np.array_equal(LEARN_DATA, learn_data_10):
        learn_data_str = "data10"
    elif np.array_equal(LEARN_DATA, learn_data_30):
        learn_data_str = "data30"
    elif np.array_equal(LEARN_DATA, learn_data_60):
        learn_data_str = "data60"

    htmp_path = "layers_" + str(num_layer) + "/" + SOLVER + "/" + learn_data_str + ".png"
    return htmp_path


def plotHeatmap(data, comb_layers, path, activation_fs=None):
    if activation_fs is None:
        activation_fs = activationKey

    plt.subplots(figsize=(20, 15))
    sb.set(font_scale=1.4)
    heat_map = sb.heatmap(data, xticklabels=activation_fs, yticklabels=comb_layers, annot=True, annot_kws={'size': 16}, cbar=False)
    # heat_map.set_yticklabels()
    plt.yticks(rotation=0)
    plt.title(path[:-4].replace("/", " "))
    plt.savefig(path)
    # plt.show()
    plt.close()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_MainWindow()
    ui.setupUi()
    # ui.show()

    for data in learn_data_vector:
        LEARN_DATA = data

        for learn_alg in learningAlgorithm:
            SOLVER = learn_alg

            for num_layer in num_layers:
                if num_layer == 1:
                    htmp_mse = np.zeros((len(combs_layers1), len(activationKey)))
                    i = 0
                    for neurons in combs_layers1:
                        N_PER_LAYER = neurons
                        j = 0
                        for activation_f in activationKey:
                            ACTIVATION_F = activation_f
                            htmp_mse[i, j] = ui.startCombination()
                            j += 1
                        i += 1
                    plotHeatmap(np.log10(htmp_mse), combs_layers1, getHeatmapPath(num_layer))

                elif num_layer == 2:
                    htmp_mse = np.zeros((len(combs_layers2), len(activationKey)))
                    i = 0
                    for neurons in combs_layers2:
                        N_PER_LAYER = neurons
                        j = 0
                        for activation_f in activationKey:
                            ACTIVATION_F = activation_f
                            htmp_mse[i, j] = ui.startCombination()
                            j += 1
                        i += 1
                    plotHeatmap(np.log10(htmp_mse), combs_layers2, getHeatmapPath(num_layer))

                elif num_layer == 3:
                    htmp_mse = np.zeros((len(combs_layers3), len(activationKey)))
                    i = 0
                    for neurons in combs_layers3:
                        N_PER_LAYER = neurons
                        j = 0
                        for activation_f in activationKey:
                            ACTIVATION_F = activation_f
                            htmp_mse[i, j] = ui.startCombination()
                            j += 1
                        i += 1
                    plotHeatmap(np.log10(htmp_mse), combs_layers3, getHeatmapPath(num_layer))

    logger.write("\n BEST IS: " + combination_best + " WITH MSE = " + str(mse_best))
    logger.close()
    sys.exit(app.exec_())