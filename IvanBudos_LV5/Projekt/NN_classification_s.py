# -*- coding: utf-8 -*-

import errno
import os
import os.path
import sys
import operator
# Make sure that we are using QT5
import matplotlib
import networkx as nx
from PyQt5 import QtCore, QtWidgets
from sklearn.metrics import confusion_matrix, accuracy_score, average_precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.utils.multiclass import unique_labels

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
SOLVER = 1  # Solver algorithm, 1 = "lbfgs"
MAX_ITER = 1000  # Max number of training iterations
##############################

####Other global variables####
# Load learn and test data
train_data = np.genfromtxt("iris_samples_train.csv", delimiter=',', dtype='float32', skip_header=1)
test_data = np.genfromtxt("iris_samples_test.csv", delimiter=',', dtype='float32', skip_header=1)

# MY PARAMETERS #
neurons_in_layer = [5, 10, 30]
num_layers = [1, 2, 3]
activation_f_vector = ['identity', 'logistic', 'tanh', 'relu']
learning_algorithms = ['lbfgs', 'sgd', 'adam']
filename = ""
accuracy_dict = {}
avg_prec_dict = {}

logger = open('neural_network_classification.txt', 'w')

accuracy_best = 0.0
combination_best = ""
##############################

# ALL COMBINATIONS FOR LAYER SIZES
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

    path_dir += ACTIVATION_F + "/"

    path_file = path_dir
    for layer in N_PER_LAYER:
        path_file += str(layer) + "_"

    path_file += ACTIVATION_F

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


# Function for computing and displaying confusion matrices
def PlotConfusionMatrix(mpl_axes, mpl_figure, y_true, y_pred, classes,
                        normalize=False,
                        title=None,
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    im = mpl_axes.imshow(cm, interpolation='nearest', cmap=cmap)
    mpl_axes.figure.colorbar(im, ax=mpl_axes)
    # We want to show all ticks...
    mpl_axes.set(xticks=np.arange(cm.shape[1]),
                 yticks=np.arange(cm.shape[0]),
                 # ... and label them with the respective list entries
                 xticklabels=classes, yticklabels=classes,
                 title=title,
                 ylabel='True label',
                 xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(mpl_axes.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            mpl_axes.text(j, i, format(cm[i, j], fmt),
                          ha="center", va="center",
                          color="white" if cm[i, j] > thresh else "black")
    mpl_figure.tight_layout()


def PlotConfusionMatrixPlt(y_true, y_pred, classes,
                           path,
                           normalize=False,
                           title=None,
                           cmap=plt.cm.Blues
                           ):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    # else:
    #     print('Confusion matrix, without normalization')

    # print(cm)

    htmp = sb.heatmap(cm, cmap=cmap, annot=True, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.xticks(rotation=45,
               rotation_mode="anchor")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path)
    # plt.show()
    plt.close()

    print(path_file)


def getHeatmapPath(num_layer):
    htmp_path = "layers_" + str(num_layer) + "/" + SOLVER
    return htmp_path


def plotHeatmap(data, comb_layers, path, activation_fs=None):
    if activation_fs is None:
        activation_fs = activation_f_vector

    plt.subplots(figsize=(20, 15))
    sb.set(font_scale=1.4)
    heat_map = sb.heatmap(data, xticklabels=activation_fs, yticklabels=comb_layers, annot=True, annot_kws={'size': 16},
                          cbar=False, fmt='f')
    heat_map.set_yticklabels(heat_map.get_yticklabels(), fontsize=16)
    heat_map.set_xticklabels(heat_map.get_xticklabels(), fontsize=16)
    plt.yticks(rotation=0)
    # title = path[:-4].replace("/", " ")
    plt.title(path[:-4].replace("/", " ").replace("_", " "))
    plt.tight_layout()

    plt.savefig(path)
    # plt.show()
    plt.close()
    print("Plotted heatmap: {}, {}".format(SOLVER, N_PER_LAYER))


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

    def saveFig(self, path):
        self.fig.savefig(path)


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

        self.btnStart = QtWidgets.QPushButton(self.centralwidget)
        self.btnStart.setGeometry(QtCore.QRect(650, 200, 150, 50))
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
        mlp = MLPClassifier(hidden_layer_sizes=N_PER_LAYER,
                            activation=ACTIVATION_F,
                            solver=SOLVER,
                            alpha=ALPHA,
                            max_iter=MAX_ITER)
        # Ispravno instancirati odgovarajuci tip neurnonskih mreza, uzimajuci u obzir podatke zadane preko sucelja
        # koje su pohranjene u iznad navedene globalne varijable

        # Learn neural network
        mlp.fit(train_data[:, 0:4], train_data[:, 4])
        # print(train_data[:, 4])
        output = mlp.predict(test_data[:, 0:4])
        # print(output)
        # Round up the classification
        output = np.around(output).astype('int')
        # print(output)
        # Create path and name for confusion matrix .png
        path_dir, path_file = createFilenamePathAndName()
        mkdir_p(path_dir)

        # Draw confusion matrix
        self.lineChart.Reset()
        # Name of the classes, could also be strings
        class_names = np.array([0, 1, 2])
        PlotConfusionMatrix(self.lineChart.axes, self.lineChart.figure, test_data[:, 4].astype('int'), output,
                            classes=class_names, normalize=True)
        self.lineChart.draw()
        self.lineChart.fig.savefig(path_file)

    def startCombination(self):
        # Set global variables from information on UI
        global N_PER_LAYER
        global ALPHA
        global ACTIVATION_F
        global SOLVER
        global MAX_ITER
        # N_PER_LAYER = eval(self.tbxNeuroPerLayer.text())
        # ALPHA = float(self.tbxAlpha.text())
        # ACTIVATION_F = self.comboActivation.currentText()
        # SOLVER = self.comboSolver.currentText()
        # MAX_ITER = int(self.tbxMaxIter.text())

        # Create neural network
        mlp = MLPClassifier(hidden_layer_sizes=N_PER_LAYER,
                            activation=ACTIVATION_F,
                            solver=SOLVER,
                            alpha=ALPHA,
                            max_iter=MAX_ITER)
        # Ispravno instancirati odgovarajuci tip neurnonskih mreza, uzimajuci u obzir podatke zadane preko sucelja
        # koje su pohranjene u iznad navedene globalne varijable

        # Learn neural network
        mlp.fit(train_data[:, 0:4], train_data[:, 4])

        output = mlp.predict(test_data[:, 0:4])
        # print(output)
        # Calculating accuracy and avg precision
        acc = accuracy_score(test_data[:, 4], output)
        # avg_prec = average_precision_score(test_data[:, 4], output)

        # Round up the classification
        output = np.around(output).astype('int')
        # rounded_test_classes = np.around(test_data[:, 4]).astype('int')

        # Create path and name for confusion matrix .png
        path_dir, path_file = createFilenamePathAndName()
        mkdir_p(path_dir)

        # Draw confusion matrix
        # Name of the classes, could also be strings
        class_names = np.array([0, 1, 2])
        PlotConfusionMatrixPlt(test_data[:, 4].astype('int'), output,
                               classes=class_names, normalize=True, path=path_file)

        return acc, path_file


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_MainWindow()
    ui.setupUi()
    # ui.show()

    for learn_alg in learning_algorithms:
        SOLVER = learn_alg

        for num_layer in num_layers:

            if num_layer == 1:
                htmp_acc = np.zeros((len(combs_layers1), len(activation_f_vector)))
                # htmp_avgprec = np.zeros((len(combs_layers1), len(activation_f_vector)))
                path_file = ""
                i = 0
                for neurons in combs_layers1:
                    N_PER_LAYER = neurons
                    j = 0
                    for activation_f in activation_f_vector:
                        ACTIVATION_F = activation_f
                        htmp_acc[i, j], path_file = ui.startCombination()
                        accuracy_dict[path_file] = htmp_acc[i, j]
                        # avg_prec_dict[path_file] = htmp_avgprec[i, j]
                        j += 1
                    i += 1

                plotHeatmap(htmp_acc, combs_layers1, getHeatmapPath(num_layer) + "_acc.png")
                # plotHeatmap(htmp_avgprec, combs_layers1, getHeatmapPath(num_layer) + "avgprec.png")

            elif num_layer == 2:
                htmp_acc = np.zeros((len(combs_layers2), len(activation_f_vector)))
                # htmp_avgprec = np.zeros((len(combs_layers2), len(activation_f_vector)))
                path_file = ""
                i = 0
                for neurons in combs_layers2:
                    N_PER_LAYER = neurons
                    j = 0
                    for activation_f in activation_f_vector:
                        ACTIVATION_F = activation_f
                        htmp_acc[i, j], path_file = ui.startCombination()
                        accuracy_dict[path_file] = htmp_acc[i, j]
                        # avg_prec_dict[path_file] = htmp_avgprec[i, j]
                        j += 1
                    i += 1

                plotHeatmap(htmp_acc, combs_layers2, getHeatmapPath(num_layer) + "_acc.png")
                # plotHeatmap(htmp_avgprec, combs_layers2, getHeatmapPath(num_layer) + "avgprec.png")

            elif num_layer == 3:
                htmp_acc = np.zeros((len(combs_layers3), len(activation_f_vector)))
                # htmp_avgprec = np.zeros((len(combs_layers3), len(activation_f_vector)))
                i = 0
                for neurons in combs_layers3:
                    N_PER_LAYER = neurons
                    j = 0
                    for activation_f in activation_f_vector:
                        ACTIVATION_F = activation_f
                        htmp_acc[i, j], path_file = ui.startCombination()
                        accuracy_dict[path_file] = htmp_acc[i, j]
                        # avg_prec_dict[path_file] = htmp_avgprec[i, j]
                        j += 1
                    i += 1

                plotHeatmap(htmp_acc, combs_layers3, getHeatmapPath(num_layer) + "_acc.png")
                # plotHeatmap(htmp_avgprec, combs_layers2, getHeatmapPath(num_layer) + "avgprec.png")

    print("Sorting and writing to files...")
    # logger.write("\n BEST IS: " + combination_best + " WITH MSE = " + str(mse_best))
    sorted_dict_acc = sorted(accuracy_dict.items(), key=operator.itemgetter(1))
    sorted_dict_avgprec = sorted(avg_prec_dict.items(), key=operator.itemgetter(1))

    with open('combs_accuracy.txt', 'w') as f:
        f.write('\n'.join('{}: {}'.format(x[0], x[1]) for x in sorted_dict_acc))
    with open('combs_avg_prec.txt', 'w') as f:
        f.write('\n'.join('{}: {}'.format(x[0], x[1]) for x in sorted_dict_avgprec))

    logger.close()
    print("Done!")
    sys.exit(app.exec_())