from keras import layers
from keras import models
from keras.datasets import mnist
from keras import utils
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix,plot_confusion_matrix
import numpy as np
import sklearn.metrics as metrics
import os
import errno
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import SVC

activation_functions = ['relu', 'tanh', 'sigmoid']
kernel_sizes = [(3, 3), (5, 5), (7, 7)]
logger1 = open("CNN_model1.txt", 'w')
logger2 = open("CNN_model2.txt", 'w')

(x_train, y_train), (x_test, y_test1) = mnist.load_data()
x_train = x_train.reshape(60000, 28, 28, 1)
x_train = x_train.astype('float32') / 255
x_test = x_test.reshape(10000, 28, 28, 1)
x_test = x_test.astype('float32') / 255
y_train = utils.to_categorical(y_train, num_classes=10)
y_test = utils.to_categorical(y_test1, num_classes=10)
classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


#PRVI
def first_CNN(activation_f='relu', kernel_size=(3, 3)):
    model = models.Sequential()

    model.add(layers.Conv2D(64, kernel_size, activation=activation_f, input_shape=(28, 28, 1), strides=(1, 1)))

    model.add(layers.Conv2D(32, kernel_size, activation=activation_f, strides=(1, 1)))

    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    return model

#DRUGI
def second_CNN(activation_f='relu', kernel_size=(3, 3)):
    model = models.Sequential()

    model.add(layers.Conv2D(64, kernel_size, activation=activation_f, input_shape=(28, 28, 1), strides=(1, 1)))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(layers.Conv2D(32, kernel_size, activation=activation_f, strides=(1, 1)))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    return model


# Create path if it doesn't exist
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

# Define path to model conf matrix and the model's name
def createModelPathAndName(model=1, activ_f='relu', kernel_size=(3, 3)):
    path_dir = "CNN/model" + str(model) + '/'
    path_file = path_dir + activ_f + "_kernelsize"
    for kern in kernel_size:
        path_file += str(kern) + "x"
    path_file = path_file[:-1]
    return path_dir, path_file

def ConfusionMatrix(confusionMatrixdf, path):
    plt.figure()
    sns.heatmap(confusionMatrixdf, annot=True, cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    title_temp = path.split('/')[-1].split('_')
    title = "Act fnc: " + title_temp[0] + " -- Kernel size: " + title_temp[1]
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path + ".png")
    plt.close()

def saveAccuracyHeatmap(modelnum, data, path, act_fs, kernel_sizes):
    plt.figure()
    sns.heatmap(data, annot=True, xticklabels=act_fs, yticklabels=kernel_sizes)
    plt.tight_layout()
    plt.title("Model " + str(modelnum))
    plt.savefig(path + ".png")
    plt.close()

model1HeatmapData = np.zeros((len(kernel_sizes), len(activation_functions)))
i = 0
for act_f in activation_functions:
    j = 0
    for kernel in kernel_sizes:
        first_model = first_CNN(activation_f=act_f, kernel_size=kernel)
        first_model.summary()
        first_model.summary(print_fn=lambda x: logger1.write(x + '\n'))
        history = first_model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5)
        logger1.write('\n'.join('{}: {}'.format(x[0], x[1]) for x in history.history))
        logger1.write('\n\n\n')
        y_pred = first_model.predict_classes(x_test)
        cm = confusion_matrix(y_test1, y_pred)
        cm_norm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
        cm_norm_df = pd.DataFrame(cm_norm, index=classes, columns=classes)
        path_dir, path_file = createModelPathAndName(model=1, activ_f=act_f, kernel_size=kernel)
        mkdir_p(path_dir)
        ConfusionMatrix(confusionMatrixdf=cm_norm_df, path=path_file)

        model1HeatmapData[i][j] = accuracy_score(y_test1, y_pred)
        j += 1
    i += 1

saveAccuracyHeatmap(1, model1HeatmapData, "CNN/model1/combinations", activation_functions, kernel_sizes)
logger1.close()

model2HeatmapData = np.zeros((len(kernel_sizes), len(activation_functions)))
i = 0
for act_f in activation_functions:
    j = 0
    for kernel in kernel_sizes:
        second_model = second_CNN(activation_f=act_f, kernel_size=kernel)
        second_model.summary(print_fn=lambda x: logger2.write(x + '\n'))
        history = second_model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5)
        logger2.write('\n'.join('{}: {}'.format(x[0], x[1]) for x in history.history))
        logger2.write('\n\n\n')
        y_pred = second_model.predict_classes(x_test)
        cm = confusion_matrix(y_test1, y_pred)
        cm_norm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
        cm_norm_df = pd.DataFrame(cm_norm, index=classes, columns=classes)
        path_dir, path_file = createModelPathAndName(model=1, activ_f=act_f, kernel_size=kernel)
        mkdir_p(path_dir)
        ConfusionMatrix(confusionMatrixdf=cm_norm_df, path=path_file)
        model2HeatmapData[i][j] = accuracy_score(y_test1, y_pred)
        j += 1
    i += 1
logger2.close()


X,Y = fetch_openml('mnist_784',version=1,return_X_y=True)
X=X/255

X_train, X_test=X[:60000],X[60000:]
Y_train, Y_test=Y[:60000],Y[60000:]

mlp = MLPClassifier(hidden_layer_sizes=(100),activation='logistic',solver='adam', alpha=0.0001,max_iter = 100)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=ConvergenceWarning,module='sklearn')
    mlp.fit(X_train, Y_train)

print("Training set score: %f" % mlp.score(X_train,Y_train))
print("Test set score: %f" % mlp.score(X_test,Y_test))
score = mlp.score(X_test,Y_test)

y_pred = mlp.predict(X_test)
cm = metrics.confusion_matrix(Y_test,y_pred)
cm_norm=np.around(cm.astype('float')/cm.sum(axis=1)[:,np.newaxis],decimals=2)
sns.heatmap(cm_norm, annot=True)
plt.show()

