# -*- coding: utf-8 -*-

import sys
import os
import random
import math
import xml.etree.ElementTree as ET
import numpy as np
import time

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPainter, QBrush, QPen, QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtChart import QLineSeries, QChart, QValueAxis, QChartView
from PyQt5.QtWidgets import QFileDialog

from deap import base, creator, tools

#Change current working directory to this script path
import pathlib
os.chdir(pathlib.Path(__file__).parent.absolute())

####Global GA parameters####
IND_SIZE = 0 #Size of the individual (number of cities)
NGEN = 5000 #number of generations
POP_SIZE = 100  #population size
MUTPB = 0.02 #probability for mutating an individual
NELT = 4    #number of elite individuals
#########################
populacija = [50, 100, 200, 400]
mutacija = [0.05, 0.1, 0.15, 0.2]
elitni_clanovi = [5, 10, 15, 20]

rezultat = []
rezultat_fitnnes = []
dirname = ""
dirname_granice = ""
iteracija = 0
najbolji_kromoson = ()
najbolji_individualni = []
############



####Other global variables####
stop_evolution = False
q_min_series = QLineSeries()
q_min_series.setName("MIN")
q_max_series = QLineSeries()
q_max_series.setName("MAX")
q_avg_series = QLineSeries()
q_avg_series.setName("AVG")

croatia_map_img = QImage("Croatia620.png")
gradovi_imena = []
sirineGradova = []
duzineGradova = []
borderCheck = False

sirineGranica = []
duzineGranica = []
##############################


# Load the list of cities
tree = ET.parse('gradovi.xml')
root = tree.getroot()
for child in root:
    gradovi_imena.append(str(child.attrib['ime_grada']))
    sirineGradova.append(float(child.attrib['sirina']))
    duzineGradova.append(float(child.attrib['duzina']))

# Load the list of borders
border_tree = ET.parse("granice.xml")
border_root = border_tree.getroot()
for child in border_root:
    sirineGranica.append(float(child.attrib['sirina']))
    duzineGranica.append(float(child.attrib['duzina']))
    
    
# Set the number of cities when they have been parsed
IND_SIZE = len(gradovi_imena)
BORDER_SIZE = len(duzineGranica)


# Other useful global variables #
koordinateGranice = []
koordinateGradova = []
udaljenostGradova = np.zeros((IND_SIZE, IND_SIZE))
matricaPrelaskaGranica = np.zeros((IND_SIZE, IND_SIZE))

def findMedian(arr, n):
    # sorting array

    a = sorted(arr)

    # check if odd amount of numbers in list
    if n % 2 != 0:
        return a[round(n / 2)]
    else:
        return -1
    
def GlobToImgCoords(coord_x, coord_y):
    stupnjevi_1 = math.floor(coord_x)
    minute_1 = round((coord_x - math.floor(coord_x)) * 60)
    stupnjevi_2 = math.floor(coord_y)
    minute_2 = round((coord_y - math.floor(coord_y)) * 60)

    kor_x = 0
    kor_y = 0
    if stupnjevi_2 > 13:
        kor_x = ((stupnjevi_2 - 14) * 60) + (minute_2 + 54)
    else:
        kor_x = minute_2 - 6

    if stupnjevi_1 < 46:
        kor_y = (((46 - (stupnjevi_1 + 1)) * 60) + (48 + (60 - minute_1)))
    else:
        kor_y = (48 - minute_1)

    kor_x = kor_x + math.floor(kor_x * 0.52)
    kor_y = (kor_y * 2) + math.floor(kor_y * 0.12)

    return kor_x, kor_y

for i in range(len(duzineGranica)):
    koordinataX, koordinataY = GlobToImgCoords(sirineGranica[i], duzineGranica[i])
    koordinateGranice.append((koordinataX, koordinataY))

for i in range(len(gradovi_imena)):
    koordinataX, koordinataY = GlobToImgCoords(sirineGradova[i], duzineGradova[i])
    koordinateGradova.append((koordinataX, koordinataY))

def udaljenost(sirina1, duzina1, sirina2, duzina2):
    udaljenostDuzina = (duzina1 - duzina2) * 78.85
    udaljenostSirina = (sirina1 - sirina2) * 110.64
    return math.sqrt(udaljenostSirina ** 2 + udaljenostDuzina ** 2)

for i in range(IND_SIZE):
    for j in range(IND_SIZE):
        udaljenostGradova[i][j] = udaljenost(sirineGradova[i],
                                                        duzineGradova[i],
                                                        sirineGradova[j],
                                                        duzineGradova[j])

def parametriPravca(x1, y1, x2, y2):
    a = (y2 - y1) / (x2 - x1)
    b = -a * x1 + y1
    return a, b

def sjeciste(p1, p2, p3, p4):
    a1, b1 = parametriPravca(p1[0], p1[1], p2[0], p2[1])
    a2, b2 = parametriPravca(p3[0], p3[1], p4[0], p4[1])

    if a1 == a2:
        return False

    x = (b2 - b1) / (a1 - a2)
    y = a2 * x + b2

    if (min(p1[0], p2[0]) <= x <= max(p1[0], p2[0])) \
            and (min(p1[1], p2[1]) <= y <= max(p1[1], p2[1])) \
            and (min(p3[0], p4[0]) <= x <= max(p3[0], p4[0])) \
            and (min(p3[1], p4[1]) <= y <= max(p3[1], p4[1])):
        return True
    return False

def prelazakGranice(koorPrvogGrada, koorDrugogGrada):
    for i in range(-1, len(koordinateGranice) - 1):
        if (max(koorPrvogGrada[0], koorDrugogGrada[0]) < min(koordinateGranice[i][0], koordinateGranice[i + 1][0])) \
                or (max(koorPrvogGrada[1], koorDrugogGrada[1]) < min(koordinateGranice[i][1], koordinateGranice[i + 1][1])) \
                or (min(koorPrvogGrada[0], koorDrugogGrada[0]) > max(koordinateGranice[i][0], koordinateGranice[i + 1][0])) \
                or (min(koorPrvogGrada[1], koorDrugogGrada[1]) > max(koordinateGranice[i][1], koordinateGranice[i + 1][1])):
            continue
        else:
            if sjeciste(koorPrvogGrada, koorDrugogGrada, koordinateGranice[i], koordinateGranice[i + 1]):
                return True
    return False


for k in range(IND_SIZE):
    for l in range(IND_SIZE):
        if k == l:
            matricaPrelaskaGranica[k][l] = False
        else:
            matricaPrelaskaGranica[k][l] = prelazakGranice(koordinateGradova[k], koordinateGradova[l])

def evaluateInd(individual):
    fit_val = 0.0  

    for i in range(len(individual) - 1):
        fit_val += udaljenostGradova[individual[i]][individual[i + 1]]
        if borderCheck and matricaPrelaskaGranica[individual[i]][individual[i + 1]]:
            fit_val += 3000

    return fit_val, 

def generateWorldImage(individual):
    img = QImage(620, 600, QImage.Format_ARGB32)
    img.fill(Qt.transparent)

    painter = QPainter(img)
    g_first = individual[0]
    g_last = individual[IND_SIZE - 1]
    x1, y1 = GlobToImgCoords(sirineGradova[g_first], duzineGradova[g_first])
    x2, y2 = GlobToImgCoords(sirineGradova[g_last], duzineGradova[g_last])
    painter.setBrush(Qt.green)
    painter.drawEllipse(x1 - 10, y1 - 10, 15, 15)
    painter.setBrush(Qt.blue)
    painter.drawEllipse(x2 - 10, y2 - 10, 15, 15)

    painter.setPen(QPen(Qt.black, 2, Qt.DashLine))
    for i in range(IND_SIZE - 1):
        x1, y1 = GlobToImgCoords(sirineGradova[individual[i]], duzineGradova[individual[i]])
        x2, y2 = GlobToImgCoords(sirineGradova[individual[i + 1]], duzineGradova[individual[i + 1]])
        painter.drawLine(x1, y1, x2, y2)

    painter.setPen(QPen(Qt.red, 2, Qt.DotLine))
    for i in range(len(sirineGranica)):
        if i == (len(sirineGranica) - 1):
            x1, y1 = GlobToImgCoords(sirineGranica[i], duzineGranica[i])
            x2, y2 = GlobToImgCoords(sirineGranica[0], duzineGranica[0])
        else:
            x1, y1 = GlobToImgCoords(sirineGranica[i], duzineGranica[i])
            x2, y2 = GlobToImgCoords(sirineGranica[i + 1], duzineGranica[i + 1])
        painter.drawLine(x1, y1, x2, y2)

    painter.end()

    return img

class MyQFrame(QtWidgets.QFrame):
    def paintEvent(self, event):
        painterWorld = QPainter(self)
        painterWorld.drawPixmap(self.rect(), self.img)
        painterWorld.end()
        
def btnSaveChartSeries_Click():
    global q_min_series
    global q_max_series
    global q_avg_series
    filename, _ = QFileDialog.getSaveFileName(None, "Save series to text file", "", "Text Files (*.txt, *.csv)")
    with open(filename, 'w') as dat:
        for i in range(q_min_series.count()):
            dat.write('%f,%f,%f\n' % (q_min_series.at(i).y(), q_avg_series.at(i).y(), q_max_series.at(i).y()))
    print("Chart series saved to: ", filename)


def saveToCSV(rezultat, rezultat_fitnnes):
    filename = ""
    if dirname == "populacija":
        filename = dirname + "_" + str(POP_SIZE)
    elif dirname == "mutacija":
        filename = dirname + "_" + str(MUTPB)
    elif dirname == "elitni":
        filename = dirname + "_" + str(NELT)

    try:
        os.makedirs(dirname_granice + "/" + dirname)
    except:
        print("Directory already exists.")

    path_file = dirname_granice + "/" + dirname + "/" + filename + ".csv"
    index_of_median = rezultat.index(findMedian(rezultat, len(rezultat)))
    with open(path_file, 'w') as dat:
        for i in range(rezultat_fitnnes[index_of_median].count()):
            dat.write('%f\n' % (rezultat_fitnnes[index_of_median].at(i).y()))
    print("Chart series saved to: ", path_file)
        

class Ui_MainWindow(QtWidgets.QMainWindow):
    def setupUi(self):
        self.setObjectName("MainWindow")
        self.resize(850, 1080)
        self.setWindowTitle("GA - Queens")
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.frameWorld = MyQFrame(self.centralwidget)
        self.frameWorld.img = QPixmap(1000, 1000)
        self.frameWorld.setGeometry(QtCore.QRect(10, 10, 620, 600))
        self.frameWorld.setFrameShape(QtWidgets.QFrame.Box)
        self.frameWorld.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frameWorld.setObjectName("frameWorld")
        self.frameChart = QChartView(self.centralwidget)
        self.frameChart.setGeometry(QtCore.QRect(10, 620, 620, 400))
        self.frameChart.setFrameShape(QtWidgets.QFrame.Box)
        self.frameChart.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frameChart.setRenderHint(QPainter.Antialiasing)
        self.frameChart.setObjectName("frameChart")
        self.gaParams = QtWidgets.QGroupBox(self.centralwidget)
        self.gaParams.setGeometry(QtCore.QRect(650, 10, 161, 145))
        self.gaParams.setObjectName("gaParams")
        self.gaParams.setTitle("GA parameters")
        self.label1 = QtWidgets.QLabel(self.gaParams)
        self.label1.setGeometry(QtCore.QRect(10, 20, 61, 16))
        self.label1.setObjectName("label1")
        self.label1.setText("Population:")
        self.label2 = QtWidgets.QLabel(self.gaParams)
        self.label2.setGeometry(QtCore.QRect(10, 50, 47, 16))
        self.label2.setObjectName("label2")
        self.label2.setText("Mutation:")
        self.label3 = QtWidgets.QLabel(self.gaParams)
        self.label3.setGeometry(QtCore.QRect(10, 80, 81, 16))
        self.label3.setObjectName("label3")
        self.label3.setText("Elite members:")
        self.label4 = QtWidgets.QLabel(self.gaParams)
        self.label4.setGeometry(QtCore.QRect(10, 110, 91, 16))
        self.label4.setObjectName("label4")
        self.label4.setText("No. generations:")
        self.tbxPopulation = QtWidgets.QLineEdit(self.gaParams)
        self.tbxPopulation.setGeometry(QtCore.QRect(100, 20, 51, 20))
        self.tbxPopulation.setObjectName("tbxPopulation")
        self.tbxMutation = QtWidgets.QLineEdit(self.gaParams)
        self.tbxMutation.setGeometry(QtCore.QRect(100, 50, 51, 20))
        self.tbxMutation.setObjectName("tbxMutation")
        self.tbxElite = QtWidgets.QLineEdit(self.gaParams)
        self.tbxElite.setGeometry(QtCore.QRect(100, 80, 51, 20))
        self.tbxElite.setObjectName("tbxElite")
        self.tbxGenerations = QtWidgets.QLineEdit(self.gaParams)
        self.tbxGenerations.setGeometry(QtCore.QRect(100, 110, 51, 20))
        self.tbxGenerations.setObjectName("tbxGenerations")
        self.cbxNoVis = QtWidgets.QCheckBox(self.centralwidget)
        self.cbxNoVis.setGeometry(QtCore.QRect(650, 170, 170, 17))
        self.cbxNoVis.setObjectName("cbxNoVis")
        self.cbxNoVis.setText("No visualization per generation")
        self.cbxBorder = QtWidgets.QCheckBox(self.centralwidget)
        self.cbxBorder.setGeometry(QtCore.QRect(650, 200, 100, 17))
        self.cbxBorder.setObjectName("cbxBorder")
        self.cbxBorder.setText("Border patrol")
        self.btnStart = QtWidgets.QPushButton(self.centralwidget)
        self.btnStart.setGeometry(QtCore.QRect(650, 230, 75, 23))
        self.btnStart.setObjectName("btnStart")
        self.btnStart.setText("Start")
        self.btnStop = QtWidgets.QPushButton(self.centralwidget)
        self.btnStop.setEnabled(False)
        self.btnStop.setGeometry(QtCore.QRect(730, 230, 75, 23))
        self.btnStop.setObjectName("btnStop")
        self.btnStop.setText("Stop")
        self.btnSaveWorld = QtWidgets.QPushButton(self.centralwidget)
        self.btnSaveWorld.setGeometry(QtCore.QRect(650, 570, 121, 41))
        self.btnSaveWorld.setObjectName("btnSaveWorld")
        self.btnSaveWorld.setText("Save world as image")
        self.btnSaveChart = QtWidgets.QPushButton(self.centralwidget)
        self.btnSaveChart.setGeometry(QtCore.QRect(650, 930, 121, 41))
        self.btnSaveChart.setObjectName("btnSaveChart")
        self.btnSaveChart.setText("Save chart as image")
        self.btnSaveChartSeries = QtWidgets.QPushButton(self.centralwidget)
        self.btnSaveChartSeries.setGeometry(QtCore.QRect(650, 980, 121, 41))
        self.btnSaveChartSeries.setObjectName("btnSaveChartSeries")
        self.btnSaveChartSeries.setText("Save chart as series")
        self.setCentralWidget(self.centralwidget)
        QtCore.QMetaObject.connectSlotsByName(self)

        # Connect events
        self.btnStart.clicked.connect(self.btnStart_Click)
        self.btnStop.clicked.connect(self.btnStop_Click)
        self.btnSaveWorld.clicked.connect(self.btnSaveWorld_Click)
        self.btnSaveChart.clicked.connect(self.btnSaveChart_CLick)
        self.btnSaveChartSeries.clicked.connect(btnSaveChartSeries_Click)

        # Set default GA variables
        self.tbxGenerations.insert(str(NGEN))
        self.tbxPopulation.insert(str(POP_SIZE))
        self.tbxMutation.insert(str(MUTPB))
        self.tbxElite.insert(str(NELT))

        self.new_image = QPixmap(1000, 1000)

    def btnStart_Click(self):
        # Set global variables
        global stop_evolution
        global q_min_series
        global q_max_series
        global q_avg_series
        stop_evolution = False
        q_min_series.clear()
        q_max_series.clear()
        q_avg_series.clear()

        # Set global variables from information on UI
        global NGEN
        global POP_SIZE
        global MUTPB
        global NELT
        global iteracija

        # NGEN = int(self.tbxGenerations.text())
        # POP_SIZE = int(self.tbxPopulation.text())
        # MUTPB = float(self.tbxMutation.text())
        # NELT = int(self.tbxElite.text())
        global borderCheck
        borderCheck = self.cbxBorder.isChecked()

        iteracija += 1

        # Loading Croatia map
        self.img = QPixmap(620, 600)
        self.img.load('Croatia620.png')
        self.frameWorld.img = self.img
        # Drawing towns
        painter = QPainter(self.img)
        painter.setPen(QPen(Qt.black, 10, Qt.SolidLine))
        painter.setFont(QFont('Arial', 12))
        for i in range(len(gradovi_imena)):
            x, y = GlobToImgCoords(sirineGradova[i], duzineGradova[i])
            painter.drawPoint(x, y)
            painter.drawText(x + 5, y + 5, gradovi_imena[i])

        painter.end()
        # Redrawing frames
        self.frameWorld.repaint()
        app.processEvents()

        # ####Initialize deap GA objects####

        # Make creator that minimize. If it would be 1.0 instead od -1.0 than it would be maxmize
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

        # Create an individual (a blueprint for cromosomes) as a list with a specified fitness type
        creator.create("Individual", list, fitness=creator.FitnessMin)

        # Create base toolbox for finishing creation of a individual (cromosome)
        self.toolbox = base.Toolbox()

        # This is if we want a permutation coding of genes in the cromosome
        self.toolbox.register("indices", random.sample, range(IND_SIZE), IND_SIZE)

        # initIterate requires that the generator of genes (such as random.sample) generates an iterable (a list)
        # variable
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.indices)

        # Create a population of individuals (cromosomes). The population is then created by
        # toolbox.population(n=300) where 'n' is the number of cromosomes in population
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Register evaluation function
        self.toolbox.register("evaluate", evaluateInd)

        # Register what genetic operators to use
        self.toolbox.register("mate", tools.cxUniformPartialyMatched,
                              indpb=0.2)  # Use uniform recombination for permutation coding

        # Permutation coding
        self.toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)

        self.toolbox.register("select", tools.selTournament, tournsize=3)  # Use tournament selection

        ##################################

        # Generate initial poplation. Will be a member variable so we can easely pass everything to new thread
        self.pop = self.toolbox.population(n=POP_SIZE)

        # Evaluate initial population, we map() the evaluation function to every individual and then assign their
        # respective fitness, map runs evaluate function for each individual in pop
        fitnesses = list(map(self.toolbox.evaluate, self.pop))
        for ind, fit in zip(self.pop, fitnesses):
            ind.fitness.values = fit  # Assign calcualted fitness value to individuals

        # Extracting all the fitnesses of all individuals in a population so we can monitor and evovlve the algorithm
        # until it reaches 0 or max number of generation is reached
        self.fits = [ind.fitness.values[0] for ind in self.pop]

        # Disable start and enable stop
        self.btnStart.setEnabled(False)
        self.btnStop.setEnabled(True)
        self.gaParams.setEnabled(False)
        self.cbxBorder.setEnabled(False)
        self.cbxNoVis.setEnabled(False)

        # Start evolution
        self.evolve()
        
    
    def btnStop_Click(self):
        global stop_evolution
        stop_evolution = True
        # Disable stop and enable start
        self.btnStop.setEnabled(False)
        self.btnStart.setEnabled(True)
        self.gaParams.setEnabled(True)
        self.cbxBorder.setEnabled(True)
        self.cbxNoVis.setEnabled(True)
    
    # Function for GA evolution
    def evolve(self):
        global q_min_series
        global q_max_series
        global q_avg_series
        global najbolji_kromoson

        # Variable for keeping track of the number of generations
        curr_g = 0

        # Begin the evolution till goal is reached or max number of generation is reached
        while min(self.fits) != 0 and curr_g < NGEN:
            # Check if evolution and thread need to stop
            if stop_evolution:
                break  # Break the evolution loop

            # A new generation
            curr_g = curr_g + 1
            # print("-- Generation %i --" % curr_g)

            # Select the next generation individuals Select POP_SIZE - NELT number of individuals. Since
            # recombination is between neigbours, not two naighbours should be the clone of the same individual
            offspring = [self.toolbox.select(self.pop, 1)[0]]
            for i in range(POP_SIZE - NELT - 1):  # -1 because the first seleceted individual is already added
                while True:
                    new_o = self.toolbox.select(self.pop, 1)[0]
                    if new_o != offspring[len(
                            offspring) - 1]:  # if it is different than the last inserted then add to offspring and
                        # break
                        offspring.append(new_o)
                        break

            # Clone the selected individuals because all of the changes are inplace
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover on the selected offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                self.toolbox.mate(child1, child2)  # inplace recombination
                # Invalidate new children fitness values
                del child1.fitness.values
                del child2.fitness.values

            # Apply mutation on the offspring
            for mutant in offspring:
                if random.random() < MUTPB:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Add elite individuals #Is clonning needed?
            offspring.extend(list(map(self.toolbox.clone, tools.selBest(self.pop, NELT))))

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # print("  Evaluated %i individuals" % len(invalid_ind))

            # Replace population with offspring
            self.pop[:] = offspring

            # Gather all the fitnesses in one list and print the stats
            self.fits = [ind.fitness.values[0] for ind in self.pop]

            # length = len(self.pop)
            # mean = sum(self.fits) / length
            # sum2 = sum(x * x for x in self.fits)
            # std = abs(sum2 / length - mean ** 2) ** 0.5

            q_min_series.append(curr_g, min(self.fits))
            # q_max_series.append(curr_g, max(self.fits))
            # q_avg_series.append(curr_g, mean)

            # print("  Min %s" % q_min_series.at(q_min_series.count() - 1).y())
            # print("  Max %s" % q_max_series.at(q_max_series.count() - 1).y())
            # print("  Avg %s" % mean)
            # print("  Std %s" % std)

            if self.cbxNoVis.isChecked():
                app.processEvents()
            else:
                self.chart = QChart()
                self.chart.addSeries(q_min_series)
                self.chart.addSeries(q_max_series)
                self.chart.addSeries(q_avg_series)
                self.chart.setTitle("Fitness value over time")
                self.chart.setAnimationOptions(QChart.NoAnimation)
                self.chart.createDefaultAxes()
                self.frameChart.setChart(self.chart)

                # Draw queen positions of best individual on a image
                best_ind = tools.selBest(self.pop, 1)[0]
                self.updateWorldFrame(generateWorldImage(best_ind))

        # Printing best individual
        best_ind = tools.selBest(self.pop, 1)[0]
        
        # if best_ind.fitness.values[0] < best_chromosome.fitness.values[0]:
        #     best_chromosome = best_ind
        print("Finished iteration: {}".format(iteracija))

        najbolji_individualni.append(best_ind)
        rezultat.append(best_ind.fitness.values[0])
        rezultat_fitnnes.append(q_min_series)
        # if len(results) == 5:
        #     saveToCSV(results, results_fitnesses)
        #     results_fitnesses.clear()
        #     results.clear()

        # Visulaize final solution
        # if self.cbxNoVis.isChecked():
        #     self.chart = QChart()
        #     self.chart.addSeries(q_min_series)
        #     self.chart.addSeries(q_max_series)
        #     self.chart.addSeries(q_avg_series)
        #     self.chart.setTitle("Fitness value over time")
        #     self.chart.setAnimationOptions(QChart.NoAnimation)
        #     self.chart.createDefaultAxes()
        #     self.frameChart.setChart(self.chart)
        #
        #     # Draw queen positions of best individual on a image
        #     best_ind = tools.selBest(self.pop, 1)[0]
        #     self.updateWorldFrame(generateWorldImage(best_ind))

        # Disable stop and enable start
        self.btnStop.setEnabled(False)
        self.btnStart.setEnabled(True)
        self.gaParams.setEnabled(True)
        self.cbxBorder.setEnabled(True)
        self.cbxNoVis.setEnabled(True)
        
    def updateWorldFrame(self, best_individual_img):
        # new_image = QPixmap(1000,1000)
        self.new_image.fill()  # White color is default
        painter = QPainter(self.new_image)
        # First draw the map with towns
        painter.drawPixmap(self.new_image.rect(), self.img)
        # Then draw the best individual
        painter.drawImage(self.new_image.rect(), best_individual_img)
        painter.end()
        # Set new image to the frame
        self.frameWorld.img = self.new_image
        # Redrawing frames
        self.frameWorld.repaint()
        self.frameChart.repaint()
        app.processEvents()
    
    def btnSaveWorld_Click(self, filename_world):
        self.frameWorld.img.save(filename_world, "PNG")
        print("World image saved to: ", filename_world)

    def btnSaveChart_CLick(self):
        p = self.frameChart.grab()
        filename, _ = QFileDialog.getSaveFileName(None, "Save series chart as a image", "", "Image Files (*.png)")
        p.save(filename, "PNG")
        print("Chart series image saved to: ", filename)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_MainWindow()
    ui.setupUi()
    ui.cbxNoVis.setChecked(True)

    for bool_var in [False, True]:
        ui.cbxBorder.setChecked(bool_var)

        if ui.cbxBorder.isChecked():
            file_log = open("logger_border.txt", 'a')
            file_log.write("\n### BORDER CHECK ###\n")
            dirname_granice = "border_check"


        else:
            file_log = open("logger_no_border.txt", 'a')
            file_log.write("\n### NO BORDER CHECK ###\n")
            dirname_granice = "no_border_check"

        print("# Changing population #")
        file_log.write("# Changing population #\n")
        for pop_from_vec in populacija:
            iteracija = 0
            POP_SIZE = pop_from_vec
            dirname = "pop"
            rezultat.clear()
            rezultat_fitnnes.clear()
            najbolji_individualni.clear()

            print("Finding for config => pop: {}, mut: {}, el: {}"
                  .format(POP_SIZE, MUTPB, NELT))
            file_log.write("Finding for config => pop: {}, mut: {}, el: {}\n"
                           .format(POP_SIZE, MUTPB, NELT))

            while iteracija < 5:
                ui.btnStart_Click()

            file_log.write("Results: {} -- median: {}\n".format(rezultat, findMedian(rezultat, len(rezultat))))


            saveToCSV(rezultat, rezultat_fitnnes)
            print("Best individual is %s, %s" % (najbolji_individualni[rezultat.index(min(rezultat))],
                                                 najbolji_individualni[rezultat.index(min(rezultat))].fitness.values))

            print("Median individual is %s, %s" % (najbolji_individualni[rezultat.index(findMedian(rezultat, len(rezultat)))],
                                                   najbolji_individualni[rezultat.index(
                                                       findMedian(rezultat, len(rezultat)))].fitness.values))
            file_log.write("Best individual is %s, %s\n" % (najbolji_individualni[rezultat.index(min(rezultat))],
                                                            najbolji_individualni[
                                                                rezultat.index(min(rezultat))].fitness.values))
            file_log.write(
                "Median individual is %s, %s\n" % (najbolji_individualni[rezultat.index(findMedian(rezultat, len(rezultat)))],
                                                   najbolji_individualni[rezultat.index(
                                                       findMedian(rezultat, len(rezultat)))].fitness.values))
            filename = dirname_granice + "/" + dirname + "/" + dirname + "_" + str(POP_SIZE) + ".png"
            ui.updateWorldFrame(generateWorldImage(najbolji_individualni[rezultat.index(min(rezultat))]))
            ui.btnSaveWorld_Click(filename)
        POP_SIZE = 100

        print("# Changing mutation #")
        file_log.write("# Changing mutation #\n")
        for mut_from_vec in mutacija:
            iteracija = 0
            MUTPB = mut_from_vec
            dirname = "mut"
            rezultat.clear()
            rezultat_fitnnes.clear()
            najbolji_individualni.clear()

            print("Finding for config => pop: {}, mut: {}, el: {}"
                  .format(POP_SIZE, MUTPB, NELT))
            file_log.write("Finding for config => pop: {}, mut: {}, el: {}\n"
                           .format(POP_SIZE, MUTPB, NELT))

            while iteracija < 5:
                ui.btnStart_Click()

            file_log.write("Results: {} -- median: {}\n".format(rezultat, findMedian(rezultat, len(rezultat))))


            saveToCSV(rezultat, rezultat_fitnnes)
            print("Resuts: {}".format(rezultat))

            print("Best individual is %s, %s" % (najbolji_individualni[rezultat.index(min(rezultat))],
                                                 najbolji_individualni[rezultat.index(min(rezultat))].fitness.values))

            print("Median individual is %s, %s" % (najbolji_individualni[rezultat.index(findMedian(rezultat, len(rezultat)))],
                                                   najbolji_individualni[rezultat.index(
                                                       findMedian(rezultat, len(rezultat)))].fitness.values))
            file_log.write("Best individual is %s, %s\n" % (najbolji_individualni[rezultat.index(min(rezultat))],
                                                            najbolji_individualni[
                                                                rezultat.index(min(rezultat))].fitness.values))
            file_log.write(
                "Median individual is %s, %s\n" % (najbolji_individualni[rezultat.index(findMedian(rezultat, len(rezultat)))],
                                                   najbolji_individualni[rezultat.index(
                                                       findMedian(rezultat, len(rezultat)))].fitness.values))

            ui.updateWorldFrame(generateWorldImage(najbolji_individualni[rezultat.index(min(rezultat))]))
            filename = dirname_granice + "/" + dirname + "/" + dirname + "_" + str(MUTPB) + ".png"
            ui.btnSaveWorld_Click(filename)

        MUTPB = 0.05

        print("# Changing elitism #")
        file_log.write("# Changing elitism #\n")
        for el_from_vec in elitni_clanovi:
            iteracija = 0
            NELT = el_from_vec
            dirname = "el"
            rezultat.clear()
            rezultat_fitnnes.clear()
            najbolji_individualni.clear()

            print("Finding for config => pop: {}, mut: {}, el: {}"
                  .format(POP_SIZE, MUTPB, NELT))
            file_log.write("Finding for config => pop: {}, mut: {}, el: {}\n"
                           .format(POP_SIZE, MUTPB, NELT))

            while iteracija < 5:
                ui.btnStart_Click()

            file_log.write("Results: {} -- median: {}\n".format(rezultat, findMedian(rezultat, len(rezultat))))


            saveToCSV(rezultat, rezultat_fitnnes)

            print("Resuts: {}".format(rezultat))
            print("Best individual is %s, %s" % (najbolji_individualni[rezultat.index(min(rezultat))],
                                                 najbolji_individualni[rezultat.index(min(rezultat))].fitness.values))

            print("Median individual is %s, %s" % (najbolji_individualni[rezultat.index(findMedian(rezultat, len(rezultat)))],
                                                   najbolji_individualni[rezultat.index(
                                                       findMedian(rezultat, len(rezultat)))].fitness.values))
            file_log.write("Best individual is %s, %s\n" % (najbolji_individualni[rezultat.index(min(rezultat))],
                                                            najbolji_individualni[
                                                                rezultat.index(min(rezultat))].fitness.values))
            file_log.write(
                "Median individual is %s, %s\n" % (najbolji_individualni[rezultat.index(findMedian(rezultat, len(rezultat)))],
                                                   najbolji_individualni[rezultat.index(
                                                       findMedian(rezultat, len(rezultat)))].fitness.values))

            ui.updateWorldFrame(generateWorldImage(najbolji_individualni[rezultat.index(min(rezultat))]))
            filename = dirname_granice + "/" + dirname + "/" + dirname + "_" + str(NELT) + ".png"
            ui.btnSaveWorld_Click(filename)
            
        NELT = 5
        file_log.close()

    # ui.show()
    sys.exit(app.exec_())