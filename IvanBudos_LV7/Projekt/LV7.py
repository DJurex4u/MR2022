import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np
import matplotlib.pyplot as plt

#Ulazna varijabla
Greska_temp = ctrl.Antecedent(np.linspace(0, 30, 100), 'Greska_temp')
#Izlazna varijabla
Grijac = ctrl.Consequent(np.linspace(0, 10, 100), 'Grijac', defuzzify_method='centroid')

#trokutasta funkcija
Greska_temp['Niska'] = fuzz.trimf(Greska_temp.universe, [0, 0, 15]) #Trokut, lomi se u 3 točke
Greska_temp['Srednja'] = fuzz.trimf(Greska_temp.universe, [0, 15, 30]) #Trokut, lomi se u 3 točke
Greska_temp['Visoka'] = fuzz.trimf(Greska_temp.universe, [15, 30, 30]) #Trokut, lomi se u 3 točke

Grijac['Slabo'] = fuzz.trimf(Grijac.universe, [0, 0, 5]) #Gaussova funkcija definirana s mi i sigma
Grijac['Srednje'] = fuzz.trimf(Grijac.universe, [0, 5, 10]) #Gaussova funkcija definirana s mi i sigma
Grijac['Jako'] = fuzz.trimf(Grijac.universe, [5, 10, 10]) #Gaussova funkcija definirana s mi i sigma

"""
#trapezasta funkcija
Greska_temp['Niska'] = fuzz.trapmf(Greska_temp.universe, [0, 0, 5, 15]) #Trapez, lomi se u 4 točke
Greska_temp['Srednja'] = fuzz.trapmf(Greska_temp.universe, [0, 10, 20, 30]) #Trapez, lomi se u 4 točke
Greska_temp['Visoka'] = fuzz.trapmf(Greska_temp.universe, [15, 25, 30, 30]) #Trapez, lomi se u 4 točke

Grijac['Slabo'] = fuzz.trapmf(Grijac.universe, [0, 0, 2.5, 5]) #Gaussova funkcija definirana s mi i sigma
Grijac['Srednje'] = fuzz.trapmf(Grijac.universe, [0, 2.5, 7.5, 10]) #Gaussova funkcija definirana s mi i sigma
Grijac['Jako'] = fuzz.trapmf(Grijac.universe, [5, 7.5, 10, 10]) #Gaussova funkcija definirana s mi i sigma
"""
"""
#gaussova funkcija
Greska_temp['Niska'] = fuzz.gaussmf(Greska_temp.universe, 0, 7.5) #Trokut, lomi se u 3 točke
Greska_temp['Srednja'] = fuzz.gaussmf(Greska_temp.universe, 15, 7.5) #Trokut, lomi se u 3 točke
Greska_temp['Visoka'] = fuzz.gaussmf(Greska_temp.universe, 30, 7.5) #Trokut, lomi se u 3 točke

Grijac['Slabo'] = fuzz.gaussmf(Grijac.universe, 0, 2.5) #Gaussova funkcija definirana s mi i sigma
Grijac['Srednje'] = fuzz.gaussmf(Grijac.universe, 5, 2.5) #Gaussova funkcija definirana s mi i sigma
Grijac['Jako'] = fuzz.gaussmf(Grijac.universe, 10, 2.5) #Gaussova funkcija definirana s mi i sigma
"""


pravilo1=ctrl.Rule(Greska_temp['Niska'],Grijac['Slabo'])
pravilo2=ctrl.Rule(Greska_temp['Srednja'],Grijac['Srednje'])
pravilo3=ctrl.Rule(Greska_temp['Visoka'],Grijac['Jako'])


regulator=ctrl.ControlSystem([pravilo1,pravilo2,pravilo3])
regulator_simulacija=ctrl.ControlSystemSimulation(regulator)


y=[]
x=np.linspace(0,30,num=100)

for i in x:
    regulator_simulacija.input['Greska_temp'] = i
    regulator_simulacija.compute()
    y.append(regulator_simulacija.output['Grijac'])

figure=plt.figure(1)
plt.plot(x,y)
plt.show()