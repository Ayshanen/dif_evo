import dipole_base as dp
import config as conf
import numpy as np
import scipy.linalg as scp
from cmath import *
from scipy.optimize import differential_evolution
import time
import matplotlib.pyplot as plt
import matplotlib as mplt
import multiprocessing
import os
import mirrored_copy as m_c

font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 32}
mplt.rc('font', **font)
#cm = mplt.colormaps['bwr']

folder_name = m_c.output_folder_name

#f = open(folder_name + '/gamma_global.txt', 'r')

f = open(folder_name + 'gamma_global.txt', 'r')

c = f.readlines()
c = [i for i in c if i != '\n']
a = []
b = []

for i in c:
    temp = i.split()
    a.append(float(temp[0]))
    b.append(float(temp[1]))


f.close()
j = [0.01 * i for i in range(1, 101)]


x_population = plt.figure()
ax = x_population.add_subplot()
x_population.subplots_adjust(top=0.85)
ax.semilogy(a[:30:], b[:30:], 'bo')
ax.grid()
ax.set_xlabel('rmin')
ax.set_ylabel('Г/Г0')
#x_population.savefig(folder_name + 'stairs.png')

x_population.savefig(folder_name + 'stairs.png')