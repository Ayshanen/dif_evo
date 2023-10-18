import dipole_base as dp
import config as conf
import numpy as np
import scipy.linalg as scp
from cmath import *
from scipy.optimize import differential_evolution, OptimizeResult
import time
import matplotlib.pyplot as plt
import matplotlib as mplt
import multiprocessing
import os

mplt.rcParams['figure.figsize'] = [10, 10]

iteration_file = open('input.txt')
iteration = int(iteration_file.read())
start = iteration + 9
finish = iteration + 100


N = conf.N
#xmin = conf.xmin
xmin = 0.01 * iteration
xmax = xmin + 1
x = [i * 0.01 for i in range(start, finish)]
dimensions = conf.dimensions
minimum = conf.minimum
symmetry = conf.symmetry
n = conf.n
center = conf.center

if n > 1:
    N_evo =  int((N-(n+1))/n + 0) if center else int((N-n)/n + 0)
elif symmetry == 'mirror':
   if dimensions == 2:
        N_evo = N - conf.dipoles_on_axis
   elif dimensions == 1:
       if center == 0:
           N_evo = (N - 2) // 2
       else:
           N_evo = (N - 3) // 2
else:
     N_evo = N - 2
if conf.dipoles_on_axis != 0:    
    output_folder_name = 'results_external/' + str(N) + '_' + str(dimensions) + 'D_' + symmetry + str(conf.dipoles_on_axis) + '/'
else:
    output_folder_name = 'results_external/' + str(N) + '_' + str(dimensions) + 'D_' + symmetry + '/'

print(output_folder_name)
if __name__ == '__main__':

    if not os.path.isdir(output_folder_name):
        os.mkdir(output_folder_name)
    if not os.path.exists(output_folder_name + 'gamma_global.txt'):

        gamma_file = open(output_folder_name + 'gamma_global.txt', 'w')
        for i in [i for i in np.linspace(0.1, 1, 100)]:
            string = str(round(i, 2)) + '          ' + '1000000000000000000'
            print(string)
            gamma_file.write(string + "\n")
            gamma_file.write("")
        gamma_file.close()
            

def evolution_base_func(x1):  # оптимизируемая функция. Возвращает в функцию эволюции значение минимума или большое число, если решение не подходит.
    # работает с полярными координатами.
    # return sin(x[0]).real * cos(x[1]).real                                       # можно проверить эволюцию на более простой функции от нескольких переменных.
    A = []
    gam = []
    om = []

    if dimensions == 1 and symmetry == 'mirror':
      if center == 0:
          a = [[0.5, 0], [-0.5, 0]]                                                          # строим массив точек
      else:
          a = [[-1, 0], [0, 0], [1, 0]]
      for i in range(len(x1)):
        a.append([x1[i], 0])
        a.append([-1*x1[i], 0])
      figure_plt = dp.polar_to_decartes(a)
      
      flag = True
      for i in range(len(figure_plt)):
        for j in range(len(figure_plt)):                                                     # проверяем, удовлетворяет ли решение условиям по расстояниям между диполями. Если нет, возвращаем большое число.
          if (dp.dist(figure_plt[i], figure_plt[j]) < 0.99) and i != j:
            flag = False

            break
      
      if not flag:
        return 10 ** 18

      for s in x:
        A.append( dp.matrix([[s * figure_plt[j][0], s * figure_plt[j][1]] for j in range(np.shape(figure_plt)[0])]) )

      min_gamma = 10 ** 6

      if minimum == 'global':
        w = []
        for i in range(np.shape(A)[0]):
          a, v = scp.eig(A[i])
          w.append(a)
        for j in range(len(w)):
          gam.append(w[j].imag * -2)
        for i in range(np.shape(gam)[0]):
          for j in gam[i]:
            if j < min_gamma:
              min_gamma = j
      if min_gamma <= 0:
        return 10 ** 18
      return min_gamma
      
    elif dimensions == 1:
        a = [[0, 0], [1, 0]]                                                          # строим массив точек
        for i in range(len(x1)):
          a.append([x1[i], 0])
    
        figure_plt = dp.polar_to_decartes(a)
        flag = True
        for i in range(len(figure_plt)):
          for j in range(len(figure_plt)):                                                     # проверяем, удовлетворяет ли решение условиям по расстояниям между диполями. Если нет, возвращаем большое число.
            if (dp.dist(figure_plt[i], figure_plt[j]) < 0.99) and i != j:
              flag = False
              break
        
        if not flag:
          return 10 ** 18

        for s in x:
          A.append( dp.matrix([[s * figure_plt[j][0], s * figure_plt[j][1]] for j in range(np.shape(figure_plt)[0])]) )

        min_gamma = 10 ** 6

        if minimum == 'global':
          w = []
          for i in range(np.shape(A)[0]):
            a, v = scp.eig(A[i])
            w.append(a)
          for j in range(len(w)):
            gam.append(w[j].imag * -2)
          for i in range(np.shape(gam)[0]):
            for j in gam[i]:
              if j < min_gamma:
                min_gamma = j
        if min_gamma <= 0:
          return 10 ** 18
        return min_gamma

    elif dimensions == 2:
        if n > 1:
            a = dp.D_symmetry(x1, conf.n)
            figure_plt = dp.polar_to_decartes(a)
            flag = True
            for i in range(len(figure_plt)):
                for j in range(len(figure_plt)):                                                     # проверяем, удовлетворяет ли решение условиям по расстояниям между диполями. Если нет, возвращаем большое число.
                    if (dp.dist(figure_plt[i], figure_plt[j]) < 0.99) and i != j:
                        flag = False
                        #print(i, j, a)
                        break
            if not flag:
                return 10 ** 9
            for s in x:
                A.append( dp.matrix([[s * figure_plt[j][0], s * figure_plt[j][1]] for j in range(np.shape(figure_plt)[0])]) )
            min_gamma = 10 ** 6
            w = []
            for i in range(np.shape(A)[0]):
                a, v = scp.eig(A[i])
                w.append(a)
            for j in range(len(w)):
                gam.append(w[j].imag * -2)
            for i in range(np.shape(gam)[0]):
                for j in gam[i]:
                    if j < min_gamma:
                        min_gamma = j
            if min_gamma <= 0:
                min_gamma = 10 ** 18
            return min_gamma
      
      
        elif symmetry == 'mirror':
            a = []
            
            for i in range(conf.dipoles_on_axis - 1, len(x1), 2):
                a.append([x1[i], x1[i + 1]])

            for i in range(len(a)):
                a.append([-a[i][0], -a[i][1]])



            figure_plt = dp.polar_to_decartes(a)
            if conf.dipoles_on_axis > 0:
                figure_plt.append([0, 0])
                for i in range(conf.dipoles_on_axis - 1):
                    figure_plt.append([0, x1[i]])
            #print(figure_plt)
            flag = True
            for i in range(len(figure_plt)):
                for j in range(len(figure_plt)):                                                     # проверяем, удовлетворяет ли решение условиям по расстояниям между диполями. Если нет, возвращаем большое число.
                    if (dp.dist(figure_plt[i], figure_plt[j]) < 0.99) and i != j:
                        flag = False
                        break
            if not flag:
                return 10 ** 9
            for s in x:
                A.append( dp.matrix([[s * figure_plt[j][0], s * figure_plt[j][1]] for j in range(np.shape(figure_plt)[0])]) )
            min_gamma = 10 ** 6
            w = []
            for i in range(np.shape(A)[0]):
                a, v = scp.eig(A[i])
                w.append(a)
            for j in range(len(w)):
                gam.append(w[j].imag * -2)
            for i in range(np.shape(gam)[0]):
                for j in gam[i]:
                    if j < min_gamma:
                        min_gamma = j
            if min_gamma <= 0:
                min_gamma = 10 ** 18
            return min_gamma



        else:
            a = [[0, 0], [1, 0]]
            for i in range(0, len(x1), 2):
                a.append([x1[i], x1[i + 1]])
            figure_plt = dp.polar_to_decartes(a)
            flag = True
            for i in range(len(figure_plt)):
                for j in range(len(figure_plt)):                                                     # проверяем, удовлетворяет ли решение условиям по расстояниям между диполями. Если нет, возвращаем большое число.
                    if (dp.dist(figure_plt[i], figure_plt[j]) < 0.99) and i != j:
                        flag = False
                        break

            if not flag:
                return 10 ** 9
            for s in x:
                A.append( dp.matrix([[s * figure_plt[j][0], s * figure_plt[j][1]] for j in range(np.shape(figure_plt)[0])]) )
            min_gamma = 10 ** 6
            w = []
            for i in range(np.shape(A)[0]):
                a, v = scp.eig(A[i])
                w.append(a)
            for j in range(len(w)):
                gam.append(w[j].imag * -2)
            for i in range(np.shape(gam)[0]):
                for j in gam[i]:
                    if j < min_gamma:
                        min_gamma = j
            if min_gamma <= 0:
                min_gamma = 10 ** 18
            return min_gamma

if __name__ == '__main__':
    
    bounds = []
    if dimensions == 1:
        if symmetry == 'none':
            bounds = [(-N - 1,  N) for _ in range(N_evo)]
        elif symmetry == 'mirror':
            if center == 0:
                bounds = [(1.5, N) for _ in range(N_evo)]
            else:
                bounds = [(2, N) for _ in range(N_evo)]

    elif dimensions == 2:
        bounds = []
        if symmetry != 'mirror':
            for _ in range(N_evo):
                bounds.append((1, 5))#bounds.append(((1 / xmin) * 0.1, 5))
                bounds.append((-pi, pi))
        elif symmetry == 'mirror':
            for _ in range(conf.dipoles_on_axis - 1):
               bounds.append((1, 5))
            for _ in range(N_evo // 2):
                bounds.append((1, 5))#bounds.append(((1 / xmin) * 0.1, 5))
                bounds.append((-pi, pi))
            
    
    print(N_evo)




    print(bounds)
    start_evo = time.time()
    if len(bounds) != 0:
      result = differential_evolution(evolution_base_func, bounds, disp=True, popsize = conf.iteration_precision, 
                                    workers = -1, updating = 'deferred', maxiter = 100000, tol = 0.01)  # <------ Вот здесь творится магия!
      end_evo = time.time()
      print(end_evo - start_evo)
      result_coord = []
      
      
      if symmetry == 'none' and dimensions == 2:
            result_coord = [[0, 0], [1, 0]]
            for i in range(0, np.shape(result.x)[0], 2):
                result_coord.append([result.x[i], result.x[i + 1]])
            figure_plt = dp.polar_to_decartes(result_coord)
      elif symmetry == 'mirror' and dimensions == 1:
          if center == 0:
              result_coord = [[-0.5, 0], [0.5, 0]]
          else:
              result_coord = [[-1, 0], [0, 0], [1, 0]]
          for i in range(len(result.x)):
              result_coord.append([result.x[i], 0])
              result_coord.append([-result.x[i], 0])
          figure_plt = dp.polar_to_decartes(result_coord)
      
      elif symmetry == 'mirror' and dimensions == 2:
         result_coord = []

         for i in range(conf.dipoles_on_axis - 1, np.shape(result.x)[0], 2):
            result_coord.append([result.x[i], result.x[i + 1]])
            result_coord.append([-result.x[i], -result.x[i + 1]])
         
         figure_plt = dp.polar_to_decartes(result_coord)
         if conf.dipoles_on_axis > 0:
             figure_plt.append([0, 0])
         for i in range(conf.dipoles_on_axis - 1):
             figure_plt.append([0, result.x[i]])


      elif dimensions == 1:
          result_coord = [[0, 0], [1, 0]]
          for i in range(len(result.x)):
              result_coord.append([result.x[i], 0])
              if symmetry == 'mirror':
                  result_coord.append([-result.x[i], 0])
          figure_plt = dp.polar_to_decartes(result_coord)
      elif dimensions == 2:
          if n > 1:
            result_coord = dp.D_symmetry(result.x, conf.n)
            figure_plt = dp.polar_to_decartes(result_coord)
      print(figure_plt)
      
      
    else:
        if n <= 6:
            result_coord = dp.construct_figure_radius(1, n, center)
        else:
            result_coord = dp.construct_figure_side(1, n, center)
        result = OptimizeResult(x = result_coord)
        figure_plt = result_coord
        result.fun = dp.compute_configuration(result_coord, x)
        
    end_evo = time.time()
    evo_time = end_evo - start_evo

    
    gamma = result.fun

    file = open(output_folder_name + 'gamma_global.txt', 'r')
    lines = file.readlines()
    xmins = []
    gammas = []
    
    for i in lines:
        a = i.split()
        xmins.append(float(a[0]))
        gammas.append(float(a[1]))
    file.close()
    gamma_index = iteration - 1
    
    print(gamma)
    print(gammas[gamma_index])
    if gamma <= gammas[gamma_index]:
        text_file_name = output_folder_name + 'output_' + str(round(xmins[gamma_index], 4)) + '.txt'


        output_filename = output_folder_name + 'module_' + str(round(xmins[gamma_index], 4))

        gammas[gamma_index] = gamma
        
        
    
        file = open(output_folder_name + 'gamma_global.txt', 'w')

        for i in range(len(xmins)):
            file.write(str(xmins[i]) + '          ' + str(gammas[i]) + '\n')
            file.write('')

        file.close()

    

        A = []
        gam = []
        om = []
        y_plt = []
        x_plt = []
        xmin_plot = x[0]
        xmax_plot = xmin_plot + 1
        n_plot = 1000

        x_plot = [s for s in np.linspace(xmin_plot, xmax_plot, 101)]
        for s in x_plot:
            A.append(dp.matrix([[s * figure_plt[j][0], s * figure_plt[j][1]] for j in
                            range(np.shape(figure_plt)[0])]))  # Матрица для eigenshuffle.

        w, v = dp.eigenshuffle(A)

        for j in range(len(w)):
            gam.append(w[j].imag * -2)  # Массив Г\Г0
            om.append(w[j].real + 1)  # Массив омег.


        f = 0
        mins = [x[0], gam[0][f], v[0][f], f]  # Ищет минимальную точку локального минимума по всем модам.
        min_gamma = 100
        min_x = 0
        min_vectors = 0

        if minimum == 'global':
            for i in range(np.shape(gam)[0]):
                for j in range(len(gam[i])):
                    if gam[i][j] < min_gamma:
                        min_gamma = gam[i][j]
                        min_x = x_plot[i]
                        min_vectors = v[i][j]
                        mins = [min_x, min_gamma, min_vectors]

        elif minimum == 'local':
            mins = dp.find_sub_min(gam, v, x_plot)



        figure, axis = plt.subplots(1, 2, figsize=(10, 5))
        figure.set_figheight(7.25)
        figure.set_figwidth(17.75)
        axis[0].set_aspect('equal', adjustable='box')
        axis[1].set_aspect('equal', adjustable='box')
        
        axis[0].title.set_text('Модуль')

        axis[1].title.set_text('Фаза')
        axis[1].grid(linewidth=1)
        axis[0].set_xlabel('x / λ')
        axis[1].set_xlabel('x / λ')
        axis[0].set_ylabel('y / λ')
        axis[1].set_ylabel('y / λ')
        # norm2 = mplt.colors.Normalize(vmin=-pi, vmax=0)

        x_plt = []  # массивы х и у точек для графика
        y_plt = []

        module_data = []  # массивы модуля и фазы точек для графика
        phase_data = []

        # figure_plt = construct_figure_radius(r, N, dot)                                #строим фигуру для графика по той же схеме, что и для графика расстояний

        for j in range(np.shape(figure_plt)[0]):  # заносим х, у, модуль и фазу каждой точки в соответственные массивы
            x_plt.append(round(figure_plt[j][0] * mins[0], 6))
            y_plt.append(round(figure_plt[j][1] * mins[0], 6))

            temp = abs(round(phase(mins[2][j]), 6))
            if temp < 0:
                temp *= -1
            phase_data.append(abs(round(phase(mins[2][j]), 6)))

            module_data.append(sqrt(mins[2][j].real ** 2 + mins[2][j].imag ** 2).real)

        

        module_plt1 = axis[0].scatter(x_plt, y_plt, c=module_data, s=500, norm=None)
        phase_plt1 = axis[1].scatter(x_plt, y_plt, c=phase_data, s=500, norm=None)
        axis[0].grid(True, which = 'major')
        axis[1].grid(True, which = 'minor')
        cbar1 = plt.colorbar(module_plt1, ax=axis[0], fraction=0.05)

        cbar2 = plt.colorbar(phase_plt1, ax=axis[1], fraction=0.05)

        pre_ticklabels = ['π']
        for i in range(1, 20):
            pre_ticklabels.append('π / ' + str(2 ** i))
        pre_ticks = [pi - 0.04]
        for i in range(1, 20):
            pre_ticks.append(pi / 2 ** i)

        for i in range(len(pre_ticks) - 1):
            if pre_ticks[i] < max(phase_data):
                phase_bar = [0, pre_ticks[i + 1], pre_ticks[i]]
                cbar2.set_ticks(phase_bar)
                cbar2.set_ticklabels([0, pre_ticklabels[i + 1], pre_ticklabels[i]])
                break
        
        axis[0].set_xlim(min(min(x_plt), min(y_plt)) - 0.03, max(max(x_plt), max(y_plt)) + 0.03)
        axis[0].set_ylim(min(min(x_plt), min(y_plt)) - 0.03, max(max(x_plt), max(y_plt)) + 0.03)
        axis[1].set_xlim(min(min(x_plt), min(y_plt)) - 0.03, max(max(x_plt), max(y_plt)) + 0.03)
        axis[1].set_ylim(min(min(x_plt), min(y_plt)) - 0.03, max(max(x_plt), max(y_plt)) + 0.03)


        plt.tight_layout()
        figure.savefig(output_filename + '.png')
        if __name__ == '__main__':
            text_output = open(text_file_name, 'w')
            for i in range(len(x_plt)):
                
                text_output.write(str(x_plt[i]) + ' ' * (20 - len(str(x_plt[i]))) + str(y_plt[i]) + ' ' * (20 - len(str(y_plt[i]))) + str(module_data[i]) + 
                                ' ' * (25 - len(str(module_data[i]))) + str(phase_data[i]) + '\n')
            text_output.close()