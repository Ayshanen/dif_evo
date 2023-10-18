import numpy as np
import scipy.linalg as scp
import math
from cmath import *
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib as mplt
from matplotlib.ticker import NullLocator
import munkres
from scipy.optimize import differential_evolution
import time
import config as conf

style.use('default')

i_cx = complex(0, 1)

def mat_to_lin(matrix):
  res = []
  for i in range(np.shape(matrix)[0]):
    for j in range(np.shape(matrix)[1]):
      res.append(matrix[i][j])
  return res

def lin_to_mat(mas):
  res = []
  for i in range(0, np.shape(mas)[0] - 1, 2):
    res.append([mas[i], mas[i + 1]])
  return res

def dist(d1, d2):                                                               # Высчитывает расстояние между точками декартовой системы координат.
    return sqrt( (d1[0] - d2[0]) ** 2 + (d1[1] - d2[1]) ** 2 ).real             # Принимает два массива по два элемента: d1[x, y], d2[x, y].

def matrix(figure):                                                         # Новая функция, строящая матрицу в двумерной плоскости.
  mat = []                                                                      # Теперь держится не на расстояниях от первой точки, а на координатах на плоскости.
  for j in range(np.shape(figure)[0]):
    mat.append([])
    for k in range(np.shape(figure)[0]):
      if j == k:
        mat[j].append(-0.5 * i_cx)
      else:
        mat[j].append(green_formula(dist(figure[j], figure[k])) )
  return mat

def construct_figure_radius(r, n, dot = 0):                                     # Создаёт массив из координат точек правильной фигуры на плоскости.
  figure = []                                                                   # Если нужна точка в центре, указать третьим параметром 1.
  base_angle = 2 * pi / n                                                       # Работает на полярных координатах.
  if dot == 1:
    figure.append([0, 0])
  for j in range(0, n):
    figure.append([r.real * cos(base_angle * j).real, r.real * sin(base_angle * j).real])           # Результат выдаёт в декартовых координатах.
  return figure

def construct_figure_side(r, n, dot = 0):                                       #создаёт фигуру исходя из стороны правильной фигуры
  r = r / (2 * sin(pi / n))
  return construct_figure_radius(r, n, dot)

def green_formula(s):                                                           # возвращает значение формулы Грина для фиксированного расстояния между диполями
  kr = s * 2 * pi
  return -3 * pi * exp(i_cx * kr) * (1 + (i_cx / kr) - (1 / kr ** 2)) / (4 * pi * kr)

def V_transpose(V):                                                             #транспонирование массы векторов для eigenshuffle. Теперь вектор не в столбике, а в строчке.
  n, p = len(V), len(V[0])
  Vseq = []
  for i in range(n):
    Vseq.append(V[i].transpose())
  return Vseq

def merged_modes(D):                                                            #ищет номера слившихся мод
  mod = []
  for j in range(conf.N):
    for k in range(j + 1, conf.N + 1):
      if abs(D[0][j] - D[0][k]) < (10 ** -12) and abs(D[1][j] - D[1][k]) < (10 ** -12):
        mod.append([j, k])
  return mod

def polar_to_decartes(coord):
  res = []
  for j in range(np.shape(coord)[0]):
    res.append([coord[j][0] * cos(coord[j][1]).real, coord[j][0] * sin(coord[j][1]).real])
  return res

def line(n):
  return [[j, 0] for j in range(n)]


def eigenshuffle(Asequence):  # Функция работает правильно, но в местами путает значения.

    Ashape = np.shape(Asequence)

    p = Ashape[-1]
    if len(Ashape) < 3:
        n = 1
        Asequence = np.asarray([Asequence], dtype=complex)
    else:
        n = Ashape[0]  # В переменную n загоняется количество матриц.

    Vseq = np.zeros((n, p, p), dtype=complex)  # Создаётся n матриц для векторов, заполненных нулями.
    Dseq = np.zeros((n, p), dtype=complex)  # Создаётся n массивов для значений, заполненных нулями.

    for i in range(n):
        D, V = np.linalg.eig(Asequence[i])  # Берутся значения и вектора очередной матрицы.

        tags = np.argsort(D.real, axis=0)[::-1]  # В массив tags загоняется порядок сортировки значений по реальной части.

        Dseq[i] = D[tags]  # Dseq[i] = D[:, tags] в оригинале. Выводит ошибку, потому что D - не массив.
        Vseq[i] = V[:, tags]  # Вектора сортируются в соответствии с значениями.

        # Следующие строки написаны уже мной и к оригинальной eigenshuffle отношения не имеют.
        # Применяются методы сортировки собственных значений с помощью близости самих значений, векторов и производных во всех вариациях.

        ############################################################################################

        for v in range(
                np.shape(V)[1]):  # Приведение векторов к виду, где первая компонента имеет положительное значение.
            if V[0][v].real < 0:
                for j in range(len(V[v])):
                    V[j][v] *= -1

    ############################################################################################
    m = munkres.Munkres()
    for i in range(-1, n):
        D1 = Dseq[i - 1]
        D2 = Dseq[i]
        V1 = Vseq[i - 1]
        V2 = Vseq[i]

        dist = distancematrix(D1, D2)  # * (1 - np.abs(V1 * np.transpose(V2)))

        reorder = m.compute(np.transpose(dist))
        reorder = [coord[1] for coord in reorder]
        Vs = Vseq[i]
        Vseq[i] = Vseq[i][:, reorder]
        Dseq[i] = Dseq[i, reorder]

    return Dseq, V_transpose(Vseq)


def distancematrix(vec1, vec2):
    v1, v2 = np.meshgrid(vec1, vec2)
    return np.abs(v1 - v2)

def find_sub_min(gam, v):
  local_min = []
  global_min = []
  for j in range(1, len(conf.x) - 1):
    for k in range(np.shape(gam)[1]):
      #print(gam[k])
      if gam[j - 1][k] > gam[j][k] and gam[j][k] < gam[j + 1][k]:                 # численный поиск минимумов
        local_min.append([conf.x[j], gam[j][k], [], k])                                # по каждой моде.
        for g in range(len(v[j][k])):
          local_min[-1][2].append(v[j][k][g])
  global_min = [10 ** 6, 10 ** 6, []]
  for j in range(np.shape(local_min)[0]):
    if global_min[1] > local_min[j][1]:
      global_min = local_min[j]
  return global_min                                                             # В итоге выходит массив, где хранится коэффициент подобия фигуры, Г\Г0, вектора и номер моды.

def find_min(gam, v):
  local_min = []
  global_min = []
  for j in range(1, len(conf.x) - 1):
    for k in range(np.shape(gam)[1]):
      #print(gam[k])
      if gam[j][k] < gam[j + 1][k]:                 # численный поиск минимумов
        local_min.append([conf.x[j], gam[j][k], [], k])                                # по каждой моде.
        for g in range(len(v[j][k])):
          local_min[-1][2].append(v[j][k][g])
  global_min = [10 ** 6, 10 ** 6, []]
  print(np.shape(local_min))
  for j in range(np.shape(local_min)[0]):
    if global_min[1] > local_min[j][1]:
      global_min = local_min[j]
  return global_min


def D_symmetry(x, n): #выстраивает симметричную поворотную конфигурацию с n- порядком.
  if conf.center == 0:
    ans = []
    minr = 1 / (2 * sin(pi / n).real)
  else:
    ans = [[0, 0]]
    minr = 1
  for j in range(n):
    ans.append([minr, j*(2*pi/n)])
  for i in range(0, len(x), 2):
    for j in range(n):
      ans.append([x[i], x[i+1]+j*(2*pi/n)])
  return ans

def compute_configuration(x, x_iter): #высчитывает гамму для фиксированной конфигурации и нижнего расстояния.
  # работает с полярными координатами.
    # return sin(x[0]).real * cos(x[1]).real                                       # можно проверить эволюцию на более простой функции от нескольких переменных.
    A = []
    gam = []
    om = []

    if conf.dimensions == 1:
      a = [[0.5, 0], [-0.5, 0]]                                                          # строим массив точек
      if conf.center == 1:
        a.append([0, 0])
      for i in range(len(x)):
        a.append([x[i], 0])
        a.append([-1*x[i], 0])

      figure_plt = polar_to_decartes(a)
      flag = True
      for i in range(len(figure_plt)):
        for j in range(len(figure_plt)):                                                     # проверяем, удовлетворяет ли решение условиям по расстояниям между диполями. Если нет, возвращаем большое число.
          if (dist(figure_plt[i], figure_plt[j]) < 1) and i != j:
            flag = False
            break
      if not flag:
        return 10 ** 18

      for s in x_iter:
        A.append(matrix([[s * figure_plt[j][0], s * figure_plt[j][1]] for j in range(np.shape(figure_plt)[0])]) )

      min_gamma = 10 ** 6

      if conf.minimum == 'global':
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

    elif conf.dimensions == 2:
      figure_plt = x
      for s in x_iter:
        A.append(matrix([[s * figure_plt[j][0], s * figure_plt[j][1]] for j in range(np.shape(figure_plt)[0])]) )
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