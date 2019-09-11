import numpy as np
from scipy.integrate import quad
from math import sqrt, exp
import matplotlib.pyplot as plt
from GML_basic.G_math import G_solver
from GML_basic.G_math import g_gauss
import multiprocessing as mp

print(mp.cpu_count())

f_name = r'K:\FA_19\CS_522\data_sets\Project_1\SYNTH_tr.txt'
f_name_test = r'K:\FA_19\CS_522\data_sets\Project_1\SYNTH_te.txt'

gg = g_gauss(class_label='yc', file_name=f_name, file_name_test=f_name_test, process_d=True)


