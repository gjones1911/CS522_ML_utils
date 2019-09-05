import numpy as np
from scipy.integrate import quad
from math import sqrt, exp
import matplotlib.pyplot as plt
from GML_basic.G_math import G_solver



gs = G_solver()

classes = {0:'Class 1',
           1:'Class 2',
           2:'Class 3'}

#xvals = np.linspace(-4, 15, num=18, endpoint=True).tolist()
xvals = np.linspace(-4, 15, num=100, endpoint=True).tolist()
# xvals

N1a2 = gs.generate_gaussian(xvals, 4, 2)
N2a2 = gs.generate_gaussian(xvals, 6, 3)
N3a2 = gs.generate_gaussian(xvals, 5, 2)

print(len(N1a2))
print(len(N2a2))
print(len(N3a2))
#N1 = np.array(N1a2) * .33
#N2 = np.array(N2a2) * .33
#N3 = np.array(N3a2) * .33
#print(len(N1a2))
#print(len(N2a2))
#print(len(N3a2))

print('\t\tN1\n')
#print(N1)
print('\t\tN2\n')
#print(N2)
print('\t\tN3\n')
#print(N3)

fig_num = 1

plt.figure(fig_num)
plt.plot(xvals, N1a2, 'b--')
plt.plot(xvals, N2a2, 'r--')
plt.plot(xvals, N3a2, 'g--')
fig_num += 1

# integrate

prob_error_N3 = gs.integrate_func(xvals, N3a2, -4, 4.5, .33)
print('The probability of error for N1 = N3 region is {:f}'.format(prob_error_N3))

linebreak = '---------------------------------------------------'


xval = [4.7]
priors = [.33, .33, .33]
N1 = gs.generate_gaussian(xval, 4, 2)
N2 = gs.generate_gaussian(xval, 6, 3)
N3 = gs.generate_gaussian(xval, 5, 2)

# N1 = np.array(N1) * .33
# N2 = np.array(N2) * .33
# N3 = np.array(N3) * .33
N1 = np.array(N1)
N2 = np.array(N2)
N3 = np.array(N3)
print(linebreak)
print(linebreak)
print('problem 2a the first time')
print('N1: {:.4f}'.format(N1[0]))
print('N2: {:.4f}'.format(N2[0]))
print('N3: {:.4f}'.format(N3[0]))

vls = [N1[0], N2[0], N3[0]]
mx_idx = vls.index(max(vls))
print('This should be classified as {:s}'.format(classes[mx_idx]))

print(linebreak)
print(linebreak)

print('Checking valuesusing my calculate abc func')
print('using my calculate abc func')
a,b,c = gs.calculate_a_b_c(4,2,5,2, verbose=True)
print('a: {:f}, b: {:f}, c: {:f}'.format(a,b,c))
solnsx = gs.solve_poly(a, b, c)
print(linebreak)
print('My calculated solution is ')
print('x1: {:f}\nx2: {:f}'.format(solnsx[0], solnsx[1]))
print(linebreak)
print(linebreak)

print(linebreak)
xval = [9/2]

N1 = gs.generate_gaussian(xval, 4, 2)
N2 = gs.generate_gaussian(xval, 6, 3)
N3 = gs.generate_gaussian(xval, 5, 2)

print('N1: {:.4f}'.format(N1[0]))
print('N2: {:.4f}'.format(N2[0]))
print('N3: {:.4f}'.format(N3[0]))

print(linebreak)
xval = [6.89790614306]

N1 = gs.generate_gaussian(xval, 4, 2)
N2 = gs.generate_gaussian(xval, 6, 3)
N3 = gs.generate_gaussian(xval, 5, 2)



print('N1: {:.4f}'.format(N1[0]))
print('N2: {:.4f}'.format(N2[0]))
print('N3: {:.4f}'.format(N3[0]))

print(linebreak)
print(linebreak)

a,b,c = gs.calculate_a_b_c(6,3,5,2, verbose=True)
print('a: {:f}, b: {:f}, c: {:f}'.format(a,b,c))
solnsx = gs.solve_poly(a, b, c)

print('x1: {:f}\nx2: {:f}'.format(solnsx[0], solnsx[1]))
print(linebreak)
xval = [solnsx[1]]

N1 = gs.generate_gaussian(xval, 4, 2)
N2 = gs.generate_gaussian(xval, 6, 3)
N3 = gs.generate_gaussian(xval, 5, 2)
print('N1: {:.4f}'.format(N1[0]))
print('N2: {:.4f}'.format(N2[0]))
print('N3: {:.4f}'.format(N3[0]))
print(linebreak)
xval = [solnsx[0]]

N1 = gs.generate_gaussian(xval, 4, 2)
N2 = gs.generate_gaussian(xval, 6, 3)
N3 = gs.generate_gaussian(xval, 5, 2)
print('N1: {:.4f}'.format(N1[0]))
print('N2: {:.4f}'.format(N2[0]))
print('N3: {:.4f}'.format(N3[0]))
print(linebreak)
print(linebreak)
print(linebreak)
x = [4.7]
Ncs = [[4,2],
       [6,3],
       [5,2]]

priors = [.6, .2, .2]
N1 = gs.generate_gaussian(x, 4, 2)
N2 = gs.generate_gaussian(x, 6, 3)
N3 = gs.generate_gaussian(x, 5, 2)

N1 = np.array(N1) * priors[0]
N2 = np.array(N2) * priors[1]
N3 = np.array(N3) * priors[2]
print(linebreak)
print(linebreak)
print('Problem 2a again')
print('N1: {:.4f}'.format(N1[0]))
print('N2: {:.4f}'.format(N2[0]))
print('N3: {:.4f}'.format(N3[0]))

vls = [N1[0], N2[0], N3[0]]
mx_idx = vls.index(max(vls))
print('This should be classified as {:s}'.format(classes[mx_idx]))
print(linebreak)
print(linebreak)

#Z = gs.calculate_Z(x, Ncs, priors, verbose=True)

print()
xvals = np.linspace(-4, 15, num=100, endpoint=True).tolist()
Ns = gs.generate_posteriori_probs(xvals, Ncs, priors)
plt.figure(fig_num)
plt.plot(xvals, Ns[0], 'b--', label='Class 1')
plt.plot(xvals, Ns[1], 'r--', label='Class 2')
plt.plot(xvals, Ns[2], 'g--', label='Class 3')
plt.title('Posteriori Probability')
plt.legend()
#print(Z)
fig_num += 1

# #####################################################################
# #####################################################################
# #####################################################################
#                   Problem 2 of homework 1
# #####################################################################
# #####################################################################
print(linebreak)
print(linebreak)
print(linebreak)
print(linebreak)
print(linebreak)
xl_neg = 0
xm_neg = 1


xl_pos = .95
xm_pos = 3.95

h_neg, l_neg, ll_neg, tl_neg, rl_neg = gs.get_box_dims(xl_neg, xm_neg, lines=True, vden=50, hden=20, fancy=True)


h_pos, l_pos, ll_pos, tl_pos, rl_pos = gs.get_box_dims(xl_pos, xm_pos, lines=True, vden=50, hden=20, fancy=True)

gs.plot_box(ll_neg, tl_neg, rl_neg, figure_num=fig_num, showit=False, c='r')
gs.plot_box(ll_pos, tl_pos, rl_pos, figure_num=fig_num, showit=False, c='b')
# #############################################################
# #############################################################
# ##########    Lets do some integration    ###################
# #############################################################
# #############################################################

err_N1 = quad(gs.Nx_gaussian, 4.5, 6.897906, args=(4,2,1/3, True))
print('quad', err_N1)
print(linebreak)
print(linebreak)
print(linebreak)
print(linebreak)
print(linebreak)
print(linebreak)
print(linebreak)
print(linebreak)
err_N1 = (quad(gs.Nx_gaussian, 4.5, 6.897906, args=(4,2, .333))[0] + quad(gs.Nx_gaussian, 6.897906, np.inf, args=(4,2,1/3))[0] +quad(gs.Nx_gaussian, -np.inf, 1.502, args=(4,2,1/3))[0])
err_N2 = (quad(gs.Nx_gaussian, 4.5, 6.897906, args=(6,3))[0] + quad(gs.Nx_gaussian, 1.502, 4.5, args=(6,3))[0])*1/3
err_N2a = (quad(gs.Nx_gaussian, 4.5, 6.897906, args=(6,3,1/3))[0] + quad(gs.Nx_gaussian, 1.502, 4.5, args=(6,3,1/3))[0])
err_N3 = (quad(gs.Nx_gaussian, 1.502, 4.5, args=(5,2))[0] + quad(gs.Nx_gaussian, 6.897906, np.inf, args=(5,2))[0] +quad(gs.Nx_gaussian, -np.inf, 1.502, args=(5,2))[0])*1/3
should_one = quad(gs.Nx_gaussian, -np.inf, np.inf, args=(4,2))[0]
print('The probability of error for N1: {:f}'.format(err_N1))
print('The probability of error for N2: {:f}'.format(err_N2))
print('The probability of error for N2a: {:f}'.format(err_N2a))
print('The probability of error for N3: {:f}'.format(err_N3))
print('The overall probability of error: {:f}'.format(err_N1 + err_N2 + err_N3))
print('hopefully one: {:f}'.format(should_one))
plt.show()
