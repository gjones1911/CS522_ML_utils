import numpy as np
from math import *
from scipy.integrate import quad
import matplotlib.pyplot as plt


class G_solver:
    solver_type='quadratic'


    def __init__(self, solver_type=None):
        solver_type=solver_type

    # calculate the mean or mu
    def get_mu(self, arry, dtype=np.float32):
        return np.mean(arry, dtype=dtype)

    # calculate the std deviation or sigma in
    # symbolic terms
    def get_sig(self, arry, dtype=np.float32):
        return np.std(arry, dtype=dtype)

    # get the variance
    def get_var(self, arry, dtype=np.float32):
        return (np.std(arry, dtype=dtype))**2

    # calculate the gaussian value fo the given x value
    # with given mean and variance
    def Nx_gaussian(self, x, mu, sig):
        return ((1 / (sig * sqrt(2 * np.pi)))) * exp((-(x - mu) ** 2) / (2 * sig ** 2))


    def get_box_dims(self, xl, xm, lines=True, vden=50, hden=20, fancy=False):
        # get the length of the box
        length = xm - xl
        # calculate the height
        height = 1 / length
        if not lines:
            return length, height

        if not fancy:
            ll = [[xl, xl],[0,height]]
            tl = [[xl, xm],[height,height]]
            rl = [[xm, xm],[0,height]]
            return height, length, ll, tl, rl


        # generate components of the right line of box
        rly = np.linspace(0, height, int(height*vden)).tolist()
        rlx = [xm] * len(rly)
        rbound = [rlx, rly]

        # generate components of the top line of box
        tlx = np.linspace(xl, xm, int(length*hden)).tolist()
        tly = [height] * len(tlx)
        tbound = [tlx, tly]

        # generate components of the left line of box
        lly = np.linspace(0, height, int(height*vden)).tolist()
        llx = [xl] * len(lly)
        lbound = [llx, lly]

        return height, length, rbound, tbound, lbound


    def plot_box(self, ll, tl, rl, figure_num=None, showit=False, c='b--'):
        print('left line:\n', ll)
        print('top line:\n', tl)
        print('right line:\n', rl)
        print('',)
        print('',)
        if figure_num is not None:
            plt.figure(figure_num)
            plt.plot(ll[0], ll[1], c=c)
            plt.plot(tl[0], tl[1], c=c)
            plt.plot(rl[0], rl[1], c=c)
        if showit:
            plt.show()

    # will produce the probability of attribute value x belonging to
    # class my_class for a box probability density function
    def Box_pdf(self, xl, xm, xval, my_class='my_class', verbose=False):
        # get the length of the box
        length = xm - xl
        #calculate the height
        height = 1/length
        if xl <= xval <= xm:
            return height
        return 0


    def generate_gaussian(self, xarray, mu, sig):
        return [self.Nx_gaussian(x, mu, sig) for x in xarray]


    def calculate_a_b_c(self, mu1, sig1, mu2, sig2, verbose=False):
        sig1_sqr = sig1**2
        sig2_sqr = sig2**2
        a = ((1/(2*sig2_sqr) ) - (1/(2*sig1_sqr)))
        b = ((2*mu1/(2*sig1_sqr)) - (2*mu2/(2*sig2_sqr)))
        c = ((mu2**2/(2*sig2_sqr)) - (mu1**2/(2*sig1_sqr)) + (np.log(1/sig1)) - (np.log(1/sig2)))

        print('X^2*{:f} + X{:f}  + {:f} = 0'.format(a,b,c))

        return a, b, c


    def calculate_determinate2(self, a, b, c):
        return (b**2) - (4*a*c)


    def poly_solns2(self, a, b, d, dtype=np.float64):
        s1 = (-b-np.sqrt(d))/(2*a)
        s2 = (-b+np.sqrt(d))/(2*a)
        return   s1, s2

    def solve_poly(self, a, b, c):
        if a == 0:
            print('apparently a non poly')
            return -c/b, -c/b
        d = self.calculate_determinate2(a,b,c)
        return self.poly_solns2(a,b,d)

    # Nx_gaussian(self, x, mu, sig):


    def density_x_prior(self, x, Nc, prior):
        pdf = self.Nx_gaussian(x, Nc[0], Nc[1])
        return pdf * prior

    def calculate_Z(self, x, Ncs, priors, verbose=False):
        tosum = [self.density_x_prior(x, Nc, prior)  for Nc, prior in zip(Ncs, priors)]
        if verbose:
            print('to sum is:')
            print(tosum)
        return np.sum(tosum)



    def posterior_prob(self, x, Nc, prior, Z):

        return self.density_x_prior(x, Nc, prior)/Z


    def generate_posteriori_probs(self, xarray, Ncs, priors):
        # go ahead and calculate the z's you will need
        zs =  [self.calculate_Z(x, Ncs, priors) for x in xarray]

        # get an array of the posteriori
        rl = list()

        for p, Nc in zip(priors, Ncs):
            rl.append([self.posterior_prob(x, Nc, prior=p, Z=z)   for x,z in zip(xarray, zs)])
        return rl


    def integrate_func(self, X, Y, x0, x1, prior):
        summation = 0
        cnt = 0
        start_i, end_i = 0, 0
        for x in X:
            if x >= x0:
                start_i = cnt
            if x >= x1:
                end_i = cnt
            break
            cnt += 1

        for idx in range(start_i, end_i+1):
            rect_area = abs(X[idx+1] - X[idx]) * Y[idx]
            tri_area = .5 * abs((Y[idx+1]- Y[idx])) * abs((X[idx+1]-X[idx]))
            summation += rect_area + tri_area
        return summation * prior

