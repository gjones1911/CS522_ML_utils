#
import pandas as pd
import numpy as np
from math import *
from scipy.integrate import quad
import matplotlib.pyplot as plt
import datetime

class G_solver:
    solver_type='quadratic'


    def __init__(self, solver_type=None):
        if solver_type is not None:
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
    def Nx_gaussian(self, x, mu, sig, prior=None, verbose=False):
        if prior is None:
            if verbose:
                print('it is none')
            return ((1 / (sig * sqrt(2 * np.pi)))) * exp((-(x - mu) ** 2) / (2 * sig ** 2))
        return ((1 / (sig * sqrt(2 * np.pi)))) * exp((-(x - mu) ** 2) / (2 * sig ** 2))*prior


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


class classifier():

    class Params():
        def __init__(self, attribs=None, classes=None,class_label=None):
            self.dimensions=None
            self.observations=None
            self.attribs = attribs
            self.classes = classes
            self.class_label = class_label
            if attribs is not None and class_label is None:
                self.dimensions = attribs.shape[1]
            elif attribs is not None:
                self.dimensions = attribs.shape[1] - 1
            if classes is not None:
                self.observations = attribs.shape[1]

        def calculate_priors(self,):
            self.class_types = list(set(self.classes))
            print(self.class_types)
            self.counts = {}
            for c in self.class_types:
                print('c:', c)
                self.counts[c] = self.classes.values.tolist().count(c)
            print(self.counts)
            self.priors = {}

            for cl in self.counts:
                self.priors[cl] = self.counts[cl]/self.observations
            print(self.priors)

        def pull_classes_mean(self, df, classes):
            rd = {}
            usecols = df.columns.values.tolist()
            del usecols[usecols.index(self.class_label)]
            #print(usecols)
            for c in classes:
                rd[c] = df.loc[df[self.class_label] == c].mean()
                #print(rd[c])
                rd[c] = rd[c].loc[usecols]
            return rd

        def pull_classes_cov(self, df, classes):
            rd = {}
            usecols = df.columns.values.tolist()
            # remove the class labels so we can grab
            # everything but the class columns
            del usecols[usecols.index(self.class_label)]
            # print(usecols)
            for c in classes:
                # add an entry to the dictionary
                # that has the covariance for this class' part
                # of the data
                rd[c] = df.loc[df[self.class_label] == c].cov()
                # print(rd[c])
                #rd[c] = rd[c].loc[usecols]
            return rd

        def set_attribs(self, df, class_label='yc', verbose=False, test=False):
            print(df.columns.values)
            self.data_vec = df
            lidx = list(df.columns.values)
            del lidx[lidx.index(class_label)]
            if test:
                return df.loc[:,lidx]

            self.attribs = df.loc[:, lidx]
            if verbose:
                print(self.attribs)

        def set_classes(self, df, class_label='yc', verbose=False, test=False):
            if test:
                return df.loc[:, class_label]
            self.class_label = class_label
            self.classes = df.loc[:, class_label]
            self.class_types = list(set(self.classes.values))
            if verbose:
                print(self.classes)

        def set_stats(self, df=None, verbose=True):
            self.dimensions = self.attribs.shape[1]
            self.observations = self.attribs.shape[0]
            self.mu_array = self.attribs.mean()
            self.std_array = self.attribs.std()
            print('--------------------------------------------------------------------------------std')
            print(self.std_array)
            self.cov = self.attribs.cov()
            print('--------------------------------------------------------------------------------covariance')
            print(self.cov)
            self.shared_cov = (self.cov.values[0][0] + self.cov.values[1][1])/2
            self.det_cov = np.linalg.det(self.cov)
            self.inv_cov = np.linalg.inv(self.cov)
            self.mean_dict = self.pull_classes_mean(self.data_vec, self.class_types)
            self.cov_dict = self.pull_classes_cov(self.data_vec, self.class_types)
            if verbose:
                print('Dimensions: {:d}'.format(int(self.dimensions)))
                print('Observations: {:d}'.format(int(self.observations)))
                print('Means:')
                print(self.mu_array)
                print('\n\n')
                print('Std:\n', self.std_array, '\n\n')
                print('Cov: \n',self.cov,'\n')
                print(self.cov.loc['xs', 'xs'])
                print('Det Cov: \n',self.det_cov,'\n')
                print('Inverse Cov: \n',self.inv_cov,'\n')
                print('Mean dictionary:')
                print(self.mean_dict)

    def __init__(self, attribs=None, classes=None, class_col=None):
        self.params = self.Params(attribs=attribs, classes=classes)

    def show_performance(self, scores, verbose=False):
        true_sum = scores['tp'] + scores['tn']
        false_sum = scores['fp'] + scores['fn']
        sum = true_sum + false_sum
        accuracy = true_sum/sum
        print('False sum:', false_sum)
        if verbose:
            print('===============================================')
            print('===============================================')
            print('             |  predicted pos   |   predicted neg   |')
            print('------------------------------------------------')
            print(' actual pos  |  {:d}          |   {:d}      '.format(scores['tp'], scores['fn']))
            print('------------------------------------------------')
            print(' actual neg  |   {:d}          |   {:d}      '.format(scores['fp'], scores['tn']))
            print('-------------------------------------------------------------------')
            print('                                        Correct |   {:d}'.format(true_sum))
            print('                                          Total | % {:d}'.format(sum))
            print('                                                | ------------------------')
            print('                                       accuracy | {:.2f}'.format(accuracy))
        return accuracy, sum, true_sum, false_sum

    def bi_score(self, g, y, vals, classes, method='accuracy', verbose=True, train=False):
        scores = {'tp':0,
                  'fp':0,
                  'fn':0,
                  'tn':0}
        # go through the guesses and the actual y values scoring
        # * true positives: tp
        # * false positives: fp
        # * true negatives: tn
        # * false negatives: fn
        for gs, ay in zip(g,y):
                # check for negative
                if int(gs) == int(vals[0]):
                    if int(ay) == int(gs):
                        scores['tn'] += 1
                    else:
                        scores['fn'] += 1
                elif int(gs) == int(vals[1]):
                    if int(ay) == int(gs):
                        scores['tp'] += 1
                    else:
                        scores['fp'] += 1
                else:
                    print('Uh Oh spageghtti ohs: {0}'.format(gs))
                    quit()

        # calculate and return the overall accuracy
        if method == 'accuracy':
            accuracy, sum, true_sum, false_sum = self.show_performance(scores=scores, verbose=verbose)
            if train:
                return accuracy, scores
            return accuracy


class bayes_classifier(classifier):
    def __init__(self, attribs=None, classes=None, priors=None, z_val=None, class_col=None):
        classifier.__init__(self, attribs=attribs, classes=classes)
        self.params.priors = priors
        self.params.z_val = z_val
        self.g_slver = G_solver()
        if attribs is not None:
            self.mu_array = self.params.attribs.mean()
            self.std_array = self.params.attribs.std()
            self.cov = self.params.attribs.cov()

    def adjust_priors(self, g, y, prob, eta):
        if int(g) == int(y):
            return
        diff = (y - abs(prob))/1000
        #print('Adjusting the prob {4}, prior of g: {0}, y: {1}, diff: {2}, adjustment {3}'.format(g, y, diff, diff*eta, prob))
        self.params.priors[y] += diff*0
        self.params.priors[y] = min(abs(self.params.priors[y]),.90)
        self.params.priors[(y + 1) % len(self.params.priors)] = 1 - self.params.priors[y]

    def adjust_by_score(self, scores, eta=.01):
        # get number of false types
        fp = scores['fp']
        fn = scores['fn']
        tp = scores['tp']
        tn = scores['tn']
        total = tp + fp + tn + fn
        if fn > fp:
            self.params.priors[0] = self.params.priors[0] - self.params.priors[0] * (fn / total) * eta
            self.params.priors[1] = 1 - self.params.priors[0]
        else:
            self.params.priors[1] = self.params.priors[1] - self.params.priors[1] *(fp/total)*eta
            self.params.priors[0] = 1 - self.params.priors[1]



    # TODO: fix the method it is not done!!!!
    def GMLE(self, xvec, mu, sig, prior, verbose=False):
        ret_a = []
        for x in xvec:
            ret_a.append(self.g_slver.Nx_gaussian(x, mu, sig, prior=prior, verbose=verbose))

    def min_dist_class(self, xvec):
        # go through the classes using the minimum
        # distance classifier
        cnt = 0
        MAP = -999999
        call_it = -9
        # go through the class types
        # using them to index into
        # the various storage items to get
        # the needed value
        for c in self.params.class_types:
            #mean_i = self.params.mean_dict[c]
            # grab the mean for this class
            mean_ib = self.params.mean_dict[c].values
            #print('The mean of ',c)
            #print(mean_i)
            #print(mean_ib)
            sig1 = self.params.cov.values[int(c)][int(c)]
            #print('sig')
            #print(sig1)
            #print('\n\n\n\n\n')
            #print('-----------------------------------------------------')
            #print('-----------------------------------------------------')
            #print('prior of class', c)
            #print(self.params.priors[c])
            #print('\n\n\n\n\n')
            # gi(xrow) = (mean_i-T/sigi^2) * xrow - (mu_i-T * mu_i/2sigi^2)+lnP(cl_i)
            #print(xvec)
            gi = (np.dot(mean_ib, xvec)/sig1) - (np.dot(mean_ib, mean_ib)/(sig1*2)) + np.log(self.params.priors[c])
            #print('g'+str(c))
            #print(gi)
            if gi > MAP:
                MAP = gi
                call_it = c
        #print('it should be classed '+str(call_it))
        return int(call_it), MAP

    def predict(self, xarray):
        g = []
        for xvec in xarray:
            # print(xvec)
            guess, prob = self.min_dist_class(xvec)
            g.append(guess)
        return g

    def fit(self, xarray=None, yarray=None, params=None, step=.497, epochs=500, tx=None, ty=None):
        g = []
        print('=======================================000000000000000000000000')
        print('priors before')
        print(self.params.priors)
        bscr, btscr = -99, -99
        inc = 0
        b_ep = 0
        b_prior = list()
        for ep in range(epochs):
            print('33333333333333333333333333333333333333333333333333333333')
            print('33333333333333333333333333333333333333333333333333333333')
            print('33333333333333333333333333333333333333333333333333333333')
            print('33333333333333333333333333333333333333333333333333333333')
            print('33333333333333333333333333333333333333333333333333333333')
            f = float(ep)
            if True:
                xarray, yarray = self.params.attribs.values, self.params.classes
                params = self.params
                for xvec, y in zip(xarray, yarray):
                    #print(xvec)
                    guess, prob = self.min_dist_class(xvec)
                    g.append(guess)
                    #print('guess', g[0], ', ', y)
                    #print('priors before')
                    #print(self.params.priors)
                    #self.adjust_priors(g[-1],y,prob, step/(epochs**(e+1)))
                    #self.adjust_priors(g[-1],y,prob, step*1**(ep))
                    #print('priors after')
                    #print(self.params.priors)

                print('The priors: 0-{:.2f}, 1-{:.2f}'.format(self.params.priors[0], self.params.priors[1]))
                print(self.params.priors)
                scr, scores = self.bi_score(g, yarray,  [0,1], ['Postitive', 'Negative'], train=True)
                print(scr)
                #self.adjust_by_score(scores, .1*10**-ep)
                #inc = .5*ep%2
                inc = min(ep, 2)
                #self.adjust_by_score(scores, 1/10**inc)
                self.adjust_by_score(scores, 1/10**inc)
                tg = self.predict(tx.values)
                tscr = self.bi_score(tg, ty.values, [0,1], ['pos', 'neg'])
                print(tscr)
                print(self.params.priors)
                print('Epoch #{:d}'.format(ep+1))
                print('=========================================================================================')
                print('=========================================================================================')
                print('=========================================================================================')
                print()
                print()
                if tscr > btscr:
                    btscr = tscr
                    b_prior = list([self.params.priors[0], self.params.priors[1]])
                    b_ep = ep
        print('The best score was {:.3f} bp {:.4f} {:.4f}, best epoch {:d}'.format(btscr, b_prior[0], b_prior[1], b_ep))
        return

class g_gauss(G_solver):

    def vectorize_line(self, line, to_strip=None, split_by=' '):
            line.strip('\n')
            if to_strip is None:
                line = line.strip()
            else:
                line = line.strip(to_strip)
            rv = []
            l_vec = line.split(split_by)
            for i in l_vec:
                i = i.strip()
                if ' ' not in i and i != '':
                    rv.append(i)
            return rv

    def vectorize_lines(self, lines):
        rv = [self.vectorize_line(line) for line in lines]
        return rv

    def create_dictionary(self, d_array, class_label='yc'):
        # get labels
        label_array = d_array[0]
        cnt = 0
        label_dict = {}
        for label in label_array:
            label_dict[label] = []
        for col in range(len(d_array[0])):
            for idx in range(1, len(d_array)):
                label_dict[d_array[0][col]].append(float(d_array[idx][col]))
        return label_dict

    def process_data(self, data_file, class_name='yc', processor=None, conversion=None):
        f = open(data_file, 'r')
        l_rv = self.vectorize_lines(f.readlines())
        #print(l_rv)
        # now turn this array of arrays into a data frame
        # 1) create a dictionary
        data_dict = self.create_dictionary(l_rv, class_label=class_name)
        #print(data_dict)
        data_df = pd.DataFrame(data_dict)
        return data_df


    def __init__(self, test_data=None, train_data=None, file_name_test=None, rtst=None, class_label=None, rtr=None, dim=None,
                 file_name=None, solver_type=None, process_d=False):
        G_solver.__init__(self,solver_type=solver_type)
        self.test_data = test_data
        self.train_data = train_data
        self.rtst = rtst
        self.rtr = rtr
        dim = None
        self.bayes = bayes_classifier()
        if file_name is not None and process_d:
            self.data_vec = self.process_data(data_file=file_name, class_name=class_label)
            self.bayes.params.set_attribs(self.data_vec, class_label=class_label)
            self.bayes.params.set_classes(self.data_vec, class_label=class_label)
            self.bayes.params.set_stats()
            self.bayes.params.calculate_priors()
            print('-----------------------------         priors ------------------')
            print(self.bayes.params.priors)
            self.bayes.min_dist_class(self.bayes.params.attribs.iloc[0,:])
            #    self.bayes.fit()
        if file_name_test is not None:
            self.test_set = self.set_testing_set(file_name_test, class_label)
            print(self.test_set)
            self.test_attribs = self.bayes.params.set_attribs(self.test_set, class_label=class_label, test=True)
            # set_classes(self, df, class_label='yc', verbose=False, test=False)
            self.test_classes = self.bayes.params.set_classes(self.test_set, test=True)
            #gs = self.bayes.predict(self.test_attribs.values)
            #self.bi_score(g, yarray, [0, 1], ['Postitive', 'Negative'])
            #score = self.bayes.bi_score(gs, self.test_classes, [0,1], ['pos', 'neg'])
            self.bayes.fit(tx=self.test_attribs, ty=self.test_classes)
    def set_testing_set(self,fname, class_label):
        return self.process_data(fname, class_name=class_label)
    def classifier(self, xarray, yarray):
        pass






