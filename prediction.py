from __future__ import division
from __future__ import print_function
import pickle
#import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
#get_ipython().magic(u'matplotlib inline')
from operator import itemgetter
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from collections import Counter
from scipy.special import comb
import gc
import itertools
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import norm
from cmath import isnan

input_graphs = 4#int(sys.argv[3])
start_date = sys.argv[1]
end_date = sys.argv[2]

# set number of cpus utilized
cpus=10

class Predictor:
    def __init__(self):
        """ 
            Loads edge list, triad features, degree featuers, initializes variables 
        """
        with open('edge_list_hd.pickle', 'rb') as handle1:
            self.edge_list = pickle.load(handle1)
            #print "Finished importing edge list"

        with open('triad_features_hd.pickle', 'rb') as handle2:
            self.triad_features = pickle.load(handle2)
            #print "Finished importing triad features"

        with open('composite_triads_hd.pickle', 'rb') as handle4:
            self.composite_triads = pickle.load(handle4)
            #print "Finished importing composite triad features"

        with open('degree_features_hd.pickle', 'rb') as handle3:
            self.degree_features = pickle.load(handle3)
            #print "Finished importing degree features"
        self.X = []
        self.y = []
        self.y_all = []
        self.X_all = None
        self.X_deg = None
        self.X_triads = None
        self.X_composite = None
        self.X_triadsAndComposite = None
        self.feature_set_names = None
        self.feature_sets = None
        self.feature_names = None
        self.gt = ["f", "q", "r", "rt"]
        self.number_df = 0
        self.number_tf = 0
        self.number_ctf = 0

    def feature_arrays(self):
        """
            Creates feature arrays in numpy formats
        """
        for e in self.edge_list:
            features = ( self.degree_features[e][:] ) 
            features.extend( self.triad_features[e][:] )
            features.extend( self.composite_triads[e][:] )
            self.X.append( features[:] )
            self.y.append( self.edge_list[e][:] )
            self.y_all.append( bv2d(self.edge_list[e]) )
        self.number_df = len( self.degree_features[e] )
        self.number_tf = len( self.triad_features[e] )
        self.number_ctf = len( self.composite_triads[e] )
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        self.y_all = np.array( self.y_all )
        self.y_all = self.y_all.astype(int)
        # try to free ram space
        self.edge_list = []
        self.triad_features = []
        self.composite_triads = []
        self.degree_features = []
        #gc.collect()
    
    def construct_predictors(self):
        """
            Creates different sets of predictor variables and their respective names for results
        """
        self.X_all     = self.X.copy()
        #self.X_deg      = np.take(self.X, range(0,2*input_graphs+2), axis=1)
        self.X_deg = np.take(self.X, range(0, self.number_df), axis = 1)   
        #self.X_triads  = np.take(self.X, range(2*input_graphs+2, 2*input_graphs+2+9*input_graphs), axis=1)
        self.X_triads = np.take(self.X, range(self.number_df, self.number_df+self.number_tf), axis = 1)
        #self.X_composite  = np.take(self.X, range(2*input_graphs+2+9*input_graphs, 2*input_graphs+2+9*input_graphs+9*int(comb(input_graphs,2, exact=False))), axis=1)
        self.X_composite = np.take(self.X, range(self.number_df+self.number_tf, self.number_df+self.number_tf+self.number_ctf), axis = 1)
        self.X_triadsAndComposite = np.take(self.X, range(self.number_df, self.number_df+self.number_tf+self.number_ctf), axis = 1)
        self.feature_set_names = ["X_all", "X_deg", "X_triads", "X_composite", "X_triads_composite"]
        self.feature_sets = [ self.X_all, self.X_deg, self.X_triads, self.X_composite, self.X_triadsAndComposite ]
        dt = ["d_out(u)-", "d_in(v)-"]
        degree_names = [d+g for d in dt for g in self.gt]
        degree_names.extend(["d_out(u)-t" , "d_in(v)-t"])
        triads_names = ["t"+str(i)+"-"+g for g in self.gt for i in range(1,10)]
        composite_triads_names = ["ct"+str(i)+"-"+self.gt[t[0]]+self.gt[t[1]] for t in list(itertools.combinations(range(input_graphs), 2)) for i in xrange(1,10)]
        self.feature_names = degree_names+triads_names+composite_triads_names
    
    def count_triads( self ):
        """
            Reports count of triad features
        """
        count_fname = "triads_count/count_triads"+start_date+"_"+end_date+".out"
        sum_of_triads = self.X_triadsAndComposite.sum(axis = 0)
        np.savetxt( count_fname, sum_of_triads, fmt='%i', delimiter='\t')

    def graph_indices( self, i ):
        """
            input
                i: graph name in int form (0: follow, 1: mention, 2: quote, 3: reply, 4: retweet)
            output
                j: list with the indices of the features (X_all) array where features from this graph are used
          
        """
        at_degrees = [i,len(self.gt)+i, self.number_df-2, self.number_df-1]
        at_triads = [t for t in xrange(self.number_df+9*i, self.number_df+9*(i+1))]
        at_composite_triads = []
        combs = list(itertools.combinations(range(input_graphs), 2))
        for c in xrange(len(combs)):
            l, m = combs[c]
            if l == i or m == i:
                at_composite_triads.extend([(self.number_df+self.number_tf+tt) for tt in range(c*9,c*9+9)])
        j = at_degrees+at_triads+at_composite_triads
        
        return j

    def predictionTask(self, algorithm = "logistic_regression", prediction_task = "single", balanced = True):
        """
            input
                prediction_task: 
                    "single": predict whether there was a directed interaction between users u,v of type i (baseline for balaned dataset: 50%)
                    "pairwise": predict vectors of size 2 (baseline for balanced dataset: 100/3% - or 25% if we give as input random edges)
                    "full": predict the whole 5-column row vector (baseline for balanced dataset: 1/2^5-1 - or 1/2^5 if we give as input random edges) (TO DO)
                balanced:
                    True for balanced dataset - NOTE THAT THIS MAY NOT BE POSSIBLE, False otherwise.
                algorithm: (TO DO : use more algorithms)
                    "logistic_regression": Linear logistic regression
           output
               prediction_rates
               features_weights
        """
        if prediction_task == "single":
            if balanced:
                # single prediction task
                for i in xrange( input_graphs ):
                # # # # #
                #   1   #
                # # # # #
                    # SIMPLE TASK - PREDICT WHETHER AN EDGE BELONGS TO GRAPH i USING ALL FEATURES
                    y_i = np.take(self.y, i, axis=1)
                    y_i = 2*y_i - 1
                    positive_candidates = np.where(y_i == 1 )[0]
                    negative_candidates = np.where(y_i == -1 )[0]
                    sample_size = min( len(positive_candidates), len(negative_candidates) )
                    print("Balanced dataset for graph {}, contains {} samples".format(i, sample_size))
                    for j in xrange(0,1):
                        positives = np.random.choice(positive_candidates, size=sample_size)
                        negatives = np.random.choice(negative_candidates, size=sample_size)
                        balanced = np.concatenate([positives, negatives])
                        weights = np.zeros(self.X_all.shape[1])
                        for x, name in zip(self.feature_sets, self.feature_set_names):
                            if algorithm == "logistic_regression":
                                lg = make_pipeline(StandardScaler(), LogisticRegression(solver='lbfgs', max_iter=2000))
                                scores = cross_val_score(lg, x[balanced], y_i[balanced], n_jobs=cpus, cv=10)
                            print(name+"\t"+"score = {:.3f}, std = {:.3f}".format(np.mean(scores), np.std(scores)))
                        # calculate the weights of the logistic regression coefficients
                        num_of_iterations = 1
                        weights = np.zeros(self.X_all.shape[1])
                        scaler = StandardScaler()
                        for _ in range(num_of_iterations):
                            lg = LogisticRegression(solver='lbfgs', n_jobs=cpus, max_iter=2000)
                            lg.fit(scaler.fit_transform(self.X_all[balanced]), y_i[balanced])
                            weights += lg.coef_[0]
                        weights /= num_of_iterations
                        weights = weights.round(decimals=4)
                        print(*self.feature_names, sep='\t')
                        print(*weights, sep='\t')
                # # # # #
                #   2   #
                # # # # #
                    # PREDICT WHETHER AN EDGE BELONGS TO GRAPH i USING ONLY GRAPH i
                        index_i = [j for j in self.graph_indices(i) if j < (self.number_df + self.number_tf) and j!= 2*input_graphs and j != 2*input_graphs + 1]
                        X_only_i = np.take(self.X_all, index_i, axis=1)
                        if algorithm == "logistic_regression":
                            lg = make_pipeline(StandardScaler(), LogisticRegression(solver='lbfgs', max_iter=2000))
                            scores = cross_val_score(lg, X_only_i[balanced], y_i[balanced], n_jobs=cpus, cv=10)
                        print("X_only_i\t"+"score = {:.3f}, std = {:.3f}".format(np.mean(scores), np.std(scores)))
                # # # # #
                #   3   #
                # # # # #
                    # PREDICT WHETHER AN EDGE BELONGS TO GRAPH i, USING i + one other j
                        for j in xrange( input_graphs ):
                            if i == j:
                                index_ij = [t for t in self.graph_indices(i) if t < (self.number_df + self.number_tf) and t!= 2*input_graphs and t != 2*input_graphs+1]
                                #X_ij = np.take(self.X_all, index_ij, axis=1)
                            else:
                                index_i = [t for t in self.graph_indices(i) if t!= 2*input_graphs and t != 2*input_graphs+1] 
                                index_j = [t for t in self.graph_indices(j) if t!= 2*input_graphs and t != 2*input_graphs+1] 
                                index_ij = sorted([t for t in index_i+index_j if t < (self.number_df + self.number_tf)]) 
                                index_ij.extend([t for t in index_i if t in index_j]) 
                            X_ij = np.take(self.X_all, index_ij, axis=1)
                        #for x, name in zip(self.feature_sets, self.feature_set_names):
                        #    if algorithm == "logistic_regression":
                            lg = make_pipeline(StandardScaler(), LogisticRegression(solver='lbfgs', max_iter=2000))
                            scores = cross_val_score(lg, X_ij[balanced], y_i[balanced], n_jobs=cpus, cv=10)
                            print("{} with {} types".format(i, j))
                            print("score = {:.3f}, std = {:.3f}".format(np.mean(scores), np.std(scores)))

def bv2d( vector ):
    """
        Converts a binary vector to decimal integer, e.g. [1,0,1] -> 5
    """
    x = ''.join([str(i) for i in vector])
    return int(x,2)


def main():
    print("Start date: {}, end date: {}".format(start_date, end_date))
    llr = Predictor()
    llr.feature_arrays()
    llr.construct_predictors()
    llr.predictionTask()
    #llr.count_triads()


if __name__ == '__main__':
    main()
