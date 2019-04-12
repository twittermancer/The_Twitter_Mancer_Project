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
from collections import Counter, defaultdict
from scipy.special import comb
import gc
import itertools
from sklearn.metrics import balanced_accuracy_score
import random
from multiprocessing import Process, Lock, Manager
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


input_graphs = int(sys.argv[3])
start_date = sys.argv[1]
end_date = sys.argv[2]
#emb_thres = int(sys.argv[4])
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
        with open('edge_embeddedness.pickle', 'rb') as handle5:
            self.edge_emb = pickle.load(handle5) 
        self.X = []
        self.y = []
        self.y_all = []
        self.X_all = None
        self.X_deg = None
        self.X_triads = None
        self.X_composite = None
        self.X_triadsAndComposite = None
        self.X_emb = []
        self.feature_set_names = None
        self.feature_sets = None
        self.feature_names = None
        self.gt = ["f", "q", "r", "rt"]
        self.number_df = 0
        self.number_tf = 0
        self.number_ctf = 0
        #self.out_degrees = [[] for _ in range(input_graphs)]
        #self.in_degrees = [[] for _ in range(input_graphs)]
        #self.inout_degrees = [[] for _ in range(input_graphs)]
        #self.nonzero = [[] for _ in range(input_graphs)]
        #self.embeddedness1 = [[] for _ in range(input_graphs)]
        self.embeddednessAll = [[] for _ in range(input_graphs)]

    def feature_arrays(self):
        """
            Creates feature arrays in numpy formats
        """
        for e in self.edge_list:
            #if self.edge_emb[e][input_graphs] < emb_thres:
            #    continue
            features = ( self.degree_features[e][:] ) 
            features.extend( self.triad_features[e][:] )
            features.extend( self.composite_triads[e][:] )
            self.X.append( features[:] )
            self.y.append( self.edge_list[e][:] )
            self.y_all.append( bv2d(self.edge_list[e]) )
            self.X_emb.append( self.edge_emb[e][:] )
        self.number_df = len( self.degree_features[e] )
        self.number_tf = len( self.triad_features[e] )
        self.number_ctf = len( self.composite_triads[e] )
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        self.X_emb = np.array(self.X_emb)
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
        #self.X_all = self.X.copy()
        
        #self.X_deg      = np.take(self.X, range(0,2*input_graphs+2), axis=1)
        self.X_deg = np.take(self.X, range(0, self.number_df), axis = 1)   
        
        
        #self.X_triads  = np.take(self.X, range(2*input_graphs+2, 2*input_graphs+2+9*input_graphs), axis=1)
        self.X_triads = np.take(self.X, range(self.number_df, self.number_df+self.number_tf), axis = 1)
        
        #self.X_composite  = np.take(self.X, range(2*input_graphs+2+9*input_graphs, 2*input_graphs+2+9*input_graphs+9*int(comb(input_graphs,2, exact=False))), axis=1)
        self.X_composite = np.take(self.X, range(self.number_df+self.number_tf, self.number_df+self.number_tf+self.number_ctf), axis = 1)
        
        self.X_triadsAndComposite = np.take(self.X, range(self.number_df, self.number_df+self.number_tf+self.number_ctf), axis = 1)
        
        self.feature_set_names = ["X_all", "X_deg", "X_triads", "X_composite", "X_triads_composite"]
        self.feature_sets = [ self.X, self.X_deg, self.X_triads, self.X_composite, self.X_triadsAndComposite ]
        dt = ["d_out(u)-", "d_in(v)-"]
        degree_names = [d+g for d in dt for g in self.gt]
        degree_names.extend(["d_out(u)-t" , "d_in(v)-t"])
        triads_names = ["t"+str(i)+"-"+g for g in self.gt for i in range(1,10)]
        composite_triads_names = ["ct"+str(i)+"-"+self.gt[t[0]]+self.gt[t[1]] for t in list(itertools.combinations(range(input_graphs), 2)) for i in xrange(1,10)]
        self.feature_names = degree_names+triads_names+composite_triads_names

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
                    #self.out_degrees[i] = {}
                    #self.in_degrees[i] = {}
                    #self.inout_degrees[i] = {}
                    #self.nonzero[i] = {}
                    #self.embeddedness1[i] = {}
                    self.embeddednessAll[i] = {}
                    y_i = np.take(self.y, i, axis=1)
                    y_i = 2*y_i -1
                    positive_candidates = np.where(y_i == 1 )[0]
                    negative_candidates = np.where(y_i == -1 )[0]
                    sample_size = min( len(positive_candidates), len(negative_candidates) )
                    print("Balanced dataset for graph {}, contains {} samples\n".format(i, sample_size))
                    positives = np.random.choice(positive_candidates, size=sample_size)
                    #print("Positive instances for graph {}, are {} samples\n".format(i,len(positive_candidates)))
                    #print("Negative instances for graph {}, are {} samples\n".format(i, len(negative_candidates)))
                    negatives = np.random.choice(negative_candidates, size=sample_size)
                    balanced = np.concatenate([positives, negatives])
                    y_balanced = y_i[balanced]
                    y_all_balanced = self.y[balanced]
                    x_emb_balanced = self.X_emb[balanced]
                    num_of_iterations = 10
                    sample_indices = [si for si in xrange(2*sample_size-1)]
                    random.shuffle(sample_indices)
                    sample_indices = list(split(sample_indices, num_of_iterations))
                    #scores = [0.5]
                    #for x, name in zip(self.feature_sets, self.feature_set_names):
                    #    if algorithm == "logistic_regression":
                    #        lg = make_pipeline(StandardScaler(), LogisticRegression(solver='lbfgs', max_iter=2000))
                    #        scores = cross_val_score(lg, x[balanced], y_i[balanced], n_jobs=cpus, cv=10)
                    #        print(name+"\t"+"score = {:.3f}, std = {:.3f}".format(np.mean(scores), np.std(scores)))
                    #if True:
                    #    continue # hope we will not get past through this point
                    #print("Test!!!")
                    for x, name in zip(self.feature_sets, self.feature_set_names):
                        if name != "X_all":
                            continue
                    #    self.out_degrees[i] = defaultdict(list)
                    #    self.in_degrees[i] = defaultdict(list)
                    #    self.inout_degrees[i] = defaultdict(list)
                    #    self.nonzero[i] = defaultdict(list)
                    #    self.embeddedness1[i] = defaultdict(list)
                        self.embeddednessAll[i] = defaultdict(list)
                        # calculate the weights of the logistic regression coefficients
                        num_of_iterations = 10
                        x_balanced = x[balanced]
                        #lock = Lock()
                        manager = Manager()
                        pre_dict = manager.dict()
                        #for s in xrange(num_of_iterations):
                        def doCrossVal(s, pred_dict, sample_indices, x_balanced, y_balanced):
                            print("Graph {} - Thread {}".format(i, s))
                            lg = LogisticRegression(solver='lbfgs', n_jobs=2, max_iter=2000)
                            train_set = sample_indices[:s] + sample_indices[s+1:]
                            train_set = list(itertools.chain(*train_set))
                            test_set = sample_indices[s]
                            scaler = StandardScaler()
                            lg.fit(scaler.fit_transform(x_balanced[train_set]), y_balanced[train_set])
                            y_predicted = lg.predict(scaler.fit_transform(x_balanced[test_set]))
                            #l.acquire()
                            pre_dict[s] = y_predicted[:]
                            #self.update_dictionary( x_balanced[test_set], x_emb_balanced, y_balanced[test_set], y_predicted, y_all_balanced[test_set], gt, name )
                            #l.release()
                            print("Thread {} - Finished work, lock released.".format(s))
                        p = [[] for _ in xrange(num_of_iterations)]
                        for s in xrange(num_of_iterations):
                            p[s] = Process(target=doCrossVal, args=(s, pre_dict, sample_indices, x_balanced, y_balanced))
                            p[s].start() 
                        for s in xrange(num_of_iterations):
                            p[s].join()
                        print("Processes finished work - Update dictionaries")
                        for s in xrange(num_of_iterations):
                            test_set = sample_indices[s]
                            self.update_dictionary( x_balanced[test_set], x_emb_balanced[test_set], y_balanced[test_set], pre_dict[s], y_all_balanced[test_set], i )    

    def update_dictionary( self, x, x_emb, y_true, y_pred, y_all, gt ):
        """
            Function that updates the dictionary that keeps track of accuracy
            based on the degree of a node
            input
                x : feature matrix for the test samples
                x_emb : embeddedness of every edge
                y_true : true values for the test samples (-1,1)
                y_pred : predicted values for the test samples (-1,1)
                gt : graph type
             updates
                self.out_degrees : dictionary for accuracy based on the out degree of nodes
                self.in_degrees : dictionary for accuracy based on the in degree of nodes
                self.inout_degrees : dictionary for accuracy based on tuple of out and in degree of nodes
                self.nonzero : dictionary for accuracy based on the number of different interactions between two users
                self.embeddedness1 : dictionary for accuracy based on the embeddedness of an edge on an undirected graph of a particular type
                self.embeddednessAll : dictionary for accuracy based on the embeddedness of an edge on all-type undirected graph
        """
        print("Sanity check 1: {}".format(len(y_pred)))
        for i in xrange(len(y_pred)):
            true_value = (y_true[i] + 1)//2 # revert to 0-1
            pred_value = (y_pred[i] + 1)//2 # revert to 0-1
            #out_deg = x[i][gt] + true_value
            #in_deg = x[i][input_graphs+gt] + true_value
            #in_n_out = (out_deg, in_deg)
            #nonzero = np.nonzero(y_all[i])[0].shape[0]
            #emb_i = int(x_emb[i][gt])
            emb_all = int(x_emb[i][input_graphs])
            
            # update out degree
            #self.out_degrees[gt][out_deg].append( true_value == pred_value )

            # update in degree
            #self.in_degrees[gt][in_deg].append( true_value == pred_value )

            # update (in,out)
            #self.inout_degrees[gt][in_n_out].append( true_value == pred_value )

            # update number of different interactions
            #self.nonzero[gt][nonzero].append( true_value == pred_value )

            # update embeddedness of same type
            #self.embeddedness1[gt][emb_i].append( true_value == pred_value )

            # update embeddedness of all types
            self.embeddednessAll[gt][emb_all].append( true_value == pred_value )

        return

    def save_dictionaries(self, sd, ed):
        #out_name = 'results/out_degrees_'+sd+'_'+ed+'.pk'
        #in_name = 'results/in_degrees_'+sd+'_'+ed+'.pk'
        #inout_name = 'results/inout_degrees_'+sd+'_'+ed+'.pk'
        #nonzero_name = 'results/nonzero_degrees_'+sd+'_'+ed+'.pk'
        #emb1_name = 'results/emb1_'+sd+'_'+ed+'.pk'
        embAll_name = 'results/embAll_'+sd+'_'+ed+'.pk'
	#print("Sanity check, {}".format(len(self.out_degrees)))
        #with open( out_name, 'wb' ) as f1:
        #    pickle.dump(self.out_degrees, f1, protocol=pickle.HIGHEST_PROTOCOL)

        #with open( in_name, 'wb' ) as f2:
        #    pickle.dump(self.in_degrees, f2, protocol=pickle.HIGHEST_PROTOCOL)

        #with open( inout_name, 'wb' ) as f3:
        #    pickle.dump(self.inout_degrees, f3, protocol=pickle.HIGHEST_PROTOCOL)
        
        #with open( nonzero_name, 'wb' ) as f4:
        #    pickle.dump(self.nonzero, f4, protocol=pickle.HIGHEST_PROTOCOL)
        
        #with open( emb1_name, 'wb' ) as f5:
        #    pickle.dump(self.embeddedness1, f5, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open( embAll_name, 'wb' ) as f6:
            pickle.dump(self.embeddednessAll, f6, protocol=pickle.HIGHEST_PROTOCOL)

        return

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in xrange(n))

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
    llr.save_dictionaries(start_date, end_date)

if __name__ == '__main__':
    main()
