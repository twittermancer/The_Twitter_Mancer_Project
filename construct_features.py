#!/usr/bin/env python
# coding: utf-8

#from itertools import gzip
import cPickle as pickle
#import pandas as pd
import numpy as np
import os
import gzip
import time
import sys
#get_ipython().magic(u'matplotlib inline')
from operator import itemgetter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import random
from random import randint

input_graphs = 4#int(sys.argv[3])

start_date = sys.argv[1]
end_date = sys.argv[2]


def get_offset( filename ):
        if filename.startswith('follow'):
            offset = 0
        elif filename.startswith('quote'):
            offset = 1
        elif filename.startswith('reply'):
            offset = 2
        elif filename.startswith('retweet'):
            offset = 3
        else:
            print("Not valid input name")
        return offset

def get_graph( i ):
    if i == 0:
        return "Follow"
    if i == 1:
        return "Quote"
    if i == 2:
        return "Reply"
    if i == 3:
        return "Retweet"

def preload_follow():
    """
 
    (pre) Loads the follow graph till (Jan 2018) from pickle file.

    """
    with open('followSet.pk', 'rb') as handle:
            followSet = pickle.load(handle)
    return followSet

def read_graph( ):
    start = time.time()
    """ Reads input graph, saves edge list ,node mapping (useful for twitter graphs), edge mapping (for numpy arrays)
        returns number of nodes, number of edges
        """
    
    # load follow graph as crawled so far
    followSet = preload_follow()
    # create node mapping from (hashed) IDs to 0..n range
    # compresses storing in dictionaries
    node_mapping = {}
    k = 0
    ffc = 0
    eList = dict()
            
    eList_undirected = set()
    
    graph_types = ["quote","reply","retweet"]
    filenames = [gt+"-2018-02-"+str(i).zfill(2)+".txt" for i in xrange(int(start_date), int(end_date)+1) for gt in graph_types]       
    for filename in filenames:
        t = 0
        offset = get_offset( filename )
        with open('data/'+filename, 'r') as f:
            for edge in f:
                nodes = edge.split()
                u = nodes[0]
                v = nodes[1]
                if (u == "deleted") or (v == "deleted"):
                    pass
                elif ( u == v ):
                    pass
                else:
                    if u not in node_mapping:
                        node_mapping[u] = k
                        k += 1
                    if v not in node_mapping:
                        node_mapping[v] = k
                        k += 1
                    u1 = node_mapping[u]
                    v1 = node_mapping[v]
                    if (u1, v1) not in eList:
                        eList[(u1, v1)] = [0 for _ in range(input_graphs)]
                        eList[(u1, v1)][offset] = 1
                        # first time an edge is observed, look if our crawler has identified, some follow relationship
                        if (u,v) in followSet:
                            ffc += 1
                            eList[(u1,v1)][0] = 1
                    else:
                        t += 1
                        eList[(u1, v1)][offset] = 1
                    if (v1,u1) not in eList_undirected:
                        eList_undirected.add((u1, v1))
               
        print "Finished reading {} graph".format( filename )
        print "Number of nodes so far: {}".format( k )
        print "Number of edges so far: {}".format( len(eList) )
	print "Common edges in this iteration: {}".format( t )
    print "So far there are: {} follow edges".format(ffc)
    # ADD FOLLOW EDGES ONLY TO EXISTING EDGES!
    print "Add follow relationships to existing edges"
    graph_types = ["follow"]
    filenames = [gt+"-2018-02-"+str(i).zfill(2)+".txt" for i in xrange(int(start_date), int(end_date)+1) for gt in graph_types]
    for filename in filenames:
        offset = get_offset(filename)
        with open('data/'+filename, 'r') as f:
            for edge in f:
                nodes = edge.split()
                u = nodes[0]
                v = nodes[1]
                if (u == "deleted") or (v == "deleted"):
                    pass
                elif ( u == v ):
                    pass
                else:
                    if (u in node_mapping) and (v in node_mapping):
                        u1 = node_mapping[u]
                        v1 = node_mapping[v]
                        if (u1,v1) in eList:
                            ffc += 1
                            eList[(u1,v1)][offset] = 1
    
    # SAVE EDGE LIST DICTIONARY AS PICKLE FILES
    with open('edge_list_hd.pickle', 'wb') as f2:
        pickle.dump(eList, f2, protocol=pickle.HIGHEST_PROTOCOL)
    end = time.time()
    elapsed = end - start
    print " FFC = {}".format(ffc)
    print "Time creating multilayer graph:{}".format(elapsed)

    return eList, eList_undirected


def create_mace_graph( eList_undirected ):

    max0 = max(eList_undirected, key=lambda x: x[0])[0]
    max1 = max(eList_undirected, key=lambda x: x[1])[1]
    nn = max(max0, max1)+1
    edges = [[] for i in range(nn)]
    for e in eList_undirected:
        edges[min(e[0],e[1])].append(max(e[0],e[1]))
    for i in xrange(len(edges)):
        edges[i] = sorted(edges[i])

    f = open('twitter_hd.mace', 'w')
    for node in edges:
        neis = ' '.join([str(t) for t in node])
        f.write(neis+'\n')
        
    
def degree_features( eList ):

    from collections import Counter

    edges = eList.keys()
    max0 = max(edges, key=lambda x: x[0])[0]
    max1 = max(edges, key=lambda x: x[1])[1]
    nnodes = max(max0,max1)+1
    degrees = [[0 for _ in range(2*input_graphs)] for n in range(nnodes)]   
    print "Finished init"
    for e in eList:
        for i in range(input_graphs):
            degrees[e[0]][i] += eList[e][i]
            degrees[e[1]][input_graphs+i] += eList[e][i]
    #print "Plotting outdegree distributions"
    #for i in range(input_graphs):
    #    outdeg = [x[i] for x in degrees]
    #    outdist = dict(Counter(outdeg))
    #    items = sorted(outdist.items())
    #    fig = plt.figure()
    #    ax = fig.add_subplot(111)
    #    ax.plot([k for (k,v) in items], [v for (k,v) in items])
    #    ax.set_xscale('log')
    #    ax.set_yscale('log')
    #    ttle1 = "Log-log outdegree distribution for "+get_graph(i)+" graph "+"("+start_date+"-"+end_date+")"
    #    plt.title(ttle1)
    #    ttle2 = "degree_figures/"+get_graph(i)+"("+start_date+"-"+end_date+")_outdegree.eps"
    #    plt.savefig(ttle2, format='eps', dpi=1000)
    #print "Plotting indegree distributions"
    #for i in range(input_graphs):
    #    indeg = [x[input_graphs+i] for x in degrees]
    #    indist = dict(Counter(indeg))
    #    items = sorted(indist.items())
    #    fig = plt.figure()
    #    ax = fig.add_subplot(111)
    #    ax.plot([k for (k,v) in items], [v for (k,v) in items])
    #    ax.set_xscale('log')
    #    ax.set_yscale('log')
    #    ttle1 = "Log-log indegree distribution for "+get_graph(i)+" graph "+"("+start_date+"-"+end_date+")"
    #    plt.title(ttle1)
    #    ttle2 = "degree_figures/"+get_graph(i)+"("+start_date+"-"+end_date+")_indegree.eps"
    #    plt.savefig(ttle2, format='eps', dpi=1000)

    degree_feature = {}
    for e in eList:
        degree_vector = degrees[e[0]][:input_graphs]+degrees[e[1]][input_graphs:]
        for i in xrange(input_graphs):
            degree_vector[i] -= eList[e][i]
            degree_vector[input_graphs+i] -= eList[e][i]
        degree_vector += [sum(i > 0 for i in degree_vector[:input_graphs])]+[sum(i>0 for i in degree_vector[input_graphs:])]
        degree_feature[e] = degree_vector[:]
    with open('degree_features_hd.pickle', 'wb') as f1:
        pickle.dump(degree_feature, f1, protocol=pickle.HIGHEST_PROTOCOL)
    
    return

def get_triad_type(u, v, t, eList, i):
    try:
        if eList[(u,v)][i] == 1:
            e1 = True
        else:
            e1 = False
    except:
        e1 = False
    try:
        if eList[(v,t)][i] == 1:
            e2 = True
        else:
            e2 = False
    except:
        e2 = False
    try:
        if eList[(v,u)][i] == 1:
            e1op = True
        else:
            e1op = False
    except:
        e1op = False
    try:
        if eList[(t,v)][i] == 1:
            e2op = True
        else:
            e2op = False
    except:
        e2op = False
    #triad 0
    if (not e1) and e1op and e2 and (not e2op):
        return 0
    #triad 1
    if (not e1op) and e1 and e2op and (not e2):
        return 1
    #triad 2
    if (not e1op) and e1 and e2 and (not e2op):
        return 2
    #triad 3
    if (not e1) and e1op and e2op and (not e2):
        return 3
    #triad 4
    if (not e1op) and e1 and e2 and e2op:
        return 4
    #triad 5
    if e1 and e1op and (not e2) and e2op:
        return 5
    #triad 6
    if (not e1) and e1op and e2 and e2op:
        return 6
    #triad 7
    if e1 and e1op and e2 and (not e2op):
        return 7
    #triad 8
    if e1 and e1op and e2 and e2op:
        return 8
    return -1
    

def get_composite_triad(u, v, t, eList, i, j):
    i_uv = j_uv = i_vt = j_vt = False
    if (u,v) in eList:
        if eList[(u,v)][i] == 1:
            i_uv = True
        if eList[(u,v)][j] == 1:
            j_uv = True
    if not i_uv or not j_uv:
        if (v,u) in eList:
            if not i_uv:
                if eList[(v,u)][i] == 1:
                    i_uv = True
            if not j_uv:
                if eList[(v,u)][j] == 1:
                    j_uv = True
    if (v,t) in eList:
        if eList[(v,t)][i] == 1:
            i_vt = True
        if eList[(v,t)][j] == 1:
            j_vt = True
    if not i_vt or not j_vt:
        if (t,v) in eList:
            if not i_vt:
                if eList[(t,v)][i] == 1:
                    i_vt = True
            if not j_vt:
                if eList[(t,v)][j] == 1:
                    j_vt = True
    #triad 0
    if  i_uv and j_uv and i_vt and j_vt:
        return 0
    #triad 1
    if  i_uv and j_uv and i_vt and not j_vt:
        return 1
    #triad 2
    if  i_uv and j_uv and not i_vt and j_vt:
        return 2
    #triad 3
    if  i_uv and not j_uv and i_vt and j_vt:
        return 3
    #triad 4
    if  not i_uv and j_uv and i_vt and j_vt:
        return 4
    #triad 5
    if  i_uv and not j_uv and i_vt and not j_vt:
        return 5
    #triad 6
    if  i_uv and not j_uv and not i_vt and j_vt:
        return 6
    #triad 7
    if  not i_uv and j_uv and i_vt and not j_vt:
        return 7
    #triad 8
    if  not i_uv and j_uv and not i_vt and j_vt:
        return 8
    return -1

def triad_features( eList ):
    import itertools
    from scipy.special import comb
    
    triads = {}
    composite_triads = {}
    edge_emb = {} 
    init_val = [0 for _ in range(9*input_graphs)]
    init_comp = [0 for _ in range(9*int(comb(input_graphs,2, exact=True)))]
    init_emb = [0 for _ in range(input_graphs+1)]
    for e in eList:
        triads[e] = init_val[:]
        composite_triads[e] = init_comp[:]
        edge_emb[e] = init_emb[:]
    print "Start processing triangles"
    prog_check = 1000000
    l = 0
    if color_sampling == True :
        nfiles = 3
    else:
        nfiles = 1
    for i in range(1,nfiles+1):
        print "Reading file #{}".format(i)
        if nfiles > 1:
            filename = 'twitter'+str(i)+'_hd.triangles'
        else:
            filename = 'twitter_hd.triangles'
        with open(filename, 'r') as f:
            for line in f:
                triangle = [int(x) for x in line.split()]
                for item in itertools.permutations(triangle, 2):
                    u, t = item[0], item[1]
                    if (u,t) in eList:
                        edge_emb[(u,t)][input_graphs] += 1
                        v = list(set(item).symmetric_difference(set(triangle)))[0]
                        combs = list(itertools.combinations(range(input_graphs), 2))
                        for c in range(len(combs)):
                            i, j = combs[c]
                            composite_triad = get_composite_triad( u, v, t, eList, i, j )
                            if composite_triad != -1:
                                composite_triads[(u,t)][c*9+composite_triad] += 1
                        for i in range(0,input_graphs): # optimize here - do it in parallel!
                        #if eList[(u,v)][i] == 1:
                            triad_type = get_triad_type(u, v, t, eList, i)
                            if triad_type != -1:
                                edge_emb[(u,t)][i] += 1
                                triads[(u,t)][i*9+triad_type] += 1 # DO IT WITH NUMPY
                l += 1
                if l % prog_check == 0:
                    print "Done processing {} triangles".format(l)
    with open('composite_triads_hd.pickle', 'wb') as f2:
        pickle.dump(composite_triads, f2, protocol=pickle.HIGHEST_PROTOCOL)                            
    with open('triad_features_hd.pickle', 'wb') as f1:
        pickle.dump(triads, f1, protocol=pickle.HIGHEST_PROTOCOL)
    with open('edge_embeddedness.pickle', 'wb') as f3:
        pickle.dump(edge_emb, f3, protocol=pickle.HIGHEST_PROTOCOL)
    return
            



def main():
    from subprocess import Popen
    print "Working for time period: {} - {}".format(start_date, end_date)
    eList, eList_undirected = read_graph(  )
       
    create_mace_graph( eList_undirected )

    cmd = "mace/mace C -l 3 twitter_hd.mace twitter_hd.triangles"
    p = Popen(cmd, shell=True)
    p.wait()

    degree_features( eList )

    triad_features( eList )
    
if __name__ == '__main__':
    main()
