start_date = 1
end_date = 28
for i in range(1):
    """ Reads input graph, saves edge list ,node mapping (useful for twitter graphs), edge mapping (for numpy arrays)
        returns number of nodes, number of edges
        """
    # read node_mapping file from pickle form
    node_mapping = {}
    k = 0

    graph_types = ["quote", "reply", "retweet"]
    filenames = [gt+"-2018-02-"+str(i).zfill(2)+".txt" for i in xrange(int(start_date), int(end_date)+1) for gt in graph_types]
    for filename in filenames:
        t = 0
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

print "There are {} unique users in the whole dataset".format(k)

filename = "data/follow-jan-2018.txt"
followSet = set()
t = 0
z = 0
with open(filename, 'r') as f:
    for edge in f:
        t += 1
        if t % 1000000 == 0:
            print "Done processing {} edges from follow graph - {} so far added".format(t, z)
        nodes = edge.split()
        u = nodes[0]
        v = nodes[1]
        if (u == "deleted") or (v == "deleted"):
            pass
        elif (u == v):
            pass
        else:
            if u in node_mapping and v in node_mapping:
                followSet.add((u,v))
                z += 1

import cPickle as pickle
with open('followSet.pk', 'wb') as f1:
    pickle.dump(followSet, f1, protocol=pickle.HIGHEST_PROTOCOL)    
