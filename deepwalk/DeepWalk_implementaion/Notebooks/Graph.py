
# coding: utf-8

# In[1]:

import networkx as nx
import random
import scipy.io
from itertools import repeat

# In[2]:

#network = pd.read_csv('Flickr-dataset/data/edges.csv',header=None)
#groups = pd.read_csv('Flickr-dataset/data/group-edges.csv',header=None)
#g = nx.read_edgelist('edges.csv',delimiter=',',create_using=nx.Graph(),nodetype=int)
#path = 'blogcatalog.mat'


# In[114]:

def parse_mat_file(path):
    
    edges = []
    g = nx.Graph()
    mat = scipy.io.loadmat(path)
    nodes = mat['network'].tolil()
    subs_coo = mat['group'].tocoo()
    
    for start_node,end_nodes in enumerate(nodes.rows, start=0):
        for end_node in end_nodes:
            edges.append((start_node,end_node))
    
    g.add_edges_from(edges)
    g.name = path
    print(nx.info(g) + "\n---------------------------------------\n")
        
    return g, subs_coo


# In[4]:

def random_walk(G, start_node, path_len):
    
    path = [str(start_node)]
    current = start_node
    
    while(len(path) < path_len):
        neighbors = list(G.neighbors(current))
        
        if(len(neighbors) == 0):
            break
        
        current = random.choice(neighbors)
        path.append(str(current))
        
        #Restarts[Back to root node] allowed. Its also ok if it randomly picks its previous neighbour in the path.
    return path


# In[5]:

def remove_self_loops(G):
    
    loops = []
    
    #Get loops
    for i,j in G.edges_iter():
        if i==j:
            loops.append((i,j))
    
    G.remove_edges_from(loops)
    return G


# In[6]:
def save_corpus(max_paths, path_len, corpus):
    
    fname = "Random_walks/RandomWalks-w"+str(max_paths)+"-l"+str(path_len)+".txt"
    with open(fname,'w+') as f:
        [f.writelines("%s\n" % ' '.join(walk)) for walk in corpus]
    print("Corpus saved on disk as "+fname)
    return

def load_corpus(G, fname):
    try:
        with open(fname) as f:
            x = f.readlines()
        z = [list(a.rstrip('\n').split()) for a in x]
        max_path = len(z)/len(G)
        path_len = len(z[0])
        print("Successfully loaded corpus from file ",fname)
        return z, max_path, path_len, True
    except IOError:
        print("File not found. Proceeding to generate new walks")
        # Y/N here
        return _, _, _, False


def build_walk_corpus(G, max_paths, path_len):
    
    print("Building walk corpus with parameters : max_paths per node = ",max_paths," and path_length = ",path_len)
    corpus = []
    nodes = list(G)
    
    for path_count in range(max_paths):
        random.Random(0).shuffle(nodes)
        corpus = corpus + list(map(random_walk, repeat(G), nodes, repeat(path_len)))     
    
    print("Completed")
    
    return corpus


# In[ ]:



