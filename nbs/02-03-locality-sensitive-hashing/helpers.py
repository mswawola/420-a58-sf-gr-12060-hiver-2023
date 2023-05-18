import numpy as np
from scipy.sparse import csr_matrix


def load_sparse_csr(filename):
    loader = np.load(filename)
    data = loader['data']
    indices = loader['indices']
    indptr = loader['indptr']
    shape = loader['shape']
    
    return csr_matrix( (data, indices, indptr), shape)


def checkcode_ex3(model):
    table = model['table']
    if   0 in table and table[0]   == [39583] and 143 in table and table[143] == [19693, 28277, 29776, 30399]:
        print('Bravo, cela fonctionne !')
    else:
        print('Vérifier votre code...')

    
def checkcode_ex5_searchradius0(model, f):
    obama_bin_index = model['bin_index_bits'][35817] # bin index of Barack Obama
    candidate_set = f(obama_bin_index, model['table'], search_radius=0)
    if candidate_set == set([35817, 21426, 53937, 39426, 50261]):
        print('TEST#1 - Bravo, cela fonctionne !')
    else:
        print('TEST#1 - Vérifier votre code...')
    
    return candidate_set

    
def checkcode_ex5_searchradius1(model, f, candidate_set):
    obama_bin_index = model['bin_index_bits'][35817] # bin index of Barack Obama
    candidate_set = f(obama_bin_index, model['table'], search_radius=1, initial_candidates=candidate_set)
    if candidate_set == set([39426, 38155, 38412, 28444, 9757, 41631, 39207, 59050, 47773, 53937, 21426, 34547,
                         23229, 55615, 39877, 27404, 33996, 21715, 50261, 21975, 33243, 58723, 35817, 45676,
                         19699, 2804, 20347]):
        print('TEST#2 - Bravo, cela fonctionne !')
    else:
        print('TEST#2 - Vérifier votre code...')
    
    
def norm(x):
    sum_sq=x.dot(x.T)
    norm=np.sqrt(sum_sq)
    
    return(norm)
    
    
def cosine_distance(x, y):
    xy = x.dot(y.T)
    dist = xy/(norm(x)*norm(y))
    
    return 1-dist[0,0]