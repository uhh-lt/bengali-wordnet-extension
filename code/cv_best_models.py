import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from ampligraph.latent_features import DistMult, ComplEx, HolE
from ampligraph.evaluation import train_test_split_no_unseen, evaluate_performance, mrr_score, hits_at_n_score, select_best_model_ranking
import pyiwn
from itertools import combinations

beng_iwn = pyiwn.IndoWordNet(lang = pyiwn.Language.BENGALI)
all_syns = beng_iwn.all_synsets()

edges = pd.read_csv("../data/BengIwnData/edges.csv", header = None, delimiter = '\t', names = ['ent1','ent2', 'rel'])

X = pd.DataFrame()
X['ent1'] = edges['ent1']
X['rel'] = edges['rel']
X['ent2'] = edges['ent2']
print("The dataframe X: \n", X.head())
X = X.astype(str)

print("Value counts for relation:")
print(X['rel'].value_counts())

print("Changing the edge structure from (synset -> synset) to (word -> word)")

# converting synsets list into a hashmap
syn_dict = dict()
for val in all_syns:
  syn_dict[val.synset_id()] = val
  
def generate_triples(tuple_list, rel_id):
  triple_list = []
  for tup in tuple_list:
    triple = (tup[0], rel_id, tup[1])
    triple_list.append(triple)
  return triple_list

def generate_allcombs(list1, list2):
  res = []
  for val1 in list1:
    for val2 in list2:
      tup = (val1, val2)
      res.append(tup)
  return res
  
  
word_edges = set()
visited_syns = set()
X = X.astype(int)
for i, row in X.iterrows():
  syn1_id = row['ent1']
  syn1 = syn_dict[syn1_id]
  word_list1 = syn1.lemma_names()
  rel = row['rel']
  syn2_id = row['ent2']
  syn2 = syn_dict[syn2_id]
  word_list2 = syn2.lemma_names()

  if(syn1_id not in visited_syns):
    # generating tuples for synonymy relation - id = 31
    synonymy_combs1 = combinations(word_list1, 2)
    synonymy_triples1 = generate_triples(synonymy_combs1, '31') # add to set
    for i, val in enumerate(synonymy_triples1):
      word_edges.add(val)
    visited_syns.add(syn1_id)

  if(syn2_id not in visited_syns):
    synonymy_combs2 = combinations(word_list2, 2)
    synonymy_triples2 = generate_triples(synonymy_combs2, '31') # add to set
    for i, val in enumerate(synonymy_triples2):
      word_edges.add(val)
    visited_syns.add(syn2_id)

  # enumerating the same relation for all words in both synset lists
  word_combinations = generate_allcombs(word_list1, word_list2)
  if word_combinations:
    word_triples = generate_triples(word_combinations, str(rel)) # add to set

  for i, val in enumerate(word_triples):
    word_edges.add(val)

print(f"Number of edges with words as nodes = {len(word_edges)}")
word_edgelist = list(word_edges)
word_edgearr = np.asarray(word_edgelist)
# splitting into train, test and validation
X_train, X_test1 = train_test_split_no_unseen(word_edgearr, seed = 1, test_size = 40000)
X_train, X_test2 = train_test_split_no_unseen(X_train, seed = 1, test_size = 40000)
X_train, X_test3 = train_test_split_no_unseen(X_train, seed = 1, test_size=40000)
print("Training set = ", X_train.shape)
print("Test set1 = ", X_test1.shape)
print("Test set2 = ", X_test2.shape)

filter_positives = word_edgearr

def train_model(model, train, test):
  model.fit(train)
  
  test_ranks = evaluate_performance(test, model=model, filter_triples = filter_positives, verbose = True)
  mrr_test = mrr_score(test_ranks)
  hits_at_1_test = hits_at_n_score(test_ranks, 1)
  hits_at_3_test = hits_at_n_score(test_ranks, 3)
  hits_at_10_test = hits_at_n_score(test_ranks, 10)

  print("Results:")
  #print(f"Validation mrr = {mrr_val}")
  #print(f"Validation hits_at_10 score = {hits_at_10_val}")

  print(f"Test mrr = {mrr_test}")
  print(f"Test hits_at_1 score = {hits_at_1_test}")
  print(f"Test hits_at_3 score = {hits_at_3_test}")
  print(f"Test hits_at_10 score = {hits_at_10_test}")


distmult_model = DistMult(batches_count= 100, epochs=10, k= 100, eta= 50, optimizer= 'adagrad', loss= 'pairwise', regularizer= 'LP', verbose= False, optimizer_params= {'lr': 0.1}, regularizer_params= {'p': 2, 'lambda': 0.0001}, loss_params= {'margin': 0.5}, embedding_model_params= {'negative_corruption_entities': 'all'})

print("DistMult Model:")
print("Fold 1:")
train = np.vstack((X_train, X_test1, X_test2))
test = X_test3
train_model(distmult_model, train, test)
print("Fold 2:")
train = np.vstack((X_train, X_test1, X_test3))
test = X_test2
train_model(distmult_model, train, test)
print("Fold 3:")
train = np.vstack((X_train, X_test2, X_test3))
test = X_test1
train_model(distmult_model, train, test)

complex_model = ComplEx(batches_count= 100, epochs=10, k= 100, eta= 50, optimizer= 'adagrad', loss= 'pairwise', regularizer= 'LP', verbose= False, optimizer_params= {'lr': 0.1}, regularizer_params= {'p': 2, 'lambda': 0.0001}, loss_params= {'margin': 0.5}, embedding_model_params= {'negative_corruption_entities': 'all'})

print("ComplEx Model:")
print("Fold 1:")
train = np.vstack((X_train, X_test1, X_test2))
test = X_test3
train_model(complex_model, train, test)
print("Fold 2:")
train = np.vstack((X_train, X_test1, X_test3))
test = X_test2
train_model(complex_model, train, test)
print("Fold 3:")
train = np.vstack((X_train, X_test2, X_test3))
test = X_test1
train_model(complex_model, train, test)

hole_model = HolE(batches_count= 100, epochs=10, k= 100, eta= 50, optimizer= 'adagrad', loss= 'pairwise', regularizer= None, verbose= False, optimizer_params= {'lr': 0.1}, loss_params= {'margin': 0.5}, embedding_model_params= {'negative_corruption_entities': 'all'})

print("HolE Model:")
print("Fold 1:")
train = np.vstack((X_train, X_test1, X_test2))
test = X_test3
train_model(hole_model, train, test)
print("Fold 2:")
train = np.vstack((X_train, X_test1, X_test3))
test = X_test2
train_model(hole_model, train, test)
print("Fold 3:")
train = np.vstack((X_train, X_test2, X_test3))
test = X_test1
train_model(hole_model, train, test)


