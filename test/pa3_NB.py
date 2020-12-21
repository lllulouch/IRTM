import numpy as np
import random
from pa2 import create_dict, create_vector, read_dict, read_vector
from datetime import datetime

"""
1. Create dictionary and binary vectors
2. Feature Selection with training set
3. Estimate P_tc, P_c with training set
4. Predict testing set with P_tc, P_c
"""

class Config(object):
  dictionary_file = 'dictionary.txt'
  training_file = 'training.txt'
  output_file = f'{datetime.now().strftime("%Y%m%dT%H%M")}_max_seg_chi2_train.csv'
  set_size = 5739
  max_feature_size = 500
  class_size = 5
  feedback_round = 2
  p1=1
  p2=1

def init(inds=None):
  create_dict(inds)
  create_vector()

class Dataset(object):
  def __init__(self, inds, y=None, x=None):
    self.inds = inds
    self.y = y
    if x is not None:
      self.x = x
    else:
      self.x = []
      for i in inds:
        filename = f'doc{i}.txt'
        self.x.append(read_vector(filename))
      self.x = np.array(self.x, dtype=object)
  def __len__(self):
    return len(self.inds)
  def __getitem__(self, key):
    if self.y is not None:
      return self.inds[key], self.x[key], self.y[key]
    else:
      return self.x[key]
  def __iter__(self):
    if self.y is not None:
      for i in range(len(self.inds)): 
        yield self.inds[i], self.x[i], self.y[i]
    else:
      for i in range(len(self.inds)): 
        yield self.inds[i], self.x[i], None
  def get_subset(self, keys):
    if self.y is not None:
      return Dataset(inds=self.inds[keys],
                     y=self.y[keys],
                     x=self.x[keys])
    else:
      return Dataset(inds=self.inds[keys],
                     x=self.x[keys])

def read_training(filename=Config.training_file):
    with open(filename, 'r') as file:
        lines = [l.split(' ') for l in file.read().split('\n') if l]
    y = -np.ones((Config.set_size, 1))
    for l in lines:
        c = int(l[0])
        y[[int(doc)-1 for doc in l[1:] if doc], 0] = c
    return y

def get_dataset(filename=Config.training_file):
  y = read_training(filename)
  dataset_train = Dataset(np.where(y != -1)[0]+1, y[y != -1])
  dataset_test = Dataset(np.where(y == -1)[0]+1)
  return dataset_train, dataset_test

def freq(terms, dataset):
  if type(terms[0]) in [list, tuple]:
    terms = [id for id, tup in terms]
  X = np.zeros((len(dataset), len(terms)))
  Y = np.zeros((len(dataset), Config.class_size))
  for i, (id, x, y) in enumerate(dataset):
    X[i, np.isin(terms, x[:,0])] = 1
    Y[i, int(y-1)] = 1
  return X.T @ Y

def chi2(mat):
  N = np.sum(mat)
  l = np.sum(mat, axis=1).reshape(-1,1)
  r = np.sum(mat, axis=0).reshape(1,-1)
  exp = (l @ r) / N
  return (mat - exp) ** 2 / exp

###################################
# Feature Selections
def random_selection(terms, size=Config.max_feature_size, seed=None, only_id=True):
  if seed:
    np.random.seed(seed)
    random.seed(seed)
  if type(terms[0]) == tuple and only_id:
      return [id for id, (term, df, tf) in random.choices(terms, k=size)]
  else:
    return random.choices(terms, k=size)
def max_chi2(terms, dataset, size=Config.max_feature_size, only_id=True):
  f = freq(terms, dataset)
  N = np.sum(f)
  t = np.sum(f, axis=1).reshape(-1,1)
  c = np.sum(f, axis=0).reshape(1,-1)
  t1c1 = f
  t1c0 = t-f
  t0c1 = c-f
  t0c0 = N-f
  sum_chi2 = np.zeros(len(terms))
  for mat in [t1c1, t1c0, t0c1, t0c0]:
    sum_chi2 += np.sum(chi2(mat), axis=1)
  selected = np.argsort(sum_chi2)[-size:]
  if type(terms[0]) == tuple and only_id:
      return list(np.array([id for id, tup in terms])[selected])
  else:
    return list(terms[selected])
def max_seg_chi2(terms, dataset, limit=None, size=Config.max_feature_size, seed=None, only_id=True):
  f = freq(terms, dataset)
  N = np.sum(f)
  t = np.sum(f, axis=1).reshape(-1,1)
  c = np.sum(f, axis=0).reshape(1,-1)
  t1c1 = f
  t1c0 = t-f
  t0c1 = c-f
  t0c0 = N-f
  sum_chi2 = np.zeros((len(terms), Config.class_size))
  for mat in [t1c1, t1c0, t0c1, t0c0]:
    sum_chi2 += chi2(mat)
  chi2_max = np.argsort(np.sum(sum_chi2, axis=1))[::-1]
  class_max = np.argmax(sum_chi2, axis=1)
#   return chi2_max, class_max
  if limit is None:
    limit = np.ceil(size / Config.class_size) *  np.ones((Config.class_size))
  elif type(limit) == int or type(limit) == float:
    limit = limit * np.ones((Config.class_size))
  # print(limit)
  selected = []
  i = 0
  cnt = 0
  # d = []
  while cnt < size and i < len(class_max):
    cl = class_max[i]
    if limit[cl] > 0:
        selected.append(chi2_max[i])
        limit[cl] -= 1
        cnt += 1
        # d.append(cl)
    i += 1
  # print(limit)
  if type(terms[0]) == tuple and only_id:
      return list(np.array([id for id, tup in terms])[selected])
  else:
    return list(terms[selected])
###################################

class Classifier(object):
  def __init__(self, dictionary, class_size=Config.class_size):
    self.dictionary = sorted(dictionary)
    self.class_size = class_size
    self.log_P_tc = None
    self.log_P_c = None
    self.mat = None
  def train(self, dataset):
    X = np.zeros((len(dataset), len(self.dictionary)))
    Y = np.zeros((len(dataset), self.class_size))
    for i, (id, x, y) in enumerate(dataset):
        # X[i, np.isin(self.dictionary, x[:,0])] = x[np.isin(x[:,0], self.dictionary), 1]
        X[i, np.isin(self.dictionary, x[:,0])] = 1
        Y[i, int(y-1)] = 1
    self.mat = X.T @ Y + .1
    self.log_P_c = np.log((np.sum(Y,axis=0)+1) / (Y.shape[0]+Y.shape[1]))
    self.log_P_tc = np.log(self.mat / np.sum(self.mat,axis=0))
  def predict(self, dataset):
    X = np.zeros((len(dataset), len(self.dictionary)))
    for i, (id, x, y) in enumerate(dataset):
        X[i, np.isin(self.dictionary, x[:,0])] = x[np.isin(x[:,0], self.dictionary), 1]
    Y = np.argmax((X @ self.log_P_tc) + self.log_P_c, axis=1)+1
    return np.concatenate((dataset.inds.reshape(-1,1),Y.reshape(-1,1)), axis=1)
  def predict_(self, dataset, feedback_round=0, p1=1, p2=1):
    if feedback_round == 0:
      return self.predict(dataset)
    X = np.zeros((len(dataset), len(self.dictionary)))
    X_tmp = np.zeros((len(dataset), len(self.dictionary)))
    for i, (id, x, y) in enumerate(dataset):
        X[i, np.isin(self.dictionary, x[:,0])] = x[np.isin(x[:,0], self.dictionary), 1]
    Y_tmp = np.argmax((X @ self.log_P_tc) + self.log_P_c, axis=1)+1
    Y = np.zeros((len(dataset), self.class_size))
    for i, y in enumerate(Y_tmp):
        Y[i, int(y-1)] = 1
    mat = X.T @ Y + .1
    log_P_c = np.log((np.sum(Y,axis=0)+1) / (Y.shape[0]+Y.shape[1]))
    log_P_tc = np.log(mat / np.sum(mat,axis=0))
    self.log_P_c = self.log_P_c*(1-p1)+ log_P_c*p1
    self.log_P_tc = self.log_P_tc*(1-p2)+ log_P_tc*p2
    return self.predict_(dataset, feedback_round=feedback_round-1, p1=p1, p2=p2)

def F1(pred, y, smooth=True):
  macro_p = 0
  macro_r = 0
  micro_tp = 0
  micro_fp = 0
  micro_fn = 0
  for c in range(1,Config.class_size+1):
    tp = np.sum((pred == c) & (y == c))
    fp = np.sum((pred == c) & (y != c))
    fn = np.sum((pred != c) & (y == c))
    # print(f'tp: {tp} / fp: {fp} / fn: {fn}')
    if smooth:
      p = (tp+np.finfo(float).eps) / (tp+fp+2*np.finfo(float).eps)
      r = (tp+np.finfo(float).eps) / (tp+fn+2*np.finfo(float).eps)
    else:
      p = tp / (tp+fp)
      r = tp / (tp+fn)
    macro_p += p
    macro_r += r
    micro_tp += tp
    micro_tp += fp
    micro_fn += fn
  macro_p /= Config.class_size
  macro_r /= Config.class_size
  macro_f1 = 2 * macro_p * macro_r / (macro_p + macro_r)
  if smooth:
    micro_p = (micro_tp+np.finfo(float).eps) / (micro_tp+micro_fp+2*np.finfo(float).eps)
    micro_r = (micro_tp+np.finfo(float).eps) / (micro_tp+micro_fn+2*np.finfo(float).eps)
  else:
    micro_p = micro_tp / (micro_tp+micro_fp)
    micro_r = micro_tp / (micro_tp+micro_fn)
  micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r)
  return macro_f1, micro_f1

def evaluate(clf, dataset):
  prediction = clf.predict_(dataset, feedback_round=Config.feedback_round, p1=Config.p1, p2=Config.p2)
  return F1(prediction[:,1], dataset.y.reshape(-1))

def train_test_split(dataset, train_permute, test_permute):
  return dataset.get_subset(train_permute), dataset.get_subset(test_permute)

def cross_validation(clf, dataset, fold=10, permutation=True, seed=None):
  if seed:
    np.random.seed(seed)
    random.seed(seed)
  n = len(dataset)
  if permutation:
    permute = np.random.permutation(n)
  else:
    permute = np.arange(n)
  width = int(np.ceil(n / fold))
  left = 0
  right = width
  ma_train_f1 = 0
  mi_train_f1 = 0
  ma_valid_f1 = 0
  mi_valid_f1 = 0
  for i in range(fold):
    train_permute = np.concatenate((permute[:left], permute[right:]))
    test_permute = permute[left:right]
#     test_permute = np.concatenate((permute[:left], permute[right:]))
#     train_permute = permute[left:right]
    left += width
    right += width
    dataset_train, dataset_test = train_test_split(dataset, train_permute, test_permute)
    clf.train(dataset_train)
    matf1, mitf1 = evaluate(clf, dataset_train)
    mavf1, mivf1 = evaluate(clf, dataset_test)
    ma_train_f1 += matf1
    mi_train_f1 += mitf1
    ma_valid_f1 += mavf1
    mi_valid_f1 += mivf1
    print(f'cross validation:[{i+1}/{fold}] macro F1 (train): {matf1} / micro F1 (train): {mitf1} / macro F1 (valid): {mavf1} / micro F1 (valid): {mivf1}', end='\r')
  print('')
  ma_train_f1 /= fold
  mi_train_f1 /= fold
  ma_valid_f1 /= fold
  mi_valid_f1 /= fold
  return ma_train_f1, mi_train_f1, ma_valid_f1, mi_valid_f1

def save_prediction(clf, dataset, filename=Config.output_file):
  prediction = clf.predict_(dataset, feedback_round=Config.feedback_round, p1=Config.p1, p2=Config.p2)
  np.savetxt(filename, prediction, fmt='%i', delimiter=',', header='Id,Value', comments='')

def save_dictionary(dictionary, filename):
  with open(filename, 'w') as file:
    file.write('\n'.join([str(t) for t in dictionary]))

if __name__ == "__main__":
  """
  Step 1. Create dictionary and binary vectors
  (Here we can specify the corpus of dictionary by index (e.g., dataset_triain.inds))
  """
  # y = read_training()
  # init(np.where(y != -1)[0] + 1)
  dataset_train, dataset_test = get_dataset()
  """
  Step 2. Feature Selection with training set
  """
  terms = read_dict(Config.dictionary_file)
  # dictionary = max_chi2(terms, dataset_train, size=200)
  # limit =[27., 37., 31., np.inf, 51., np.inf, np.inf, 32., np.inf, 29., np.inf, np.inf, 46.]
  # limit = [np.inf] * 13
  # limit[0] = 0
  # dictionary = max_seg_chi2(terms, dataset_train, limit=limit)
  # dictionary = max_seg_chi2(terms, dataset_train)
  dictionary = max_seg_chi2(terms, dataset_train, size=150)
  """
  Step 3. Estimate P_tc, P_c with training set
  """
  clf = Classifier(dictionary)
  # ma_train_f1, mi_train_f1, ma_valid_f1, mi_valid_f1 = cross_validation(clf, dataset_train, seed=1126)
  # print(f'average macro F1 (train): {ma_train_f1} / average micro F1 (train): {mi_train_f1}')
  # print(f'average macro F1 (valid): {ma_valid_f1} / average micro F1 (valid): {mi_valid_f1}')
  # random.seed(1126)
  # np.random.seed(1126)
  # ttl = np.zeros((4))
  # n = 10
  # for i in range(n):
  #   r = cross_validation(clf, dataset_train, fold=6)
  #   ttl += np.array(r)
  # ttl /= n
  # print(ttl)
  """
  Step 4. Predict testing set with P_tc, P_c
  """
  clf.train(dataset_train)
  save_prediction(clf, dataset_test)