import pa1
import os  # for listing file under folder
import pandas as pd
import numpy as np

class Config(object):
  # input_folder = 'IRTM'
  input_file = '../data/irtm.csv'
  dict_file = 'dictionary.txt'
  binary_vector_folder = 'binary_vector'
  DOCx = 'doc1.txt'
  DOCy = 'doc2.txt'

def get_input_file(filenames=None):
  return [os.path.join(Config.input_folder, f)
          for f in os.listdir(Config.input_folder)
          if '.txt' in f and
          (filenames is None or f in filenames) and
          os.path.isfile(os.path.join(Config.input_folder, f))]

def get_token(text):
#     text  = pa1.read_from_file(file)
    token = pa1.tokenize(text)
    token = pa1.lower(token)
    # token = pa1.remove_stopword(token)
    token = pa1.stemming(token)
    token = pa1.remove_stopword(token)
    return token

def make_dict(input_files):
  result = {}
  N = len(input_files)
  for i, text in enumerate(input_files):
    print(f'\rdictionary processing: [{i+1}/{N}]', end='')
    token = get_token(str(text))
    for t in set(token):
      if t in result:
        result[t][0] += 1
        result[t][1] += token.count(t)
      else:
        result[t] = [1, token.count(t)]
  print()
  return result

def save_dict(result, output_file):
  with open(output_file, 'w') as file:
    file.write('\n'.join([' '.join([str(i+1), k, str(df), str(tf)])
               for i, (k, (df, tf)) in enumerate(sorted(result.items()))]))

def create_dict(inds = None):
  input_files = get_input_file([f'{i}.txt' for i in inds] if inds is not None else None)
  result = make_dict(input_files)
  save_dict(result, Config.dict_file)

def read_dict(filename):
  with open(filename, 'r') as file:
    terms = [t.split(' ') for t in file.read().split('\n')]
  terms =  {int(id): [term, int(df), int(tf)] for id, term, df, tf in terms}
  return sorted(terms.items())

def save_vector(result, output_file):
  # content = f'{len(result)}\n'+'\n'.join(result)
  content = f'{len(result)}\n'+'\n'.join([' '.join(r) for r in result])
  with open(output_file, 'w') as file:
    file.write(content)
#   if 'doc1.txt' in output_file:
#     with open('doc1.txt', 'w') as file:
#       file.write(content)

def create_vector(input_files, indexes, filename=Config.dict_file):
  if not os.path.exists(Config.binary_vector_folder):
    os.makedirs(Config.binary_vector_folder)
  terms = read_dict(filename)
#   input_files = get_input_file()
  N = len(input_files)
  for i, file in enumerate(input_files):
    print(f'\rbinary vector processing: [{i+1}/{N}]', end='')
    # print(f'\rtf-idf processing: [{i+1}/{N}]', end='')
    token = get_token(str(file))
    result = []
    if token:
      for id, (term, df, tf) in terms:
        if term in token:
          tf = token.count(term)
          result.append([str(id), str(tf)])
      #   tf = token.count(term) / len(token)
      #   if tf > 0:
      #     result.append([id, tf * np.log10(N / df)])
    # else:
    #   result = [[]]
    # result = np.array(result)
    # length = np.sqrt(np.sum(result[:, 1] ** 2))
    # if length > 0:
    #   result[:, 1] /= length
    # result = [[str(int(id)), str(tfidf)] for id, tfidf in result]
    output_file = os.path.join(Config.binary_vector_folder,
                  'doc' + str(indexes[i]) + '.txt')
    save_vector(result, output_file)
  print()

def read_vector(filename):
  with open(os.path.join(Config.binary_vector_folder, filename), 'r') as file:
    # return np.array([int(id) for id in file.read().split()[1:]])
    lines = [l.split(' ') for l in file.read().split('\n')]
    return np.array([[int(id), float(tf)] for id, tf in lines[1:]])
    # return np.array([[int(id), float(tfidf)] for id, tfidf in lines[1:]])

# def cosine(DOCx, DOCy):
#   x = read_vector(DOCx)
#   y = read_vector(DOCy)
#   term_id = list(set(x[:, 0]) & set(y[:, 0]))
#   return x[np.in1d(x[:,0], term_id), 1] @ y[np.in1d(y[:,0], term_id), 1]

if __name__ == "__main__":
  create_dict()
  create_vector()
  # print(f'cosine simularity of {Config.DOCx} & {Config.DOCy}: {cosine(Config.DOCx, Config.DOCy)}')