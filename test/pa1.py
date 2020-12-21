# pa1
from nltk.stem import PorterStemmer
# from nltk.corpus import stopwords
class Config(object):
  target_file = '28.txt'
  stopwords_file = 'stopwords.txt'
  result_file = 'result.txt'
  seperator = [
    ' ',
    '_',
    '\'s ', 's\' ', '\'ll ', '\'re ', '\'d ', '\'m ', '\'ve ',
    # '\'s\n', 's\'\n', '\'ll\n', '\'re\n', '\'d\n', '\'m\n', '\'ve\n',
    # '\n\'', '\'\n', '\n\'\'', '\'\'\n', '\n"', '"\n',
    ' \'', '\' ', ' \'\'', '\'\' ', ' "', '" ',
    '. ', ', ', ' ,', ' \'\'', '\'\' ', ' "', '" ', '?', '!', ';', '\n', '\t', '(', ')', '~', ':', '`',
    '(', ')', '[', ']', '{', '}', '*', '/', '#', '@',
    # '1','2','3','4','5','6','7','8','9','0'
    '<', '>', '=', '█', '∞', '&', '+', '°', '▒', '░', '…', '\\', '^', '”', '’', '’', '—', '®', '™', '‘', 'º', '▒҉͝', '■', '¤', '◘', '╬', '“'
  ]
  ignore_list = [
    ',', '$', '%',
    '-',
    # '.'
  ]
  ignore_2_list = [
    '\'', '"',
    '&amp'
  ]

def has_number(s):
  return sum([c in ['1','2','3','4','5','6','7','8','9','0'] for c in s])

def read_from_file(filename):
  with open(filename, 'r') as file:
    return file.read()

def ignore(text):
  for i in Config.ignore_list:
    text = text.replace(i, '')
  return text

def ignore_2(text):
  for i in Config.ignore_2_list:
    text = text.replace(i, '')
  return text

def tokenize(text):
  if Config.seperator:
    text = ' ' + text + ' '
    # text.replace('\n', ' \n')
    text = text.replace('\n', ' ')
    text = text.replace(' ', '  ')
    text = text.replace('www.', 'www ')
    text = text.replace('.org', ' org')
    text = text.replace('.com', ' com')
    text = text.replace('.mil', ' mil')
    text = text.replace('.htm', ' htm')
    sep = Config.seperator[0]
    for s in Config.seperator[1:]:
      text = text.replace(s, sep)
    return [ignore(t) for t in text.split(sep) if not has_number(t) and ignore(t)]
    # return [ignore(t) for t in text.split(sep) if ignore(t)]
  return [text] if text else []

def lower(token):
  return list(map(str.lower, token))

def stemming(token):
  ps = PorterStemmer()
  # return list(map(ps.stem, token))
  return [w.replace('.', '') for w in list(map(ps.stem, token)) if w.replace('.', '')]

ps = PorterStemmer()
with open(Config.stopwords_file) as file:
  stopwords = file.read().split()
stopwords = list(map(ps.stem, stopwords))
# stops = set(stopwords.words("english")) 
# stops = list(map(ps.stem, stops))

def remove_stopword(token):
#   return [ignore_2(w) for w in token if ignore_2(w) and ignore_2(w) not in stops]
  return [ignore_2(w) for w in token if ignore_2(w) and ignore_2(w) not in stopwords]

def save_to_file(token):
  with open(Config.result_file, 'w') as file:
    file.write(' '.join(token))

if __name__ == "__main__":
  text = read_from_file(Config.target_file)
  token = tokenize(text)
  token = lower(token)
  token = stemming(token)
  token = remove_stopword(token)
  save_to_file(token)