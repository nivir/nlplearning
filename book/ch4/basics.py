# CH. 4
empty = []
nested = [empty, empty, empty]
print nested
nested[1].append('Python')
print nested

nested = [[]] * 3
print nested
print id(nested[0]), id(nested[1]), id(nested[2])
nested[1].append('Python')
print nested
nested[1] = ['Monty']
print nested

bar = nested[:]
print bar

#print bar.deepcopy()

# is operator tests for object identity
size = 5
python = ['Python']
snake_test = [python] * size
# Not only are they equal they are identical
print snake_test[0] == snake_test[1] == snake_test[2] == snake_test[3] == snake_test[4]

print snake_test[0] is snake_test[1] is snake_test[2] is snake_test[3] is snake_test[4]

# But they are not always equal!
import random
position = random.choice(range(size))
snake_test[position] = ['Python']
snake_test

print snake_test[0] == snake_test[1] == snake_test[2] == snake_test[3] == snake_test[4]

print snake_test[0] is snake_test[1] is snake_test[2] is snake_test[3] is snake_test[4]
print [id(snake) for snake in snake_test]

print "########### Conditionals ############"

mixed = ['cat', '', ['dog'], []]
for element in mixed:
  if element:
    print element
    
sent = ['No', 'goods', 'fish', 'goes']
print all(len(w) > 4 for w in sent)

print any(len(w) > 4 for w in sent)

print "########### Sequences ############"

# So far we have seen two types of sequences strings and lists, another kind is tuples

t = 'walk', 'fem', 3
print t

print t[0]
print t[1:]

print ('snake',)

raw = 'I turned off the spectroroute'
text = ['I', 'turned', 'off', 'the', 'spectroroute']
pair = (6, 'turned')
print (raw[2], text[3], pair[1])

blah = set(text)
# The below will fail, you cannot index into a set
# print blah[1]

# pg. 134
# s = range(1,4)

# for item in s

# for item in sorted(s)

# for item in set(s)

# for item in reversed(s)

# for item in set(s).difference(t)

# for item in random.shuffle(s)
import nltk

raw = 'Red lorry, yellow lorry, red lorry, yellow lorry.'
text = nltk.word_tokenize(raw)
fdist = nltk.FreqDist(text)
list(fdist)

for key in fdist:
  print fdist[key],

words = ['I', 'turned', 'off', 'the', 'spectoroute']
words[2], words[3], words[4] = words[3], words[4], words[2]
words

tmp = words[2]
words[2] = words[3]
words[3] = words[4]
words[4] = tmp

words 

words = ['I', 'turned', 'off', 'the', 'spectoroute']
tags = ['noun', 'verb', 'prep', 'det', 'noun']
print zip(words, tags)

# pg. 136

print list(enumerate(words))

words = 'I turned off the spectroroute'.split()
wordlens = [(len(word), word) for word in words]
wordlens.sort()
' '.join(w for (_,w) in wordlens)

lexicon = [
  ('the', 'det', ['Di:', 'D@']),
  ('off', 'prep', ['Qf', 'O:f'])
]

lexicon.sort()
lexicon[1] = ('turned', 'VBD', ['t3:nd', 't3nd'])
#del lexicon[0]
# lists are mutable and tuples are immutable

# Not in the book but this is one way to set breakpoints
###import pdb; pdb.set_trace()

# This is not using a generator expression and the entire list must be built before passing it off to max
text = "John is so cool, that he went to the park to buy a kitten."
#max([w.lower() for w in nltk.word_tokenize(text)])

# This is using a generator expression and allows the input to be streamed
#max(w.lower() for w in nltk.word_tokenize(text))

# pg. 138

#fd = nltk.FreqDist(nltk.corpus.brown.words())
#cumulative = 0.0
#for rank, word in enumerate(fd):
#  cumulative += fd[word] * 100 / fd.N()
#  print "%3d %6.2f%% %s" % (rank+1, cumulative, word)
#  if cumulative > 25:
#    break
    
text = nltk.corpus.gutenberg.words('milton-paradise.txt')
longest = ''
for word in text:
  if len(word) > len(longest):
    longest = word
    
longest

maxlen = max(len(word) for word in text)
print maxlen
print [word for word in text if len(word) == maxlen]

# pg. 141

sent = ['The', 'dog', 'gave', 'John', 'the', 'newspaper']
n = 3
[sent[i:i+n] for i in range(len(sent)-n+1)]

import pprint
m, n = 3, 7
array = [[set() for i in range(n)] for j in range(m)]
array[2][5].add('Alice')
pprint.pprint(array) 

# pg. 142

def blah():
  "Blah does blah."
  print "blah"
  
# help(blah)

# parameter passing, call by value ,...

# Variable scope, LGB, local, then global, then builtin

# pg. 146

def tag(word):
  assert isinstance(word, basestring), "argument to tag() must be a string"
  if word in ['a', 'the', 'all']:
    return 'det'
  else:
    return 'noun'
    
tag("the")

# tag(["asd"]) Will cause error

# Docstrings and the doc test block pg. 148 - 149

def extract_property(prop):
  return [prop(word) for word in sent]
  
print extract_property(len)
print extract_property(lambda w: w[-1])

print sorted(sent, lambda x, y: cmp(len(y), len(x)))

def search1(substring, words):
  result = []
  for word in words:
    if substring in word:
      result.append(word)
  return result
  
def search2(substring, words):
  result = []
  for word in words:
    if substring in word:
      yield word

#print "search1:"
#for item in search1('zz', nltk.corpus.brown.words()):
#  print item

  
#print "search2:"
#for item in search2('zz', nltk.corpus.brown.words()):
#  print item

def permutations(seq):
  if len(seq) <= 1:
    yield seq
  else:
    for perm in permutations(seq[1:]):
      for i in range(len(perm)+1):
        yield perm[:i] + seq[0:1] + perm[i:]
        
print list(permutations(['police', 'fish', 'buffalo']))

lengths = map(len, nltk.corpus.brown.sents(categories='news'))
sum(lengths) / len(lengths)

print map(lambda w: len(filter(lambda c: c.lower() in "aeiou", w)), sent)

print [len([c for c in w if c.lower() in "aeiou"]) for w in sent]

def repeat(msg='<empty>', num=1):
  return msg * num

print repeat(num=3)

print repeat(msg='Alice')

def generic(*args, **kwargs):
  print args
  print kwargs
  
generic(1, "African swallow", monty="python")

song = [["Four", "calling", "birds"],
        ["Three", "french", "hens"],
        ["Two", "turtle", "doves"]
]

print zip(*song)

def freq_words(file, min=1, num=10):
  text = open(file).read()
  
#freq_words('ch01.rst', 4, 10)

# pg. 154

nltk.metrics.distance.__file__

# pg. 158

#import pdb
#import mymodule
#pdb.run('mymodule.myfunction()')

def size1(s):
  return 1 + sum(size1(child) for child in s.hyponyms())
  
def size2(s):
  layer = [s]
  total = 0
  while layer:
    total += len(layer)
    layer = [h for c in layer for h in c.hyponyms()]
  return total
  
from nltk.corpus import wordnet as wn
dog = wn.synset('dog.n.01')
print size1(dog)

print size2(dog)

def insert(trie, key, value):
  if key:
    first, rest = key[0], key[1:]
    if first not in trie:
      trie[first] = {}
    insert(trie[first], rest, value)
  else:
    trie['value'] = value
    
trie = nltk.defaultdict(dict)
insert(trie, 'chat', 'cat')
insert(trie, 'chien', 'dog')
insert(trie, 'chair', 'flesh')
insert(trie, 'chic', 'stylish')
trie = dict(trie) # for nicer printing
print trie['c']['h']['a']['t']['value']

pprint.pprint(trie)

import re
def raw(file):
  contents = open(file).read()
  contents = re.sub(r'<.*?>', ' ', contents)
  contents = re.sub(r'\s+', ' ', contents)
  return contents

def snippet(doc, term):
  text = ' '*30 + raw(doc) + ' '*30
  pos = text.index(term)
  return text[pos-30:pos+30]
  
print "Building Index..."
import nltk
files = nltk.corpus.movie_reviews.abspaths()
idx = nltk.Index((w,f) for f in files for w in raw(f).split())

#query = ''
#while query != "quit":
#  query = raw_input("query> ")
#  if query in idx:
#    for doc in idx[query]:
#      print snippet(doc, query)
#  else:
#    print "Not found"
  
def preprocess(tagged_corpus):
  words = set()
  tags = set()
  for sent in tagged_corpus:
    for word, tag in sent:
      words.add(word)
      tags.add(tag)
  wm = dict((w,i) for (i,w) in enumerate(words))
  tm = dict((t,i) for (i,t) in enumerate(tags))
  return [[(wm[w], tm[t]) for (w,t) in sent] for sent in tagged_corpus]
  
from timeit import Timer
vocab_size = 100000
setup_list = "import random; vocab = range(%d)" % vocab_size
setup_set = "import random; vocab = set(range(%d))" % vocab_size
statement = "random.randint(0,%d) in vocab" % vocab_size 
print Timer(statement, setup_list).timeit(1000)
print Timer(statement, setup_set).timeit(1000)

def virahanka1(n):
  if n == 0:
    return [""]
  elif n == 1:
    return ["S"]
  else:
    s = ["S" + prosody for prosody in virahanka1(n-1)]
    l = ["L" + prosody for prosody in virahanka1(n-2)]
    return s + l

def virahanka2(n):
  lookup = [[""], ["S"]]
  for i in range(n-1):
    s = ["S" + prosody for prosody in lookup[i+1]]
    l = ["L" + prosody for prosody in lookup[i]]
    lookup.append(s + l)

def virahanka3(n):
  print "too lazy to type"
    
from nltk import memoize
@memoize
def virahanka4(n):
  if n == 0:
    return [""]
  elif n == 1:
    return ["S"]
  else:
    s = ["S" + prosody for prosody in virahanka4(n-1)]
    l = ["L" + prosody for prosody in virahanka4(n-2)]
    return s + l
    
print virahanka1(4)    
print virahanka2(4)
print virahanka4(4)


# pg. 170

import networkx as nx
import matplotlib
from nltk.corpus import wordnet as wn

def traverse(graph, start, node):
  graph.depth[node.name] = node.shortest_path_distance(start)
  for child in node.hyponyms():
    graph.add_edge(node.name, child.name)
    traverse(graph, start, child)
    
def hyponym_graph(start):
  G = nx.Graph()
  G.depth = {}
  traverse(G, start, start)
  return G

def graph_draw(graph):
  nx.draw_graphviz(graph,
    node_size = [16 * graph.degree(n) for n in graph],
    node_color = [graph.depth[n] for n in graph],
    with_labels = False)
  matplotlib.pyplot.show()
  
dog = wn.synset('dog.n.01')
graph = hyponym_graph(dog)
graph_draw(graph)



  