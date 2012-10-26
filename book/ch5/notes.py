import nltk

# pg. 184

from nltk.corpus import brown
brown_news_tagged = brown.tagged_words(categories='news', simplify_tags=True)
tag_fd = nltk.FreqDist(tag for (word, tag) in brown_news_tagged)
print tag_fd.keys()

nltk.app.concordance()

# pg. 185

word_tag_pairs = nltk.bigrams(brown_news_tagged)
list(nltk.FreqDist(a[1] for (a,b) in word_tag_pairs if b[1] == 'N'))

wsj = nltk.corpus.treebank.tagged_words(simplify_tags=True)
word_tag_fd = nltk.FreqDist(wsj)
[word + "/" + tag for (word,tag) in word_tag_fd if tag.startswith('V')]

cfd1 = nltk.ConditionalFreqDist(wsj)
cfd1['yield'].keys()
cfd1['cut'].keys()

cfd2 = nltk.ConditionalFreqDist((tag,word) for (word,tag) in wsj)
cfd2['VN'].keys()

# pg. 186

[w for w in cfd1.conditions() if 'VD' in cfd1[w] and 'VN' in cfd1[w]]
idx1 = wsj.index(('kicked', 'VD'))
print wsj[idx1-4:idx1+1]
idx2 = wsj.index(('kicked', 'VN'))
print wsj[idx2-4:idx2+1]

cfd2 = nltk.ConditionalFreqDist((tag, word) for (word, tag) in wsj)
cfd2['VN'].keys()

# pg. 187

def findtags(tag_prefix, tagged_text):
    cfd = nltk.ConditionalFreqDist((tag, word) for (word, tag) in tagged_text if tag.startswith(tag_prefix))
    return dict((tag, cfd[tag].keys()[:5]) for tag in cfd.conditions())

tagdict = findtags('NN', nltk.corpus.brown.tagged_words(categories='news'))
for tag in sorted(tagdict):
	print tag, tagdict[tag] 

from nltk.corpus import brown
brown_learned_text = brown.words(categories='learned')
sorted(set(b for (a, b) in nltk.ibigrams(brown_learned_text) if a == 'often'))

# pg. 188

brown_lrnd_tagged = brown.tagged_words(categories='learned', simplify_tags=True)
tags = [b[1] for (a, b) in nltk.ibigrams(brown_lrnd_tagged) if a[0] == 'often']
fd = nltk.FreqDist(tags)
fd.tabulate()

from nltk.corpus import brown
def process(sentence):
	for (w1, t1), (w2,t2), (w3,t3) in nltk.trigrams(sentence):
		if (t1.startswith('V') and t2 == 'TO' and t3.startswith('V')):
			print w1, w2, w3

for tagged_sent in brown.tagged_sents():
	process(tagged_sent)

brown_news_tagged = brown.tagged_words(categories='news', simplify_tags=True)
data = nltk.ConditionalFreqDist((word.lower(), tag)
                                for (word, tag) in brown_news_tagged)
for word in data.conditions():
     if len(data[word]) > 3:
         tags = data[word].keys()
         print word, ' '.join(tags)

# pg. 189 - pg. 193 All about dictionaries and their basics


# pg. 194

# caroll-alice can't be found
alice = nltk.corpus.gutenberg.words('caroll-alice.txt')
vocab = nltk.FreqDist(alice)
v1000 = list(vocab)[:1000]
mapping = nltk.defaultdict(lambda: 'UNK')
for v in v1000:
	mapping[v] = v

alice2 = [mapping[v] for v in alice]
alice2[:100]

# pg. 195

counts = nltk.defaultdict(int)
from nltk.corpus import brown
for (word, tag) in brown. tagged_words(categories='news'):
	counts[tag] += 1

counts['N']

from operator import itemgetter
sorted(counts.items(), key=itemgetter(1), reverse=True)

pair = ('NP', 8336)
pair[1]

itemgetter(1)(pair)

# pg. 196

words = nltk.corpus.words.words('en')

anagrams = nltk.defaultdict(list)

for word in words:
	key = ''.join(sorted(word))
	anagrams[key].append(word)

anagrams['aeilnrt'] 

# Two equivalent ways, below is more concise

anagrams = nltk.Index((''.join(sorted(w)), w) for w in words)
anagrams['aeilnrt'] 

pos = nltk.defaultdict(lambda: nltk.defaultdict(int))
brown_news_tagged = brown.tagged_words(categories='news', simplify_tags=True)
for ((w1, t1), (w2, t2)) in nltk.ibigrams(brown_news_tagged): 
    pos[(t1, w2)][t2] += 1 

pos[('DET', 'right')]

# pg. 197

pos = {'colorless' : 'ADJ', 'ideas' : 'N', 'sleep' : 'V', 'furiously' : 'ADV'}
pos.update({'peacefully' : 'ADV'})

pos2 = nltk.Index((value, key) for (key, value) in pos.items())
pos2['ADV']

# pg. 198 Default Tagger blah blah
brown_tagged_sents = brown.tagged_sents(categories='news')
# pg. 199 Regular Expression Tagger more useful

# pg. 200 Lookup Tagger and backoff

fd = nltk.FreqDist(brown.words(categories='news'))
cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
most_freq_words = fd.keys()[:100]
likely_tags = dict((word, cfd[word].max()) for word in most_freq_words)
baseline_tagger = nltk.UnigramTagger(model=likely_tags)
baseline_tagger.evaluate(brown_tagged_sents)

sent = brown.sents(categories='news')[3]
baseline_tagger.tag(sent)

baseline_tagger = nltk.UnigramTagger(model=likely_tags,
                                     backoff=nltk.DefaultTagger('NN'))

# pg. 201 - pg. 202 # Talks about performance of Unigram Tagger varying with model size

# pg. 203

from nltk.corpus import brown
brown_tagged_sents = brown.tagged_sents(categories='news')
brown_sents = brown.sents(categories='news')
unigram_tagger = nltk.UnigramTagger(brown_tagged_sents)
unigram_tagger.tag(brown_sents[2007])
unigram_tagger.evaluate(brown_tagged_sents)

size = int(len(brown_tagged_sents) * 0.9)
size
train_sents = brown_tagged_sents[:size]
test_sents = brown_tagged_sents[size:]
unigram_tagger = nltk.UnigramTagger(train_sents)
print unigram_tagger.evaluate(test_sents)

# pg. 204
bigram_tagger = nltk.BigramTagger(train_sents)
bigram_tagger.tag(brown_sents[2007])

unseen_sent = brown_sents[4203]
bigram_tagger.tag(unseen_sent)
bigram_tagger.evaluate(test_sents)

# pg. 205

# Precision/Recall Tradeoff - tradeoff between accuracy and coverage of our results

t0 = nltk.DefaultTagger('NN')
t1 = nltk.UnigramTagger(train_sents, backoff=t0)
t2 = nltk.BigramTagger(train_sents, backoff=t1)
print t2.evaluate(test_sents)

t2 = nltk.BigramTagger(train_sents, cutoff=2, backoff=t1)

# pg. 206

from cPickle import dump
output = open('t2.pkl', 'wb')
dump(t2, output, -1)
output.close()

from cPickle import load
input = open('t2.pkl', 'rb')
tagger = load(input)
input.close()

text = """The board's action shows what free enterprise
     is up against in our complex maze of regulatory laws ."""
tokens = text.split()
tagger.tag(tokens)

# pg. 207

cfd = nltk.ConditionalFreqDist(
            ((x[1], y[1], z[0]), z[1])
            for sent in brown_tagged_sents
            for x, y, z in nltk.trigrams(sent))
ambiguous_contexts = [c for c in cfd.conditions() if len(cfd[c]) > 1]
sum(cfd[c].N() for c in ambiguous_contexts) / cfd.N()

test_tags = [tag for sent in brown.sents(categories='editorial')
                  for (word, tag) in t2.tag(sent)]
gold_tags = [tag for (word, tag) in brown.tagged_words(categories='editorial')]
print nltk.ConfusionMatrix(gold, test)   

# pg. 208 - pg. 209

# Brill tagging is a kind of Transformation-based learning
#   The general idea is guess the tag of each word, then go back and fix mistakes
#   However unlike n-gram tagging it does not count observations but compiles a list
#     of transformational correction rules.

# pg. 210

nltk.brill.demo()

print(open("errors.out").read())

# pg. 211 - pg. 212 - end of chapter










