# pg. 185

wsj = nltk.corpus.treebank.tagged_words(simplify_tags=True)
word_tag_fd = nltk.FreqDist(wsj)
[word + "/" + tag for (word, tag) in word_tag_fd if tag.startswith('V')]

cfd1 = nltk.ConditionalFreqDist(wsj)
cfd1['yield'].keys()
cfd1['cut'].keys()

# pg. 186

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

# pg. 201



