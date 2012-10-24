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

# pg. 187

def findtags(tag_prefix, tagged_text):
  cfd = nltk.ConditionalFreqDist((tag, word) for (word, tag) in tagged_text
    if tag.startswith(tag_prefix))
    
  return dict((tag, cfd[tag].keys()[:5]) for tag in cfd.conditions())
  
tagdict = findtags('NN', nltk.corpus.brown.tagged_words(categories='news'))
for tag in sorted(tagdict):
  print tag, tagdict[tag]

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
t0 = nltk.DefaultTagger('NN')
t1 = nltk.UnigramTagger(train_sents, backoff=t0)
t2 = nltk.BigramTagger(train_sents, backoff=t1)
print t2.evaluate(test_sents)