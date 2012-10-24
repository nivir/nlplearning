import nltk

text = nltk.word_tokenize("And now for something completely different")
print nltk.pos_tag(text)

# pg. 180

print nltk.help.upenn_tagset('RB')
print nltk.help.upenn_tagset('NN.*')

text = nltk.word_tokenize("They refuse to permit us to obtain the refuse permit")
print nltk.pos_tag(text)

text = nltk.Text(word.lower() for word in nltk.corpus.brown.words())
text.similar('woman')

text.similar('bought')

text.similar('over')

text.similar('the')

# pg. 181

tagged_token = nltk.tag.str2tuple('fly/NN')
tagged_token

tagged_token[0]
tagged_token[1]

sent = '''
  The/AT grand/JJ jury/NN commented/VBD on/IN a/AT number/NN of/IN
'''

print [nltk.tag.str2tuple(t) for t in sent.split()]
  
  
# pg. 182

print nltk.corpus.brown.tagged_words(simplify_tags=True)

print nltk.corpus.treebank.tagged_words(simplify_tags=True)





