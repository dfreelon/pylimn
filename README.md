# PyLimn
 
## A small Python module for NLP-based text description

PyLimn implements a few NLP-type functions I've found useful in my own research. YMMV.

## System requirements

* [geostring](https://github.com/dfreelon/geostring)
* [nltk](https://www.nltk.org/)

## Installation
```pip install pylimn```

## Overview

PyLimn contains the following functions:

* ```rank_named_entities```: Collects and ranks named entities by frequency from a corpus of documents.
* ```kwic```: Generates keyword-in-context strings from a single document.
* ```pairwise_stem```: Generates a pairwise stem from two input strings.
* ```pairwise_stem_all```: Generates pairwise stems from a corpus of input strings.
* ```get_context_terms```: Collects and ranks unigrams and bigrams from a list of keyword-in-context strings generated by ```kwic```. 

### rank_named_entities

Sample code:

```python
import pandas as pd
import pylimn as pyl

docs = pd.read_csv("docfile.csv") # CSV file containing multiple documents
doc_list = docs.DOC_HEADER.tolist()
docs_ne = pyl.rank_named_entities(doc_list,
                                  min_entity_ct=50,
                                  min_entity_len=5) #adjust numbers based on the size of your corpus
print(docs_ne[:10]) #show top ten most frequently-occurring named entities
```
### kwic

Sample code:

```python
huck = '''
YOU don't know about me without you have read a book by the name of The
Adventures of Tom Sawyer; but that ain't no matter.  That book was made
by Mr. Mark Twain, and he told the truth, mainly.  There was things
which he stretched, but mainly he told the truth.  That is nothing.  I
never seen anybody but lied one time or another, without it was Aunt
Polly, or the widow, or maybe Mary.  Aunt Polly--Tom's Aunt Polly, she
is--and Mary, and the Widow Douglas is all told about in that book, which
is mostly a true book, with some stretchers, as I said before.
''' #from The Adventures of Huckleberry Finn, https://www.gutenberg.org/files/76/76-0.txt
pyl.huck_kwic = kwic(huck,'Tom')
print(huck_kwic)
```

### pairwise_stem

Sample code:

```python
t1 = 'nationalist'
t2 = 'nationalism'
nat_ps = pyl.pairwise_stem(t1,t2)
print(nat_ps)
```

### pairwise_stem_all

Sample code:

```python
nat_list = ['nation','national','nationalism','nationalist','nationality']
nat_psa = pyl.pairwise_stem_all(nat_list)
print(nat_psa)
```

### get_context_terms

Sample code:

```python
#see above sample code for rank_named_entities to get the doc_list var
kwic_list = []
kw = 'trump'
for i in doc_list:
    if kw in i:
        kwic_list.extend(pyl.kwic(i,kw)[0])

trump_ct = pyl.get_context_terms(kwic_list,
                                 min_entity_ct=50,
                                 min_entity_len=5) #adjust numbers based on the size of your corpus
print(trump_ct['unigrams'][:10]) #show top ten most frequently-occurring contextual unigrams, minus stopwords
print(trump_ct['bigrams'][:10]) #show top ten most frequently-occurring contextual bigrams, minus stopwords
```
