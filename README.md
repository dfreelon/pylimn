# PyLimn
 
PyLimn implements several NLP-type functions I've found useful in my own research. YMMV.

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

### ```rank_named_entities```

__Sample code__

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

__Parameters__

* ```news_iterable```: a list-like object containing strings (preferably news-article length ones)
* ```min_entity_ct```: either 1) the minimum number of times a named entity can appear in a dataset, or 2) the minimum number of articles in which an entity can appear (which of these it is depends on the value of ```once_per_doc```). Default is 5.
* ```min_entity_len```: the minimum character length for a named entity. Default is 4.
* ```stop_words```: a list-like objects of words to exclude from the analysis
* ```include_first_words```: Boolean indicating whether the first words of sentences should be included in the analysis. Default is ```False```.
* ```remove_upper_terms```: Boolean indicating whether terms in all caps should be removed. Default is ```False```.
* ```find_hyphenated```: Boolean indicating whether terms containing hyphens that may not necessarily have their first character capitalized should be included. Default is ```True```.
* ```once_per_doc```: Boolean indicating whether terms should be counted once per document or the total number of times they appear across all documents. Default is ```True```.
* ```remove_dates```: Boolean indicating whether to remove date-related information. Default is ```True```.
* ```remove_digits```: Boolean indicating whether to remove digits. Default is ```True```.
* ```remove_news```: Boolean indicating whether to remove the names of well-known news organizations. Default is ```True```.
* ```remove_i_s```: Boolean indicating whether to remove free-standing capital letter I's. Default is ```True```.
* ```remove_geo```: Boolean indicating whether to remove geographical information (as determined by [geostring](https://github.com/dfreelon/geostring)). Default is ```True```.

#### Output 
A list of lists in which each sub-list contains the name of an entity and the number of times it appeared in the corpus. Entities are listed in descending order by count.


### kwic

__Sample code__

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

__Sample code__

```python
t1 = 'nationalist'
t2 = 'nationalism'
nat_ps = pyl.pairwise_stem(t1,t2)
print(nat_ps)
```

### pairwise_stem_all

__Sample code__

```python
nat_list = ['nation','national','nationalism','nationalist','nationality']
nat_psa = pyl.pairwise_stem_all(nat_list)
print(nat_psa)
```

### get_context_terms

__Sample code__

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
