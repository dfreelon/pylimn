# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 10:11:42 2019

@author: Deen Freelon

PyLimn: a small module for NLP-based text description
"""

import collections
import geostring as geo
import itertools as itr
import nltk
import re

### REGEXES FOR NAMED ENTITIES ###

re_find_all_proper_names = re.compile('(?<!\-)([A-Z]+.*?)(?:\s+[a-z\W]|[`\'’"^,;\:\—\\\*\.\(\)\[\]]|$)')
re_find_hyphenated = re.compile('(?:\s)([a-z]+\-[A-Z]+.*?)(?:\s+[a-z\W]|[`\'’"^,;\:\—\\\*\.\(\)\[\]]|$)')
re_split_remove_endpunct = re.compile('(?:[!\.\?\:])(?:\s)')
re_remove_digits = re.compile('^\d+$')
re_remove_dates = re.compile('^([a-z]+ )(\d{2}|\d{4})$')
re_org_words1 = re.compile("(?<=\s)(al|and|at|bin|by|de|for|in|of|on|the|to)(?=\s|\-)")
re_org_words2 = re.compile("^((al|and|at|bin|by|de|for|in|of|on|the|to)\s)+|(\s(al|and|at|bin|by|de|for|in|of|on|the|to))+$",re.VERBOSE | re.M) #add "bin" and "al-"
re_remove_mi_periods = re.compile("\s[A-Z]\.\s")
re_remove_inner_i = re.compile("^i | i$")

### STOPWORDS FOR NAMED ENTITIES ###

stop_nltk = [i.replace("'","") 
             for i 
             in nltk.corpus.stopwords.words('english')]
days_wk = ['monday',
           'tuesday',
           'wednesday',
           'thursday',
           'friday',
           'saturday',
           'sunday']
months = ['january',
          'february',
          'march',
          'april',
          'may',
          'june',
          'july',
          'august',
          'september',
          'october',
          'november',
          'december']
stop_other = ["thank","well"]
news_orgs = set(['afp',
                 'anchor',
                 'associated press',
                 'cbs',
                 'correspondent',
                 'cnn',
                 'dateline',
                 'facebook',
                 'fox news',
                 'highlight',
                 'nbc',
                 'newsweek',
                 'new york times',
                 'national public radio',
                 'npr',
                 'reuters',
                 'twitter',
                 'united press international',
                 'usa today',
                 'washington post'])
stop_ne = set(stop_nltk + 
              months + 
              days_wk +
              stop_other) 

### STOPWORDS FOR DESCRIPTIVE TERMS ###

stop_ct = set(stop_nltk + ['accord','act','actually','another','also','base','back','become','begin','call','city','com','come','day','dont','else','expect','even','event','every','get','face','first','give','hear','hold','home','however','http','include','indeed','instead','know','last','least','leave','listen','local','made','make','man','maybe','mean','meanwhile','meet','month','move','much','name','need','next','often','part','people','place','plan','point','really','recent','return','said','say','schedule','seek','show','site','stand','state','step','still','take','talk','tell','thanks','thing','think','tie','time','today','tonight','try','use','view','want','way','week','word','work','www','year'])

### FUNCTIONS ###

#lowercase first words of sentences
def snt_to_lower(snt):
    if re.search('"',snt[0]):
        return '"' + snt[1].lower() + snt[2:]
    else:
        return snt[0].lower() + snt[1:]

def rank_named_entities(news_series,
                        min_entity_ct=5,
                        min_entity_len=4,
                        stop_words=stop_ne,
                        include_first_words=False,
                        remove_upper_terms=False,
                        find_hyphenated=True,
                        once_per_doc=True,
                        remove_dates=True,
                        remove_digits=True,
                        remove_news=True,
                        remove_i_s=True,
                        remove_geo=True):
    news_split = news_series.tolist() #must be a Pandas series
    news_split = [re.sub('[^\w\s\.\?\!:,\(\-\)]',
                  ' ',
                  i.replace("'s","")) 
                  for i 
                  in news_split 
                  if i != ''] #remove most punctuation, newlines, dupes 

    news_split = [re_org_words1.sub(
                  lambda x: x.group().strip().capitalize(),i)
                  for i 
                  in news_split] #capitalizes org words like "of" "for" etc.
    news_split = [re_remove_mi_periods.sub(
                  lambda x: x.group().replace(".",""),i)
                  for i
                  in news_split] #remove periods from after people's middle initials
    
    pnames = []
    first_words = []
    for n,i in enumerate(news_split): 
        news_split[n] = ' '.join([j for j 
                                  in news_split[n].split() 
                                  if len(j) >= min_entity_len
                                  or j[0].isupper()]) #remove words that are shorter than min_entity_len
        
        punct_split = re_split_remove_endpunct.split(news_split[n])
        
        if include_first_words == True:
            f_words = []
            nxt_words = []
            for j in punct_split:
                pn = re_find_all_proper_names.findall(j)
                if len(pn) > 0:
                    f_words.append(pn[0].lower())
                if len(pn) > 1:
                    nxt_words.extend([k.lower() for k in pn[1:]])
            if once_per_doc == True:
                f_words = set(f_words)
                nxt_words = set(nxt_words)
            if len(nxt_words) > 0:                
                f_words = [k for k 
                           in f_words 
                           if k not in nxt_words]
            first_words.extend(f_words)
        
        news_split[n] = ' '.join([snt_to_lower(j.strip()) 
                if re.search("[a-z]",j.split()[0])
                else j
                for j 
                in punct_split
                if len(j) >= min_entity_len 
                or ' ' in j]) #lowercase first letters of sentences
        if remove_upper_terms == True:
            news_split[n] = ' '.join([j for j 
                                      in news_split[n].split() 
                                      if not j.isupper()
                                      or len(j) == 1])
        p_list = [re_org_words2.sub('',
                  j.lower().strip()) 
                  for j 
                  in re_find_all_proper_names.findall(news_split[n])]
        
        if find_hyphenated == True:
            p_list2 = [re_org_words2.sub('',
                       j.lower().strip()) 
                       for j 
                       in re_find_hyphenated.findall(news_split[n])]
        else:
            p_list2 = []
        
        if once_per_doc == True:
            p_all = set(p_list + p_list2) #count each name only once per article
        else:
            p_all = tuple(p_list + p_list2)
            
        if len(p_all) > 0: 
            for j in p_all:
                conditions = []
                conditions.append((len(j) >= min_entity_len 
                                   or ' ' in j))
                conditions.append("copyright" not in j)
                #conditions.append(j not in stop_words)
                if remove_dates == True:
                    conditions.append(re_remove_dates.search(j) == None)
                if remove_digits == True:
                    conditions.append(re_remove_digits.search(j) == None)
                if remove_news == True:
                    conditions.append(not any(o 
                                      in j
                                      for o 
                                      in news_orgs))
                if remove_i_s == True:
                    conditions.append(re_remove_inner_i.search(j) == None)
                if all(conditions):
                    pnames.append(j)
    
    #return locals()
    ptop = collections.Counter(pnames).most_common()
    ptop = [i for i in ptop if i[1] >= min_entity_ct]
    ptop_filtered = []

    if remove_geo == True:
        for i in ptop:
            if geo.resolve(i[0],True) == None:
                ptop_filtered.append(list(i))
    else:
        ptop_filtered = [list(i) for i in ptop]
        
    ptop_uniq = set([i[0] for i in ptop_filtered])
    fw_top = collections.Counter(first_words).most_common()
    for i in fw_top:
        if i[0] in ptop_uniq:
            tmp_index = [n 
                         for n,j 
                         in enumerate(ptop_filtered) 
                         if i[0] == j[0]][0]
            ptop_filtered[tmp_index][1] += i[1]
    
    return sorted([i for i in ptop_filtered if i[0] not in stop_words],
                  key=lambda x: x[1],
                  reverse=True)

def kwic(string,
         regex_term,
         n_terms=25,
         overlap=False,
         ignore_case=True,
         underscore_join=False):
    kwic_list = []
    pos2 = []
    if ignore_case == True:
        positions = [[i.start(),i.end()] 
                    for i 
                    in re.finditer(regex_term,
                                   string,
                                   re.IGNORECASE)]
    else:
        positions = [[i.start(),i.end()] 
                    for i 
                    in re.finditer(regex_term,
                                   string)]        
    if overlap == False:
        start = 0
        for n,i in enumerate(positions):
            string_bn = string[start:positions[n][0]]
            words_bn = string_bn.split()
            if len(words_bn) >= n_terms * 2 or n == 0:
                pos2.append(i)
                start = positions[n][1]
    else:
        pos2 = positions
            
    for i in pos2:
        before = string[:i[0]]
        after = string[i[1]:]
        before_terms = before.split()[-n_terms:]
        after_terms = after.split()[:n_terms]
        if underscore_join == True:
            regex_term = regex_term.replace(' ','_')
        kwic_instance = ' '.join(before_terms) + \
                        ' >>>' + \
                        regex_term + \
                        '<<< ' + \
                        ' '.join(after_terms)           
        kwic_list.append(kwic_instance)
    return [kwic_list,positions]

def pairwise_stem(s1,
                  s2,
                  regex_remove="[^a-z_ ]"):
    s1 = re.sub(regex_remove," ",s1.lower())
    s2 = re.sub(regex_remove," ",s2.lower())
    if len(s1) > len(s2): #make sure shorter word goes first
        sa = s2
        sb = s1
    else:
        sa = s1
        sb = s2
    common = ""
    
    for n,i in enumerate(sa):
       if sa[n] == sb[n]:
           common += i
       else:
           break
       
    return {'p_stem':common,
            'ps_coef':(2*len(common))/(len(sa) + len(sb)),
            'originals':(s1,s2)}

def pairwise_stem_all(tokens,
                      coef_threshold=0.8,
                      regex_remove="[^a-z_ ]",
                      longest_stem=True):
    uniq_tokens = set(tokens)
    if longest_stem == True:
        pairs = tuple(
                sorted(
                    itr.combinations(
                            uniq_tokens,2)))
    else:
        pairs = tuple(
                sorted(
                    itr.combinations(
                            uniq_tokens,2),
                            reverse=True))
    os_dict = {}
    for i in pairs:
        ps = pairwise_stem(i[0],i[1],regex_remove)
        if ps['ps_coef'] >= coef_threshold:
            if ((i[0] in os_dict 
                 and len(ps['p_stem']) > len(os_dict[i[0]])) 
                    or i[0] not in os_dict):
                os_dict[i[0]] = ps['p_stem']
            if ((i[1] in os_dict 
                 and len(ps['p_stem']) > len(os_dict[i[1]])) 
                    or i[1] not in os_dict):
                os_dict[i[1]] = ps['p_stem']
    tokens = [os_dict[i]
              if i 
              in os_dict
              else i
              for i 
              in tokens]
    return tokens

def get_context_terms(kwic_list,
                      min_term_ct=5,
                      min_term_len=3,
                      stop_words=stop_ct,
                      pos=set(['NN','NNS','VB','VBD','VBG', 'VBN','VBP','VBZ','JJ','JJR','JJS']),
                      remove_proper=False,
                      remove_digits=True,
                      remove_hyphens=True,
                      lemmatize=True,
                      stem_pairwise=True,
                      merge_ngrams=True):
    strip_punc = nltk.tokenize.RegexpTokenizer(r'\w+')
    wnl = nltk.stem.WordNetLemmatizer()
    all_tokens = []
    
    for string in kwic_list:
        #remove all capitalized words except where preceded by a dash or by a lowercase letter (e.g. neoNazi)
        if remove_proper == True:
            string = re.sub('(?<![\-a-z])([A-Z]+.*?)(?=\W)',
                            '',
                            string) 
        string = re.sub('>>>.+?<<<','',string)
        if remove_digits == True:
            string = re.sub('[0-9]','',string)
        if remove_hyphens == True:
            string = string.replace('-','').lower()
        string = strip_punc.tokenize(string)
        string = [i
                  for i 
                  in string 
                  if i not in stop_words
                  and len(i) > 2]
        kwic_bigrams = nltk.bigrams(string)
        if lemmatize == True:
            for n,i in enumerate(string):
                vlem = wnl.lemmatize(i,'v')
                if vlem != i:
                    string[n] = vlem
                else: 
                    string[n] = wnl.lemmatize(i)
        str_tokens = [i for i 
                      in string 
                      if len(i) >= min_term_len]

        tagged_tokens = nltk.pos_tag(str_tokens)
        descr_tokens = [i[0] for i 
                        in tagged_tokens 
                        if i[1] in pos]
        all_tokens.extend(descr_tokens)
        str_bigrams = [i[0] + '_' + i[1] #add underscore to separate bigrams
                       for i 
                       in kwic_bigrams]
        all_tokens += str_bigrams

    all_tokens = [i for i in all_tokens 
                  if len(i) >= min_term_len]
    # filter out infrequent words
    top_at = collections.Counter(all_tokens).most_common()
    top_at = set([i[0] 
                  for i 
                  in top_at 
                  if i[1] >= min_term_ct])
    all_tokens = [i
                  for i 
                  in all_tokens 
                  if i in top_at]
    
    if stem_pairwise == True:
        all_tokens = pairwise_stem_all(all_tokens)
    
    unigrams = [i for i in all_tokens if '_' not in i]
    bigrams = [i for i in all_tokens if '_' in i]
    top_unigrams = collections.Counter(unigrams).most_common()
    top_bigrams = collections.Counter(bigrams).most_common()
    
    final_uni = [i for i 
                 in top_unigrams 
                 if i[0] not in stop_words
                 and i[1] >= min_term_ct
                 and len(i[0]) >= min_term_len]
    final_bi = [list(i)
                for i 
                in top_bigrams 
                if i[0] not in stop_words
                and i[1] >= min_term_ct
                and len(i[0]) >= min_term_len]
    #merge bigrams with unigram equivalents
    if merge_ngrams == True:
        bi_nounder = [i[0].replace("_","") for i in final_bi]
        space_bi = [i for i in final_uni if i[0] in bi_nounder]
        if len(space_bi) > 0:
            for i in space_bi:
                if i[0] in bi_nounder:
                    x = [n 
                         for n,j 
                         in enumerate(bi_nounder) 
                         if i[0] == j][0]
                    final_bi[x][1] += i[1]
                    final_uni = [j 
                                 for j 
                                 in final_uni 
                                 if j[0] != i[0]]
            
    return {"unigrams":final_uni,
            "bigrams":final_bi}