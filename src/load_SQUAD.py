import json
from pprint import pprint
import codecs
import re
import itertools
import numpy
import operator
import string
from sklearn.metrics import f1_score
from nltk.tokenize import TreebankWordTokenizer
import nltk
from nltk.tag import pos_tag
from nltk.tag.stanford import StanfordNERTagger
from nltk.data import load
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.chunk import conlltags2tree, tree2conlltags
from collections import defaultdict
from evaluate import evaluate_groundtruthlist_predlist



from common_functions import cosine_simi

path='/mounts/data/proj/wenpeng/Dataset/SQuAD/'
_digits = re.compile('\d')

# postagdict = load('help/tagsets/upenn_tagset.pickle')
# postag_list =  postagdict.keys()
# postag_dict = {k: v for v, k in enumerate(postag_list)}
# postag_dict['O']=len(postag_dict)
postag_dict = {'PRP$': 0, 'VBG': 1, 'FW': 18, 'VBN': 4, 'POS': 19, "''": 6, 'VBP': 7, 'WDT': 8, 'JJ': 9, 'WP': 10, 'VBZ': 11, 'DT': 12, 'RP': 13, '$': 14, 'NN': 15, ')': 16, '(': 17, 'VBD': 2, ',': 5, '.': 20, 'TO': 21, 'LS': 22, 'RB': 23, ':': 24, 'NNS': 25, 'NNP': 26, '``': 3, 'WRB': 28, 'CC': 29, 'PDT': 30, 'RBS': 31, 'RBR': 32, 'CD': 33, 'PRP': 34, 'EX': 35, 'IN': 36, 'WP$': 37, 'MD': 38, 'NNPS': 39, '--': 40, 'JJS': 41, 'JJR': 42, 'SYM': 43, 'VB': 27, 'UH': 44, '#':45}
postag_size = len(postag_dict)
nertag_dict = {'GPE':0, 'PERSON':1,'ORGANIZATION':2, 'O':3, 'FACILITY':4,'LOCATION':5, 'GSP':6}
nertag_size = len(nertag_dict)
wh_word_dict = {#'What':0, 'When':1, 'Where':2, 'Which':3, 'Who':4, 'Whose':5, 'Why':6, 'How':7, 'How long':8, 'How many':9, 'How much':10,
                'what':0, 'when':1, 'where':2, 'which':3, 'who':4, 'whose':5, 'why':6, 'how':7, 'how long':8, 'how many':9, 'how much':10}
wh_word_size = len(wh_word_dict)
months = set(['january','february','march','april','May','june','july','august','september','october','november','december'])
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
#
#     def lower(text):
#         return text.lower()

    return white_space_fix(remove_articles(remove_punc(s)))


def form_pos2id():
    pos_list=['CC','CD','DT','EX','FW','IN','JJ','JJR','JJS','LS','MD','NN','NNS','NNP','NNPS','PDT','POS','PRP','PRP$','RB','RBR',
        'RBS','RP','SYM','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WP$','WRB','.']
    return dict(zip(pos_list, range(len(pos_list))))

def form_ner2id():
    ner_list=['LOCATION', 'PERSON', 'ORGANIZATION', 'MONEY', 'PERCENT', 'DATE', 'TIME', 'O']
    return dict(zip(ner_list, range(len(ner_list))))

def tokenize(str):
    listt=TreebankWordTokenizer().tokenize(str)
    refined_listt=[]
    for word in listt:
        if word !='.' and word[-1]=='.':
            refined_listt.append(word[:-1])
            refined_listt.append('.')
        else:
            refined_listt.append(word)
    return refined_listt

path='/mounts/data/proj/wenpeng/Dataset/SQuAD/'



def pos_and_ner(wordlist, ner_tagger, pos2id, ner2id, pos_size, ner_size):
    word_pos_list=pos_tag(wordlist)
    word_ner_list=ner_tagger.tag(wordlist)
    pos_list=[ tag for word, tag, in word_pos_list]
    pos_ids=strs2ids_with_max(pos_list, pos2id, pos_size)
    ner_list=[ ner for word, ner, in word_ner_list]
    ner_ids=strs2ids_with_max(ner_list, ner2id, ner_size)
    return pos_ids, ner_ids

def poslist_nerlist_2_featurematrix(poslist, nerlist, pos_size, ner_size):
    pos_featurematrix=[]
    ner_featurematrix=[]
    for pos in poslist:
        features=[0.0]*pos+[1.0]+[0.0]*(pos_size-pos-1)
        pos_featurematrix.append(features)
    for ner in nerlist:
        features=[0.0]*ner+[1.0]+[0.0]*(ner_size-ner-1)
        ner_featurematrix.append(features)
    return pos_featurematrix, ner_featurematrix


def transform_raw_paragraph(raw_word_list):
    #concatenate upper case words
    new_para=[]
    tmp_word=''
    for word in raw_word_list:

        if word[0].isupper():
            tmp_word+='='+word
        else:
            if len(tmp_word)>0:
                new_para.append(tmp_word[1:]) #remove the first '='
                tmp_word=''
            new_para.append(word)
    if len(tmp_word)>0:
        new_para.append(tmp_word[1:])
    return new_para

def strs2ids_onehot(str_list, word2id, size):
    ids=[]
    for word in str_list:
        id=word2id.get(word)
        if id is None:
            id=size-1
        features=[0.0]*id+[1.0]+[0.0]*(size-id-1)
        ids.append(id)
    return ids

def strs2ids_with_max(str_list, word2id, size):
    return [word2id.get(word, size-1) for word in str_list]

def strs2ids(str_list, word2id):
    ids=[]
    for word in str_list:
        id=word2id.get(word)
        if id is None:
            id=len(word2id)+1   # start from 1
        word2id[word]=id
        ids.append(id)
    return ids

def strs2ids_vocab(str_list, word2id, vocab):
    ids=[]
    for word in str_list:
        if word not in vocab:
            word='UNK'
        word=word.lower()
        id=word2id.get(word)
        if id is None:
            id=len(word2id)+1   # start from 2, 0 for pad, 1 for UNK

        word2id[word]=id
        ids.append(id)
    return ids
def load_stopwords():
    readfile=open(path+'stopwords.txt', 'r')
    stopwords=set()
    for line in readfile:
        stopwords.add(line.strip())
    readfile.close()
    return stopwords
def extra_features(stop_words, paragraph_wordlist, Q_wordlist):
    Q_wordset=set(Q_wordlist)

    remove_pos=[]
    for i in range(len(paragraph_wordlist)):
        word=paragraph_wordlist[i]
        if word in Q_wordset and word.lower() not in stop_words:
            remove_pos.append(i)

    _digits = re.compile('\d')
    features=[]
    for i in range(len(paragraph_wordlist)):
        word=paragraph_wordlist[i]
        word_f_v=[]# uppercase, digits, distance
        if  word[:1].isupper():
            word_f_v.append(1.0)#uppcase
        else:
            word_f_v.append(0.0)
        if bool(_digits.search(word)):
            word_f_v.append(1.0)
        else:
            word_f_v.append(0.0)

        if len(remove_pos)==0:
            word_f_v.append(0.0)
        else:
            shortest_distance=numpy.min(numpy.abs(numpy.asarray(remove_pos)-i))
            if shortest_distance==0:
                word_f_v.append(0.0)
            else:
                word_f_v.append(1.0/shortest_distance)
        features.append(word_f_v)
#     print features
    return features

def truncate_by_punct( wordlist, from_right_to_left):
    if from_right_to_left:
        i=len(wordlist)-1
        while i > -1:
            if wordlist[i]=='.' and i < len(wordlist)-1:
                break
            i-=1
        if i==0:
            return wordlist[i:]
        else:
            return wordlist[i+1:]
    else:
        i=0
        while i < len(wordlist):
            if wordlist[i]=='.' and i >0:
                break
            i+=1
        if i == len(wordlist)-1:
            return wordlist
        else:
            return wordlist[:i]

def  load_train(para_len_limit, q_len_limit):
    max_para_len=para_len_limit
    max_Q_len = q_len_limit

    word2id={}
#     read_file=open(path+'train-v1.0.json', 'r')
    with open(path+'train-v1.1.json') as data_file:
        data = json.load(data_file)

#     pprint(data['data'][0]['paragraphs'][0])
    doc_size=len(data['data'])
#     print 'doc_size:', doc_size
    para_size=0
    qa_size=0
    para_list=[]
    Q_list=[]
#     Q_size_list=[]
    label_list=[]
    para_mask=[]
    mask=[]
    feature_matrixlist=[]
    stop_words=load_stopwords()
    for i in range(doc_size):#each doc
        para_size_i=len(data['data'][i]['paragraphs'])
        for j in range(para_size_i):#each paragraph
            question_size_j=len(data['data'][i]['paragraphs'][j]['qas'])
#             Q_size_list.append(question_size_j)
            paragraph=data['data'][i]['paragraphs'][j]['context']
#             print 'paragraph:', paragraph
#             paragraph_wordlist=paragraph.strip().split()
#             paragraph_idlist=strs2ids(paragraph_wordlist, word2id)
#             para_len=len(paragraph_wordlist)

#             Q_sublist=[]
#             label_sublist=[]
#             feature_tensor=[]

#             max_q_len=0
            for q in range(question_size_j):
                question_q=data['data'][i]['paragraphs'][j]['qas'][q]['question']
                question_wordlist=tokenize(question_q.strip())


#                 feature_tensor.append(feature_matrix_q)

                question_idlist=strs2ids(question_wordlist, word2id)
                q_len=len(question_idlist)
#                 if len(question_idlist)>max_q_len:
#                     max_q_len=len(question_idlist)
                answer_q=data['data'][i]['paragraphs'][j]['qas'][q]['answers'][0]['text']
                answer_q_wordlist=tokenize(answer_q)
                answer_len=len(answer_q_wordlist)
                answer_start_q=data['data'][i]['paragraphs'][j]['qas'][q]['answers'][0]['answer_start']
#                 while answer_start_q>0 and paragraph[answer_start_q-1]!=' ':
#                     answer_start_q-=1
                answer_left=paragraph[:answer_start_q]
#                 answer_left_wordlist=truncate_by_punct(tokenize(answer_left), True)
                answer_left_wordlist=tokenize(answer_left)
                answer_left_size=len(answer_left_wordlist)
                answer_right=paragraph[answer_start_q+len(answer_q):]
#                 answer_right_wordlist=truncate_by_punct(tokenize(answer_right), False)
                answer_right_wordlist=tokenize(answer_right)
                answer_right_size=len(answer_right_wordlist)
                gold_label_q=[0]*answer_left_size+[1]*answer_len+[0]*answer_right_size

                para_len=answer_left_size+answer_len+answer_right_size
                paragraph_wordlist=answer_left_wordlist+answer_q_wordlist+answer_right_wordlist
#                 print 'paragraph_wordlist:', paragraph_wordlist
#                 print 'question_wordlist:', question_wordlist
#                 exit(0)
                feature_matrix_q=extra_features(stop_words, paragraph_wordlist, question_wordlist)
                paragraph_idlist=strs2ids(paragraph_wordlist, word2id)
                #now, pad paragraph, question, feature_matrix, gold_label
                #first paragraph
                pad_para_len=max_para_len-para_len
                if pad_para_len>0:
                    paded_paragraph_idlist=[0]*pad_para_len+paragraph_idlist
                    paded_para_mask_i=[0.0]*pad_para_len+[1.0]*para_len
                    paded_feature_matrix_q=[[0]*3]*pad_para_len+feature_matrix_q
                    paded_gold_label=[0]*pad_para_len+gold_label_q
                else:
                    paded_paragraph_idlist=paragraph_idlist[:max_para_len]
                    paded_para_mask_i=([1.0]*para_len)[:max_para_len]
                    paded_feature_matrix_q=feature_matrix_q[:max_para_len]
                    paded_gold_label=gold_label_q[:max_para_len]
#                 if 1.0 not in set(paded_gold_label):
#                     print 'numpy.sum(numpy.asarray(paded_gold_label))<1'
#                     exit(0)
                para_list.append(paded_paragraph_idlist)
                para_mask.append(paded_para_mask_i)
                feature_matrixlist.append(paded_feature_matrix_q)
                label_list.append(binaryLabelList2Value(paded_gold_label))
                #then question
                pad_q_len=max_Q_len-q_len
                if pad_q_len > 0:
                    paded_question_idlist=[0]*pad_q_len+question_idlist
                    paded_q_mask_i=[0.0]*pad_q_len+[1.0]*q_len
                else:
                    paded_question_idlist=question_idlist[:max_Q_len]
                    paded_q_mask_i=([1.0]*q_len)[:max_Q_len]
                Q_list.append(paded_question_idlist)
                mask.append(paded_q_mask_i)


#             submask=[]
#             Q_sublist_padded=[]
#             for orig_q in Q_sublist: # pad zero at end of sentences
#                 existing_len=len(orig_q)
#                 pad_len=max_q_len-existing_len
#                 if pad_len>0:
#                     orig_q+=[0]*pad_len
#                 Q_sublist_padded.append(orig_q)
#                 submask.append([1.0]*existing_len+[0.0]*pad_len)

#             for orig_q in Q_sublist: # pad zero at mid of sentences
#                 existing_len=len(orig_q)
#                 pad_len=max_q_len-existing_len
#                 if pad_len>0:
#                     mid_place=existing_len/2
#                     orig_q=orig_q[:mid_place]+[0]*pad_len+orig_q[mid_place:]
#                 Q_sublist_padded.append(orig_q)
#                 submask.append([1.0]*mid_place+[0.0]*pad_len+[1.0]*(existing_len-mid_place))


#             Q_list.append(Q_sublist_padded)
#             label_list.append(label_sublist)
#             mask.append(submask)
#             feature_tensorlist.append(feature_tensor)
#             print 'question_size_j:', question_size_j
            qa_size+=question_size_j
#         print 'para_size_i:', para_size_i
        para_size+=para_size_i
#     pprint(len(data['data']))
#     print data['data'][0]['paragraphs'][0]
    print 'Load train set', para_size, 'paragraphs,', qa_size, 'question-answer pairs'
    print 'Train Vocab size:', len(word2id)
#     exit(0)
    return para_list, Q_list, label_list, para_mask, mask, word2id, feature_matrixlist

def  load_dev_or_test_google(word2vec, word2id, para_len_limit, q_len_limit):
#     Dev  max_para_len:, 629 max_q_len: 33
#     read_file=open(path+'train-v1.0.json', 'r')
    max_para_len=para_len_limit
    max_Q_len = q_len_limit
#     ner_tagger = StanfordNERTagger(path+'stanford-ner-2015-12-09/classifiers/english.muc.7class.distsim.crf.ser.gz', path+'stanford-ner-2015-12-09/stanford-ner.jar')
    pos2id=form_pos2id()
    pos_size=len(pos2id)+1
    ner2id=form_ner2id()
    ner_size=len(ner2id)+1
    read_file=codecs.open(path+'dev-reformed.txt', 'r', 'utf-8')

    qa_size=0
    para_list=[]
    Q_list=[]
    Q_list_word=[]
    para_mask=[]
    mask=[]
    feature_matrixlist=[]
    pos_matrixlist=[]
    ner_matrixlist=[]
    para_text_list=[]
    q_ansSet_list=[]
    stop_words=load_stopwords()

    past_tag=''
    for line in read_file:
        parts=line.strip().split('\t')
        if parts[0]=='W:':#is paragraph
            paragraph_wordlist=parts[1].split()
#             paragraph_idlist=strs2ids(paragraph_wordlist, word2id)
#             para_len=len(paragraph_idlist)
            past_tag=''
        if parts[0]=='P:':#is POS
            pos_list=map(int,parts[1].split())
            past_tag=''
        if parts[0]=='N:':#is NER
            ner_list=map(int,parts[1].split())
            past_tag=''
        if parts[0]=='A:':#is labels
#             gold_label_q=map(int,parts[1].split())
            q_ansSet=set()
            for i in range(1, len(parts)):
                q_ansSet.add(parts[i])
            past_tag=''
        if parts[0]=='Q:':#is question
            question_wordlist=parts[1].split()
            question_idlist=strs2ids(question_wordlist, word2id)
            q_len=len(question_idlist)
            past_tag='Q'

        if past_tag =='Q': #store


            truncate_paragraph_wordlist, sentB_list=truncate_paragraph_by_question_returnBounary(word2vec, paragraph_wordlist, question_wordlist, 1)
#                 truncate_paragraph_wordlist = paragraph_wordlist
            truncate_paragraph_idlist=strs2ids(truncate_paragraph_wordlist, word2id)
            truncate_para_len=len(truncate_paragraph_wordlist)
            feature_matrix_q=extra_features(stop_words, truncate_paragraph_wordlist, question_wordlist)
            truncate_pos_list=[]
            truncate_ner_list=[]
            for pair in sentB_list:
                truncate_pos_list+=pos_list[pair[0]:pair[1]]
                truncate_ner_list+=ner_list[pair[0]:pair[1]]

            if len(truncate_pos_list)!=truncate_para_len or len(truncate_ner_list)!=truncate_para_len:
                print 'len(truncate_pos_list)!=truncate_para_len or len(truncate_ner_list)!=truncate_para_len:', len(truncate_pos_list), len(truncate_ner_list), truncate_para_len
                exit(0)
            pos_feature_matrix, ner_feature_matrix= poslist_nerlist_2_featurematrix(truncate_pos_list, truncate_ner_list, pos_size, ner_size)
#             for i in range(len(pos_list)):
#                 if len(pos_feature_matrix[i])!=pos_size:
#                     print 'len(pos_feature_matrix)!=pos_size:', len(pos_feature_matrix[i])
#                     exit(0)
#             for i in range(len(ner_list)):
#                 if len(ner_feature_matrix[i])!=ner_size:
#                     print 'len(ner_feature_matrix)!=ner_size:', len(ner_feature_matrix[i])
#                     exit(0)

            #first paragraph
            pad_para_len=max_para_len-truncate_para_len
            if pad_para_len>0:
                paded_paragraph_idlist=[0]*pad_para_len+truncate_paragraph_idlist
                paded_para_mask_i=[0.0]*pad_para_len+[1.0]*truncate_para_len
                paded_feature_matrix_q=[[0]*3]*pad_para_len+feature_matrix_q
                paded_pos_feature_matrix=[[0.0]*pos_size]*pad_para_len+pos_feature_matrix
                paded_ner_feature_matrix=[[0.0]*ner_size]*pad_para_len+ner_feature_matrix
                paded_para_text=['UNK']*pad_para_len+truncate_paragraph_wordlist
            else:
                paded_paragraph_idlist=truncate_paragraph_idlist[:max_para_len]
                paded_para_mask_i=([1.0]*truncate_para_len)[:max_para_len]
                paded_feature_matrix_q=feature_matrix_q[:max_para_len]
                paded_pos_feature_matrix=pos_feature_matrix[:max_para_len]
                paded_ner_feature_matrix=ner_feature_matrix[:max_para_len]
                paded_para_text=truncate_paragraph_wordlist[:max_para_len]

            para_list.append(paded_paragraph_idlist)
            para_mask.append(paded_para_mask_i)
            feature_matrixlist.append(paded_feature_matrix_q)
            pos_matrixlist.append(paded_pos_feature_matrix)
            ner_matrixlist.append(paded_ner_feature_matrix)
            para_text_list.append(paded_para_text)
            #then question
            pad_q_len=max_Q_len-q_len
            if pad_q_len > 0:
                paded_question_idlist=[0]*pad_q_len+question_idlist
                paded_q_mask_i=[0.0]*pad_q_len+[1.0]*q_len
            else:
                paded_question_idlist=question_idlist[:max_Q_len]
                paded_q_mask_i=([1.0]*q_len)[:max_Q_len]
#                 paded_question_idlist=[0]*pad_q_len+question_idlist
#                 paded_q_mask_i=[0.0]*pad_q_len+[1.0]*q_len
            Q_list.append(paded_question_idlist)
            Q_list_word.append(question_wordlist)
            mask.append(paded_q_mask_i)
            #then , store answers
            q_ansSet_list.append(q_ansSet)

            qa_size+=1

    print 'Load dev set', qa_size, 'question-answer pairs'
    print 'Train+Dev Vocab size:', len(word2id)
#     print word2id
    return para_list, Q_list, Q_list_word, para_mask, mask, len(word2id), word2id, para_text_list, q_ansSet_list, feature_matrixlist, pos_matrixlist, ner_matrixlist
def  load_dev_or_test(word2id, para_len_limit, q_len_limit):
#     Dev  max_para_len:, 629 max_q_len: 33
#     read_file=open(path+'train-v1.0.json', 'r')
    max_para_len=para_len_limit
    max_Q_len = q_len_limit
    with open(path+'dev-v1.1.json') as data_file:
        data = json.load(data_file)

#     pprint(data['data'][0]['paragraphs'][0])
    doc_size=len(data['data'])
#     print 'doc_size:', doc_size

    word2vec=load_word2vec()


    para_size=0
    qa_size=0
    para_list=[]
    Q_list=[]
    Q_list_word=[]
    para_mask=[]
    mask=[]
    feature_matrixlist=[]
    para_text_list=[]
    q_ansSet_list=[]
    stop_words=load_stopwords()
    id_list=[]
    for i in range(doc_size):#each doc
        para_size_i=len(data['data'][i]['paragraphs'])
        for j in range(para_size_i):#each paragraph
            question_size_j=len(data['data'][i]['paragraphs'][j]['qas']) #how many questions for this paragraph
#             Q_size_list.append(question_size_j)
            paragraph=data['data'][i]['paragraphs'][j]['context']
#             print 'paragraph:', paragraph
            paragraph_wordlist=tokenize(paragraph.strip())
#             para_text_list.append(paragraph_wordlist)
#             paragraph_idlist=strs2ids(paragraph_wordlist, word2id)
#             para_len=len(paragraph_wordlist)

#             Q_sublist=[]
#             label_sublist=[]
#             feature_tensor=[]
#             ansSetList=[]
#             max_q_len=0
            for q in range(question_size_j): # for each question
                question_q=data['data'][i]['paragraphs'][j]['qas'][q]['question']
                q_id = data['data'][i]['paragraphs'][j]['qas'][q]['id']
#                 print 'q_id:', q_id
                question_wordlist=tokenize(question_q.strip())
                truncate_paragraph_wordlist=truncate_paragraph_by_question(word2vec, paragraph_wordlist, question_wordlist, 1)
                truncate_paragraph_idlist=strs2ids(truncate_paragraph_wordlist, word2id)
                truncate_para_len=len(truncate_paragraph_wordlist)
                feature_matrix_q=extra_features(stop_words, truncate_paragraph_wordlist, question_wordlist)
#                 feature_tensor.append(feature_matrix_q)


                question_idlist=strs2ids(question_wordlist, word2id)
                q_len=len(question_idlist)
#                 if len(question_idlist)>max_q_len:
#                     max_q_len=len(question_idlist)

                answer_no=len(data['data'][i]['paragraphs'][j]['qas'][q]['answers'])
                q_ansSet=set()
                for ans in range(answer_no):
                    answer_q=data['data'][i]['paragraphs'][j]['qas'][q]['answers'][ans]['text']
                    q_ansSet.add(' '.join(tokenize(answer_q.strip())))
#                     answer_len=len(answer_q.strip().split())

#                     answer_start_q=data['data'][i]['paragraphs'][j]['qas'][q]['answers'][ans]['answer_start']
#                     while answer_start_q>0 and paragraph[answer_start_q-1]!=' ':
#                         answer_start_q-=1
#                     answer_left=paragraph[:answer_start_q]
#                     answer_left_size=len(answer_left.strip().split())
#                     gold_label_q=[-1.0]*answer_left_size+[1.0]*answer_len+[-1.0]*(para_len-answer_left_size-answer_len)
#                 ansSetList.append(q_ansSet)
#                 Q_sublist.append(question_idlist)
#                 if len(label_sublist)>=1 and len(gold_label_q)!=len(label_sublist[-1]):
#                     print 'wired size'
#                     print len(gold_label_q),len(label_sublist[-1])
#                     exit(0)
#                 label_sublist.append(gold_label_q)
                #now, pad paragraph, question, feature_matrix, gold_label
                #first paragraph
                pad_para_len=max_para_len-truncate_para_len
                if pad_para_len>0:
                    paded_paragraph_idlist=[0]*pad_para_len+truncate_paragraph_idlist
                    paded_para_mask_i=[0.0]*pad_para_len+[1.0]*truncate_para_len
                    paded_feature_matrix_q=[[0]*3]*pad_para_len+feature_matrix_q
                    paded_para_text=['UNK']*pad_para_len+truncate_paragraph_wordlist
                else:
                    paded_paragraph_idlist=truncate_paragraph_idlist[:max_para_len]
                    paded_para_mask_i=([1.0]*truncate_para_len)[:max_para_len]
                    paded_feature_matrix_q=feature_matrix_q[:max_para_len]
                    paded_para_text=truncate_paragraph_wordlist[:max_para_len]

#                 paded_paragraph_idlist=[0]*pad_para_len+paragraph_idlist
#                 paded_para_mask_i=[0.0]*pad_para_len+[1.0]*para_len
#                 paded_feature_matrix_q=[[0]*3]*pad_para_len+feature_matrix_q
#                 paded_para_text=['UNK']*pad_para_len+paragraph_wordlist
                para_list.append(paded_paragraph_idlist)
                para_mask.append(paded_para_mask_i)
                feature_matrixlist.append(paded_feature_matrix_q)
                para_text_list.append(paded_para_text)
                #then question
                pad_q_len=max_Q_len-q_len
                if pad_q_len > 0:
                    paded_question_idlist=[0]*pad_q_len+question_idlist
                    paded_q_mask_i=[0.0]*pad_q_len+[1.0]*q_len
                else:
                    paded_question_idlist=question_idlist[:max_Q_len]
                    paded_q_mask_i=([1.0]*q_len)[:max_Q_len]
#                 paded_question_idlist=[0]*pad_q_len+question_idlist
#                 paded_q_mask_i=[0.0]*pad_q_len+[1.0]*q_len
                Q_list.append(paded_question_idlist)
                Q_list_word.append(question_wordlist)
                mask.append(paded_q_mask_i)
                #then , store answers
                q_ansSet_list.append(q_ansSet)
                id_list.append(q_id)

            qa_size+=question_size_j
#         print 'para_size_i:', para_size_i
        para_size+=para_size_i
#     pprint(len(data['data']))
#     print data['data'][0]['paragraphs'][0]
    print 'Load dev set', para_size, 'paragraphs,', qa_size, 'question-answer pairs'
    print 'Train+Dev Vocab size:', len(word2id)
#     print word2id
    return para_list, Q_list, Q_list_word, para_mask, mask, len(word2id), word2id, para_text_list, q_ansSet_list, feature_matrixlist, id_list
def fine_grained_subStr(text):
    #supposed text is a word list
    length=len(text)

    substr_set=set()
    substr_set.add(' '.join(text))
    if length>1:
        for i in range(1,length):
            for j in range(length-i+1):
#                 print ' '.join(text[j:j+i])
                substr_set.add(' '.join(text[j:j+i]))

#     print substr_set
    return substr_set

def extract_ansList_attentionList_maxlen5(word_list, att_list, extra_matrix, mask_list, q_wordlist): #extra_matrix in shape (|V|, 3)
    q_wordset=set(q_wordlist)
    max_len=3
    if len(word_list)!=len(att_list):
        print 'len(word_list)!=len(att_list):', len(word_list), len(att_list)
        exit(0)
    para_len=len(word_list)
    start_point=para_len-int(numpy.sum(numpy.asarray(mask_list)))
    average_att=numpy.mean(numpy.asarray(att_list[start_point:]))

#     pred_ans_list=[]
    token_list=[]
    score_list=[]
    ans2att={}
    att_list=list(att_list)
    att_list.append(-100.0) #to make sure to store the last valid answer
    for pos in range(start_point, para_len+1):
        if att_list[pos]>average_att and word_list[pos] not in q_wordset:# and word_list[pos] not in string.punctuation:
            token_list.append(word_list[pos])
            score_list.append(att_list[pos]+0.5*numpy.sum(extra_matrix[pos]))
#             new_answer=new_answer.strip()
#             if pos == para_len-1 and len(new_answer)>0:
#                 pred_ans_list.append(new_answer)
#                 ans2att[new_answer]=accu_att/numpy.sqrt(len(new_answer.split()))
        else:
            if len(token_list)>0:
                if len(token_list)>max_len:
                    for i in range(len(token_list)-max_len):
                        new_answer=' '.join(token_list[i:i+max_len])
                        new_score=numpy.sum(numpy.asarray(score_list[i:i+max_len]))/numpy.sqrt(max_len)
                        ans2att[new_answer]=new_score
                else:
                    new_answer=' '.join(token_list)
                    new_score=numpy.sum(numpy.asarray(score_list))/numpy.sqrt(len(token_list))
                    ans2att[new_answer]=new_score
                del token_list[:]
                del score_list[:]
            else:
                continue

#     print 'pred_ans_list:', pred_ans_list
#     fine_grained_ans_set=set()
#     for pred_ans in pred_ans_list:
#         fine_grained_ans_set|=fine_grained_subStr(pred_ans.split())
#     return fine_grained_ans_set
    if len(ans2att)>0:
        best_answer=max(ans2att, key=ans2att.get)
        #best_answer=' '.join(ans2att.keys())
    else:
        best_answer=None
#     print best_answer
#     exit(0)
#     return set(pred_ans_list)
    return best_answer

def extract_ansList_attentionList(word_list, att_list, extra_matrix, mask_list, q_wordlist): #extra_matrix in shape (|V|, 3)

    q_wordset=set(q_wordlist)
    if len(word_list)!=len(att_list):
        print 'len(word_list)!=len(att_list):', len(word_list), len(att_list)
        exit(0)
    para_len=len(word_list)
    start_point=para_len-int(numpy.sum(numpy.asarray(mask_list)))
    average_att=numpy.mean(numpy.asarray(att_list[start_point:]))

    pred_ans_list=[]
    new_answer=''
    accu_att=0.0
    ans2att={}
    for pos in range(start_point, para_len):
        if att_list[pos]>average_att and word_list[pos] not in q_wordset:# and word_list[pos] not in string.punctuation:
            new_answer+=' '+word_list[pos]
            accu_att+=att_list[pos]+0.5*numpy.sum(extra_matrix[pos])
            new_answer=new_answer.strip()
            if pos == para_len-1 and len(new_answer)>0:
                pred_ans_list.append(new_answer)
                ans2att[new_answer]=accu_att/numpy.sqrt(len(new_answer.split()))
        else:
            if len(new_answer)>0:
#                 if len(new_answer.split())<=4:
                pred_ans_list.append(new_answer)
                ans2att[new_answer]=accu_att/numpy.sqrt(len(new_answer.split()))
                new_answer=''
                accu_att=0.0
            else:
                continue

#     print 'pred_ans_list:', pred_ans_list
#     fine_grained_ans_set=set()
#     for pred_ans in pred_ans_list:
#         fine_grained_ans_set|=fine_grained_subStr(pred_ans.split())
#     return fine_grained_ans_set
    if len(ans2att)>0:
        best_answer=max(ans2att, key=ans2att.get)
        #best_answer=' '.join(ans2att.keys())
    else:
        best_answer=None
#     print best_answer
#     exit(0)
#     return set(pred_ans_list)
    return best_answer

def  restore_train():
    word2id={}
#     read_file=open(path+'train-v1.0.json', 'r')
    with codecs.open(path+'train-v1.0.json', 'r', 'utf-8') as data_file:
        data = json.load(data_file)

    writefile=codecs.open(path+'train_extractedRaw.txt', 'w', 'utf-8')

    doc_size=len(data['data'])
#     print 'doc_size:', doc_size
    para_size=0
    qa_size=0
    para_list=[]
    Q_list=[]
    Q_size_list=[]
    label_list=[]
    mask=[]
    para_co=0
    for i in range(doc_size):#each doc
        para_size_i=len(data['data'][i]['paragraphs'])
        for j in range(para_size_i):#each paragraph
            question_size_j=len(data['data'][i]['paragraphs'][j]['qas'])
            Q_size_list.append(question_size_j)
            paragraph=data['data'][i]['paragraphs'][j]['context']
            writefile.write('\n......\n\n'+paragraph+'\n')
            para_co+=1
            continue
#             print 'paragraph:', paragraph
            paragraph_wordlist=paragraph.strip().split()
            paragraph_idlist=strs2ids(paragraph_wordlist, word2id)
            para_len=len(paragraph_wordlist)

            Q_sublist=[]
            label_sublist=[]

            max_q_len=0
            for q in range(question_size_j):
                question_q=data['data'][i]['paragraphs'][j]['qas'][q]['question']
                question_idlist=strs2ids(question_q.strip().split(), word2id)
                if len(question_idlist)>max_q_len:
                    max_q_len=len(question_idlist)
                answer_q=data['data'][i]['paragraphs'][j]['qas'][q]['answers'][0]['text']
#                 print 'answer_q:', answer_q
                answer_len=len(answer_q.strip().split())
                answer_start_q=data['data'][i]['paragraphs'][j]['qas'][q]['answers'][0]['answer_start']
                while answer_start_q>0 and paragraph[answer_start_q-1]!=' ':
                    answer_start_q-=1
                answer_left=paragraph[:answer_start_q]
                answer_left_size=len(answer_left.strip().split())
                gold_label_q=[-1.0]*answer_left_size+[1.0]*answer_len+[-1.0]*(para_len-answer_left_size-answer_len)

                Q_sublist.append(question_idlist)
                if len(label_sublist)>=1 and len(gold_label_q)!=len(label_sublist[-1]):
                    print 'wired size'
                    print len(gold_label_q),len(label_sublist[-1])
                    exit(0)
                label_sublist.append(gold_label_q)

            submask=[]
            Q_sublist_padded=[]
#             for orig_q in Q_sublist: # pad zero at end of sentences
#                 existing_len=len(orig_q)
#                 pad_len=max_q_len-existing_len
#                 if pad_len>0:
#                     orig_q+=[0]*pad_len
#                 Q_sublist_padded.append(orig_q)
#                 submask.append([1.0]*existing_len+[0.0]*pad_len)

            for orig_q in Q_sublist: # pad zero at mid of sentences
                existing_len=len(orig_q)
                pad_len=max_q_len-existing_len
                if pad_len>0:
                    mid_place=existing_len/2
                    orig_q=orig_q[:mid_place]+[0]*pad_len+orig_q[mid_place:]
                Q_sublist_padded.append(orig_q)
                submask.append([1.0]*mid_place+[0.0]*pad_len+[1.0]*(existing_len-mid_place))

            para_list.append(paragraph_idlist)
            Q_list.append(Q_sublist_padded)
            label_list.append(label_sublist)
            mask.append(submask)

#             print 'question_size_j:', question_size_j
            qa_size+=question_size_j
#         print 'para_size_i:', para_size_i
        para_size+=para_size_i
#     pprint(len(data['data']))
#     print data['data'][0]['paragraphs'][0]
    writefile.close()
    print 'Load train set', para_size

def parse_NERed_train():#not useful
    readfile=codecs.open(path+'train_extractedRaw_NER.txt', 'r', 'utf-8')
    writefile=codecs.open(path+'train_tokenized.txt', 'w', 'utf-8')

    for line in readfile:

        if line.strip().find('.../O')>=0:
            writefile.write('\n')
        else:
            parts=line.strip().split()
            new_sent=''
            for part in parts:
                word=part[:part.rfind('/')]

                new_sent+=' '+word

            writefile.write(' '.join(new_sent.strip().split())+'\n')
    readfile.close()
    writefile.close()

def macrof1(str1, str2):
    vocab1=set(str1.split())
    vocab2=set(str2.split())
    vocab=vocab1|vocab2

    str1_labellist=[]
    str2_labellist=[]
    for word in vocab:
        if word in vocab1:
            str1_labellist.append(1)
        else:
            str1_labellist.append(0)
        if word in vocab2:
            str2_labellist.append(1)
        else:
            str2_labellist.append(0)

#     TP_pos=0.0
#     FP_pos=0.0
#     FN_pos=0.0
#     for word in vocab:
#         if word in vocab1 and word in vocab2:
#             TP_pos+=1
#         elif word in vocab1 and word not in vocab2:
#             FP_pos+=1
#         elif word not in vocab1 and word  in vocab2:
#             FN_pos+=1
#     recall=TP_pos/(TP_pos+FN_pos) if TP_pos+FN_pos > 0 else 0.0
#     precision=TP_pos/(TP_pos+FP_pos) if TP_pos+FP_pos > 0 else 0.0
#
#     f1=2*recall*precision/(recall+precision) if recall+precision> 0 else 0.0

    return f1_score(str1_labellist, str2_labellist, average='binary')

def MacroF1(strQ, strset):

    if strQ is None:
        return 0.0
    else:
        max_f1=0.0
        for strr in strset:
            new_f1=macrof1(strQ, strr)
            if new_f1 > max_f1:
                max_f1=new_f1
    #     print max_f1
        return max_f1

def MacroF1_crf(strQ, strset):

    if strQ=='':
        return 0.0
    else:
        max_f1=0.0
        for strr in strset:
            new_f1=macrof1(strQ, strr)
            if new_f1 > max_f1:
                max_f1=new_f1
    #     print max_f1
        return max_f1

def load_word2vec():
    word2vec = {}

    print "==> loading 300d word2vec"
#     with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/glove/glove.6B." + str(dim) + "d.txt")) as f:
    f=open('/mounts/data/proj/wenpeng/Dataset/glove.840B.300d.txt', 'r')#word2vec_words_300d.txt, glove.6B.50d.txt
    for line in f:
        l = line.split()
        word2vec[l[0]] = map(float, l[1:])

    print "==> word2vec is loaded"

    return word2vec

def load_glove():
    word2vec = {}

    print "==> loading 300d word2vec"
#     with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/glove/glove.6B." + str(dim) + "d.txt")) as f:
#     f=open('/mounts/data/proj/wenpeng/Dataset/word2vec_words_300d.txt', 'r')  #glove.6B.100d.txt, word2vec_words_300d.txt, glove.6B.300d.txt
    f=open('sub_embs.txt', 'r')
    for line in f:
        l = line.split()
        word2vec[l[0]] = map(float, l[1:])

    print "==> word2vec is loaded"

    return word2vec
def overlap_degree(sent_wordlist, q_wordlist):
    sent=set(sent_wordlist)
    q=set(q_wordlist)
    overlap=sent&q
    return len(overlap)*1.0/len(q)
def truncate_paragraph_by_question(word2vec, para_wordlist, q_wordlist, topN):
    #first convert para into sents
    zero_emb=list(numpy.zeros(300))
    sents_end_indices=[]
    sents_end_indices.append(0)
    para_wordembs=[]

    for i, word in enumerate(para_wordlist):
        if word =='.' and i > 0:
            sents_end_indices.append(i)
        para_wordembs.append(word2vec.get(word, zero_emb))
    if sents_end_indices[-1] !=len(para_wordlist)-1:
        sents_end_indices.append(len(para_wordlist)-1)
#     print sents_end_indices
    q_wordembs=[]
    for word in q_wordlist:
        q_wordembs.append(word2vec.get(word, zero_emb))
    q_emb=numpy.sum(numpy.asarray(q_wordembs), axis=0)

    sentid2cos={}
    for i in range(len(sents_end_indices)-1):
        sent_emb=numpy.sum(numpy.asarray(para_wordembs[sents_end_indices[i]:sents_end_indices[i+1]]), axis=0)
        cosine=cosine_simi(q_emb, sent_emb)
        sent_wordlist=para_wordlist[sents_end_indices[i]:sents_end_indices[i+1]]
        overlap_simi=overlap_degree(sent_wordlist, q_wordlist)
        sentid2cos[i]=cosine+overlap_simi
    sorted_x = sorted(sentid2cos.items(), key=operator.itemgetter(1), reverse=True)
    new_para_wordlist=[]
    for sentid, cos in sorted_x[:topN]:
        new_para_wordlist+=para_wordlist[sents_end_indices[sentid]:sents_end_indices[sentid+1]]
    return new_para_wordlist

def truncate_paragraph_by_question_returnBounary(word2vec, para_wordlist, q_wordlist, topN):
    #first convert para into sents
    zero_emb=list(numpy.zeros(300))
    sents_end_indices=[]
    sents_end_indices.append(0)
    para_wordembs=[]

    for i, word in enumerate(para_wordlist):
        if word =='.' and i >0: #sentence length at least 1
            sents_end_indices.append(i)
        para_wordembs.append(word2vec.get(word, zero_emb))
    if sents_end_indices[-1] !=len(para_wordlist)-1:
        sents_end_indices.append(len(para_wordlist)-1)
#     print sents_end_indices
    q_wordembs=[]
    for word in q_wordlist:
        q_wordembs.append(word2vec.get(word, zero_emb))
    q_emb=numpy.sum(numpy.asarray(q_wordembs), axis=0)

    sentid2cos={}
    sentid2pair={}
    for i in range(len(sents_end_indices)-1):
        sent_emb=numpy.sum(numpy.asarray(para_wordembs[sents_end_indices[i]:sents_end_indices[i+1]]), axis=0)
        cosine=cosine_simi(q_emb, sent_emb)
        sent_wordlist=para_wordlist[sents_end_indices[i]:sents_end_indices[i+1]]
        overlap_simi=overlap_degree(sent_wordlist, q_wordlist)
        sentid2cos[i]=cosine+overlap_simi
        sentid2pair[i]=(sents_end_indices[i], sents_end_indices[i+1])
    sorted_x = sorted(sentid2cos.items(), key=operator.itemgetter(1), reverse=True)
    new_para_wordlist=[]
    new_para_sentB=[]
    for sentid, cos in sorted_x[:topN]:
        new_para_wordlist+=para_wordlist[sents_end_indices[sentid]:sents_end_indices[sentid+1]]
        new_para_sentB.append(sentid2pair.get(sentid))
    return new_para_wordlist, new_para_sentB

def load_word2vec_to_init(rand_values, ivocab, word2vec):

    unk=0
    for id, word in ivocab.iteritems():
        emb=word2vec.get(word)
        if emb is None:
            emb=word2vec.get(word.lower())
        if emb is not None:
            rand_values[id]=emb
        else:
            unk+=1
#             uncovered_vocab.write(word+'\n')
#     uncovered_vocab.close()

    print '==> use word2vec initialization over...', unk, 'words failed to init'
#     exit(0)
    return rand_values

def strlist_2_wordidlist(strlist, word2id, id2word):
    idlist=[]
    for word in strlist:
        word_id=word2id.get(word)
        if word_id is None:
            word_id=len(word2id) # do not need plus 1 as word2vec[UNK]=0
            word2id[word]=word_id
        idlist.append(word_id)
        id2word[word_id] = word
    return idlist

def elelist_2_idlist(strlist, word2id):
    idlist=[]
    for word in strlist:
        word_id=word2id.get(word)
        if word_id is None:
            word_id=len(word2id) # do not need plus 1 as word2vec[UNK]=0
            word2id[word]=word_id
        idlist.append(word_id)
    return idlist

def str_2_charidlist(whole_str, char2id):
    idlist=[]
    for char in whole_str:
        char_id=char2id.get(char)
        if char_id is None:
            char_id=len(char2id)
            char2id[char]=char_id
        idlist.append(char_id)
    return idlist

def strlist_2_wordidlist_noIncrease(strlist, word2id):
    idlist=[]
    for word in strlist:
        word_id=word2id.get(word)
        if word_id is None:
            word_id=1
        idlist.append(word_id)
    return idlist
def pad_idlist(idlist, maxlen):
    valid_size=len(idlist)
    pad_size=maxlen-valid_size
    if pad_size > 0:
        idlist=[0]*pad_size+idlist
        mask=[0.0]*pad_size+[1.0]*valid_size
    else:
        idlist=idlist[:maxlen]
        mask=[1.0]*maxlen
    return idlist, mask
def leftpad_idlist_padsize(idlist, maxlen):
    valid_size=len(idlist)
    pad_size=maxlen-valid_size
    if pad_size >= 0:
        idlist=[0]*pad_size+idlist
        mask=[0.0]*pad_size+[1.0]*valid_size
    else:
        idlist=idlist[-pad_size:]
        mask=[1.0]*maxlen
        pad_size=0
    return idlist, mask,pad_size
def rightpad_idlist_padsize(idlist, maxlen):
    valid_size=len(idlist)
    pad_size=maxlen-valid_size
    if pad_size >= 0:
        idlist=idlist+[0]*pad_size
        mask=[1.0]*valid_size+[0.0]*pad_size
    else:
        idlist=idlist[:maxlen]
        mask=[1.0]*maxlen
    return idlist, mask,pad_size
def load_SQUAD_hinrich(example_no_limit, max_context_len, max_span_len, max_q_len):
    line_co=0
    example_co=0
    readfile=open('/mounts/work/hs/yin/20161030/squadnewtrn.txt', 'r')
    word2id={}
    word2id['UNK']=1 # use it to pad zero context
    questions=[]
    questions_mask=[]
    lefts=[]
    lefts_mask=[]
    spans=[]
    spans_mask=[]
    rights=[]
    rights_mask=[]
    for line in readfile:
        if line_co % 11==0 and line_co > 0:
            example_co+=1
#             if example_co%1000000==0:
#                 print example_co
            if example_co == example_no_limit:
                break
        if line_co%11==3 or line_co%11==4 or line_co%11==8:
            line_co+=1
            continue
        else:
            if line_co%11==1:#question
                q_example=strlist_2_wordidlist(line.strip().split(), word2id)
                pad_q_example, q_mask=pad_idlist(q_example, max_q_len)
                questions.append(pad_q_example)
                questions.append(pad_q_example) # repeat if for pos and neg
                questions_mask.append(q_mask)
                questions_mask.append(q_mask)
            elif line_co%11==2 or line_co%11==7: # span
                if line.strip()[0] not in set(['T','W']):
                    print 'line.strip()[0]!=T or W', line, line_co
                    exit(0)
                span_example=strlist_2_wordidlist(line.strip().split()[1:], word2id)
                pad_span_example, span_mask=pad_idlist(span_example, max_span_len)
                spans.append(pad_span_example)
                spans_mask.append(span_mask)
            elif line_co%11==5 or line_co%11==9:#left
                left_example=strlist_2_wordidlist(line.strip().split(), word2id)
                pad_left_example, left_mask=pad_idlist(left_example, max_context_len)
                lefts.append(pad_left_example)
                lefts_mask.append(left_mask)
            elif line_co%11==6 or line_co%11==10:#right
                right_example=strlist_2_wordidlist(line.strip().split(), word2id)
                pad_right_example, right_mask=pad_idlist(right_example, max_context_len)
                rights.append(pad_right_example)
                rights_mask.append(right_mask)
        line_co+=1
        # print line_co
    if example_co != example_no_limit:
        example_co+=1
    readfile.close()
    print 'load', example_co, 'train pairs finished'
    if len(questions)!=2*example_no_limit:
        print 'len(questions)!=2*example_co:', len(questions), example_no_limit
        exit(0)
    return     word2id,questions,questions_mask,lefts,lefts_mask,spans,spans_mask,rights,rights_mask

def load_dev_hinrich(word2id, example_no_limit, max_context_len, max_span_len, max_q_len):
    line_co=0
    example_co=0
    readfile=open('/mounts/work/hs/yin/20161030/squadnewdev.txt', 'r')
#     word2id={}
#     word2id['UNK']=1 # use it to pad zero context
    all_ground_truth=[] # is a list of string
    all_questions=[]
    all_questions_mask=[]
    all_lefts=[]
    all_lefts_mask=[]
    all_spans=[]
    all_candidates_f1=[]
    all_spans_mask=[]
    all_rights=[]
    all_rights_mask=[]

    questions=[]
    questions_mask=[]
    lefts=[]
    lefts_mask=[]
    spans=[]  #id list
    candidates_f1=[]  # string for the candidate
    spans_mask=[]
    rights=[]
    rights_mask=[]
    old_question='UNK'
    new_example_flag=False
    for line in readfile:
        if line_co%11==0 or line_co%11==3 or line_co%11==4 or line_co%11==5  or line_co%11==6:
            line_co+=1
            continue
        else:
            if line_co%11==1:#question
                q_str=line.strip()
                if q_str !=old_question:   #new question
                    old_question=q_str
                    new_example_flag=True
                    if len(questions)>0:
                        if len(questions)!=len(lefts) or len(questions)!=len(spans) or len(questions)!=len(rights) or len(questions)!=len(candidates_f1):
                            print 'len(questions)!=len(lefts) or len(questions)!=len(spans) or len(questions)!=len(rights) or len(questions)!=len(candidates)'
                            print len(questions), len(lefts), len(spans), len(rights), len(candidates_f1)
                            exit(0)
                        all_questions.append(questions)
                        all_questions_mask.append(questions_mask)
                        all_lefts.append(lefts)
                        all_lefts_mask.append(lefts_mask)
                        all_spans.append(spans)
                        all_candidates_f1.append(candidates_f1)
                        all_spans_mask.append(spans_mask)
                        all_rights.append(rights)
                        all_rights_mask.append(rights_mask)

                        example_co+=1
                        # print example_co, 'example_co'
                        if example_co == example_no_limit:
                            break
                        else:
                            #for a new question-paragraph
                            # del questions
                            # del questions_mask
                            # del lefts
                            # del lefts_mask
                            # del spans
                            # del candidates
                            # del spans_mask
                            # del rights
                            # del rights_mask
                            questions=[]
                            questions_mask=[]
                            lefts=[]
                            lefts_mask=[]
                            spans=[]
                            candidates_f1=[]
                            spans_mask=[]
                            rights=[]
                            rights_mask=[]
                else: # q equal to old question
                    new_example_flag=False
                q_example=strlist_2_wordidlist_noIncrease(q_str, word2id)
                pad_q_example, q_mask=pad_idlist(q_example, max_q_len)
                questions.append(pad_q_example)
                questions_mask.append(q_mask)
            elif line_co%11==2: #ground truth
                if new_example_flag is True:
                    if line.strip()[0]=='T':
                        all_ground_truth.append(line.strip()[2:]) # add a string
                    else:
                        print 'line.strip()[0]!=T'
                        exit(0)
                else:
                    line_co+=1
                    continue
            elif  line_co%11==7: # span
                line_str=line.strip()
                if  line_str[0]!='W':
                    print 'line.strip()[0]!=W', line, line_co
                    exit(0)
                span_example=strlist_2_wordidlist_noIncrease(line_str.split()[1:], word2id)
                pad_span_example, span_mask=pad_idlist(span_example, max_span_len)
                spans.append(pad_span_example)
#                 candidates.append(line_str[2:]) #the candidate string
                spans_mask.append(span_mask)
            elif line_co%11==8: # f1
                candidates_f1.append(float(line.strip()))

            elif line_co%11==9:#left
                left_str=line.strip()
                if len(left_str)==0:
                    left_wordlist=['UNK']*max_context_len
                else:
                    left_wordlist=left_str.split()
                left_example=strlist_2_wordidlist_noIncrease(left_wordlist, word2id)
                pad_left_example, left_mask=pad_idlist(left_example, max_context_len)
                lefts.append(pad_left_example)
                lefts_mask.append(left_mask)
            elif line_co%11==10:#right
                right_str=line.strip()
                if len(right_str)==0:
                    right_wordlist=['UNK']*max_context_len
                else:
                    right_wordlist=right_str.split()
                right_example=strlist_2_wordidlist_noIncrease(right_wordlist, word2id)
                pad_right_example, right_mask=pad_idlist(right_example, max_context_len)
                rights.append(pad_right_example)
                rights_mask.append(right_mask)
        line_co+=1
        # print line_co, 'line_co'
    readfile.close()
    print 'load', example_co, 'question-paragraph pairs finished'
    if len(all_ground_truth)!=example_no_limit or len(all_questions)!=example_no_limit or len(all_lefts)!=example_no_limit or len(all_spans)!=example_no_limit or len(all_rights)!=example_no_limit:
        print 'len(all_ground_truth)!=example_co or len(all_questions)!=example_co:', len(all_ground_truth), example_no_limit , len(all_questions)
        exit(0)

    return     all_ground_truth,all_candidates_f1, all_questions,all_questions_mask,all_lefts,all_lefts_mask,all_spans,all_spans_mask,all_rights,all_rights_mask

def  load_train_google(para_len_limit, q_len_limit):
    max_para_len=para_len_limit
    max_Q_len = q_len_limit
#     ner_tagger = StanfordNERTagger(path+'stanford-ner-2015-12-09/classifiers/english.muc.7class.distsim.crf.ser.gz', path+'stanford-ner-2015-12-09/stanford-ner.jar')
    pos2id=form_pos2id()
    pos_size=len(pos2id)+1
    ner2id=form_ner2id()
    ner_size=len(ner2id)+1
    word2id={}
    read_file=codecs.open(path+'train-reformed.txt', 'r', 'utf-8')


    qa_size=0
    para_list=[]
    Q_list=[]
#     Q_size_list=[]
    label_list=[]
    para_mask=[]
    mask=[]
    feature_matrixlist=[]
    pos_matrixlist=[]
    ner_matrixlist=[]
    stop_words=load_stopwords()
    size_control=70000
    past_tag=''
    for line in read_file:
        parts=line.strip().split('\t')
        if parts[0]=='W:':#is paragraph
            paragraph_wordlist=parts[1].split()
            paragraph_idlist=strs2ids(paragraph_wordlist, word2id)
            para_len=len(paragraph_idlist)
            past_tag=''
        if parts[0]=='P:':#is POS
            pos_list=map(int,parts[1].split())
            past_tag=''
        if parts[0]=='N:':#is NER
            ner_list=map(int,parts[1].split())
            past_tag=''
        if parts[0]=='L:':#is labels
            gold_label_q=map(int,parts[1].split())
            past_tag=''
        if parts[0]=='Q:':#is question
            question_wordlist=parts[1].split()
            question_idlist=strs2ids(question_wordlist, word2id)
            q_len=len(question_idlist)
            past_tag='Q'

        if past_tag =='Q': #store

            if para_len != len(pos_list) or para_len != len(ner_list) or para_len != len(gold_label_q):
                continue
            feature_matrix_q=extra_features(stop_words, paragraph_wordlist, question_wordlist)  #(para_len, 3)
            pos_feature_matrix, ner_feature_matrix= poslist_nerlist_2_featurematrix(pos_list, ner_list, pos_size, ner_size)

            #now, pad paragraph, question, feature_matrix, gold_label
            #first paragraph
            pad_para_len=max_para_len-para_len
            if pad_para_len>0:
                paded_paragraph_idlist=[0]*pad_para_len+paragraph_idlist
                paded_para_mask_i=[0.0]*pad_para_len+[1.0]*para_len

                paded_feature_matrix_q=[[0]*3]*pad_para_len+feature_matrix_q
                paded_pos_feature_matrix=[[0.0]*pos_size]*pad_para_len+pos_feature_matrix
                paded_ner_feature_matrix=[[0.0]*ner_size]*pad_para_len+ner_feature_matrix
                paded_gold_label=[0]*pad_para_len+gold_label_q
            else:
                paded_paragraph_idlist=paragraph_idlist[:max_para_len]
                paded_para_mask_i=([1.0]*para_len)[:max_para_len]
                paded_feature_matrix_q=feature_matrix_q[:max_para_len]
                paded_pos_feature_matrix=pos_feature_matrix[:max_para_len]
                paded_ner_feature_matrix=ner_feature_matrix[:max_para_len]
                paded_gold_label=gold_label_q[:max_para_len]
#                 if 1.0 not in set(paded_gold_label):
#                     print 'numpy.sum(numpy.asarray(paded_gold_label))<1'
#                     exit(0)
            para_list.append(paded_paragraph_idlist)
            para_mask.append(paded_para_mask_i)
            feature_matrixlist.append(paded_feature_matrix_q)
            pos_matrixlist.append(paded_pos_feature_matrix)
            ner_matrixlist.append(paded_ner_feature_matrix)
            label_list.append(binaryLabelList2Value(paded_gold_label))
            #then question
            pad_q_len=max_Q_len-q_len
            if pad_q_len > 0:
                paded_question_idlist=[0]*pad_q_len+question_idlist
                paded_q_mask_i=[0.0]*pad_q_len+[1.0]*q_len
            else:
                paded_question_idlist=question_idlist[:max_Q_len]
                paded_q_mask_i=([1.0]*q_len)[:max_Q_len]
            Q_list.append(paded_question_idlist)
            mask.append(paded_q_mask_i)

            qa_size+=1
#             if qa_size == size_control:
#                 break



    print 'Load train set', qa_size, 'question-answer pairs'
    print 'Train Vocab size:', len(word2id)
#     exit(0)
    return para_list, Q_list, label_list, para_mask, mask, word2id, feature_matrixlist, pos_matrixlist, numpy.asarray(ner_matrixlist)

def decode_crf_labels(label_list, word_list, mask_list):
    if len(label_list)!=len(mask_list) or len(word_list)!=len(mask_list):
        print 'len(label_list)!=len(mask_list) or len(word_list)!=len(mask_list):', len(label_list),len(word_list),len(mask_list)
        exit(0)
    valid_start=0
    while mask_list[valid_start]<1.0:
        valid_start+=1
    ans=''
    for i in range(valid_start, len(label_list)):
        if label_list[i] ==1 or label_list[i] ==2:
            ans+=word_list[i]
    return ans


def decode_predict_id(value, wordlist):
    length=len(wordlist)
    if value < length:
        span_len=1
        span_start=value
    elif value >= length and value < 2*length-1:
        span_len=2
        span_start=value-length
    elif value >= 2*length-1 and value < 3*length-3:
        span_len=3
        span_start=value-(2*length-1)
    elif value >= 3*length-3 and value < 4*length-6:
        span_len=4
        span_start=value-(3*length-3)
    elif value >= 4*length-6 and value < 5*length-10:
        span_len=5
        span_start=value-(4*length-6)
    elif value >= 5*length-10 and value < 6*length-15:
        span_len=6
        span_start=value-(5*length-10)
    elif value >= 6*length-15 and value < 7*length-21:
        span_len=7
        span_start=value-(6*length-15)
    return ' '.join(wordlist[span_start:span_start+span_len])

def decode_predict_id_AI2(value, para_len, wordlist):

    start=value/para_len
    end=value%para_len

    return ' '.join(wordlist[start:end+1])


def binaryLabelList2Value(values):
    one_start=-1
    one_co=0
    length=len(values)
    for index, value in enumerate(values):
        if value ==1:
            one_co+=1
            if one_start<0:
                one_start=index

    if one_co>7:
        one_co=7
    pos=(one_co-1)*length-(one_co-1)*(one_co-2)/2 + one_start


    if one_co ==0:
        return 0
    else:
        return pos

def binaryLabelList2StartEnd(values):
    one_start=-1
    one_co=0
#     length=len(values)
    for index, value in enumerate(values):
        if value ==1:
            one_co+=1
            if one_start<0:
                one_start=index

#     if one_co>7:
#         one_co=7
#     pos=(one_co-1)*length-(one_co-1)*(one_co-2)/2 + one_start

    if one_start ==-1:
        one_start=0
    if one_co==0:
        one_co=1
    return one_start, one_start+one_co-1

def  store_SQUAD_train():
    ner_tagger = StanfordNERTagger(path+'stanford-ner-2015-12-09/classifiers/english.muc.7class.distsim.crf.ser.gz', path+'stanford-ner-2015-12-09/stanford-ner.jar')
    pos2id=form_pos2id()
    pos_size=len(pos2id)+1
    ner2id=form_ner2id()
    ner_size=len(ner2id)+1

#     read_file=open(path+'train-v1.0.json', 'r')
    with codecs.open(path+'train-v1.1.json', 'r', 'utf-8') as data_file:
        data = json.load(data_file)
    writefile=codecs.open(path+'train-reformed.txt', 'w', 'utf-8')

#     pprint(data['data'][0]['paragraphs'][0])
    doc_size=len(data['data'])
#     print 'doc_size:', doc_size
    para_size=0
    qa_size=0


    for i in range(doc_size):#each doc
        para_size_i=len(data['data'][i]['paragraphs'])
        for j in range(para_size_i):#each paragraph
            question_size_j=len(data['data'][i]['paragraphs'][j]['qas'])
#             Q_size_list.append(question_size_j)
            paragraph=data['data'][i]['paragraphs'][j]['context']
            paragraph_wordlist=tokenize(paragraph.strip())
            pos_list, ner_list= pos_and_ner(paragraph_wordlist, ner_tagger, pos2id, ner2id, pos_size, ner_size)

            for q in range(question_size_j):
                question_q=data['data'][i]['paragraphs'][j]['qas'][q]['question']
                question_wordlist=tokenize(question_q.strip())


                answer_q=data['data'][i]['paragraphs'][j]['qas'][q]['answers'][0]['text']
                answer_q_wordlist=tokenize(answer_q)
                answer_len=len(answer_q_wordlist)
#                 answer_char_len=len(answer_q)
                answer_start_q=data['data'][i]['paragraphs'][j]['qas'][q]['answers'][0]['answer_start']
#                 while answer_start_q>0 and paragraph[answer_start_q-1]!=' ':
#                     answer_start_q-=1
                answer_left=paragraph[:answer_start_q]
#                 answer_left_wordlist=truncate_by_punct(tokenize(answer_left), True)
                answer_left_wordlist=tokenize(answer_left)
                answer_left_size=len(answer_left_wordlist)
#                 answer_right=paragraph[answer_start_q+answer_char_len:]
# #                 answer_right_wordlist=truncate_by_punct(tokenize(answer_right), False)
#                 answer_right_wordlist=tokenize(answer_right)
#                 answer_right_size=len(answer_right_wordlist)
                gold_label_q=[0]*answer_left_size+[1]*answer_len+[0]*(len(paragraph_wordlist)-answer_left_size-answer_len)

#                 if len(gold_label_q)!=len(paragraph_wordlist):
# #                     print 'len(gold_label_q)!=len(paragraph_wordlist):', len(gold_label_q), len(paragraph_wordlist)
# #                     print 'paragraph:', paragraph
#                     noise+=1
#                     continue
#                     exit(0)

#                 paragraph_wordlist=answer_left_wordlist+answer_q_wordlist+answer_right_wordlist

#                 pos_list, ner_list= pos_and_ner(paragraph_wordlist, ner_tagger, pos2id, ner2id, pos_size, ner_size)


                #write into file
                writefile.write('W:\t'+' '.join(paragraph_wordlist)+'\n')
                writefile.write('P:\t'+' '.join(map(str,pos_list))+'\n')
                writefile.write('N:\t'+' '.join(map(str,ner_list))+'\n')
                writefile.write('L:\t'+' '.join(map(str, gold_label_q))+'\n')
                writefile.write('Q:\t'+' '.join(question_wordlist)+'\n')


            qa_size+=question_size_j
            print 'pair size:', qa_size#, 'noise:', noise
        para_size+=para_size_i

    writefile.close()
    print 'Store train set', para_size, 'paragraphs,', qa_size, 'question-answer pairs'


def  store_SQUAD_dev():
    ner_tagger = StanfordNERTagger(path+'stanford-ner-2015-12-09/classifiers/english.muc.7class.distsim.crf.ser.gz', path+'stanford-ner-2015-12-09/stanford-ner.jar')
    pos2id=form_pos2id()
    pos_size=len(pos2id)+1
    ner2id=form_ner2id()
    ner_size=len(ner2id)+1

#     read_file=open(path+'train-v1.0.json', 'r')
    with codecs.open(path+'dev-v1.1.json', 'r', 'utf-8') as data_file:
        data = json.load(data_file)
    writefile=codecs.open(path+'dev-reformed.txt', 'w', 'utf-8')

#     pprint(data['data'][0]['paragraphs'][0])
    doc_size=len(data['data'])
#     print 'doc_size:', doc_size
    para_size=0
    qa_size=0


    for i in range(doc_size):#each doc
        para_size_i=len(data['data'][i]['paragraphs'])
        for j in range(para_size_i):#each paragraph
            question_size_j=len(data['data'][i]['paragraphs'][j]['qas'])
#             Q_size_list.append(question_size_j)
            paragraph=data['data'][i]['paragraphs'][j]['context']
            paragraph_wordlist=tokenize(paragraph.strip())
            pos_list, ner_list= pos_and_ner(paragraph_wordlist, ner_tagger, pos2id, ner2id, pos_size, ner_size)

            for q in range(question_size_j):
                question_q=data['data'][i]['paragraphs'][j]['qas'][q]['question']
                question_wordlist=tokenize(question_q.strip())


#                 answer_q=data['data'][i]['paragraphs'][j]['qas'][q]['answers'][0]['text']
#                 answer_q_wordlist=tokenize(answer_q)
#                 answer_len=len(answer_q_wordlist)
# #                 answer_char_len=len(answer_q)
#                 answer_start_q=data['data'][i]['paragraphs'][j]['qas'][q]['answers'][0]['answer_start']
#                 while answer_start_q>0 and paragraph[answer_start_q-1]!=' ':
#                     answer_start_q-=1
#                 answer_left=paragraph[:answer_start_q]
#                 answer_left_wordlist=truncate_by_punct(tokenize(answer_left), True)
#                 answer_left_wordlist=tokenize(answer_left)
#                 answer_left_size=len(answer_left_wordlist)
#                 answer_right=paragraph[answer_start_q+answer_char_len:]
# #                 answer_right_wordlist=truncate_by_punct(tokenize(answer_right), False)
#                 answer_right_wordlist=tokenize(answer_right)
#                 answer_right_size=len(answer_right_wordlist)
                answer_no=len(data['data'][i]['paragraphs'][j]['qas'][q]['answers'])
                q_ansSet=set()
                for ans in range(answer_no):
                    answer_q=data['data'][i]['paragraphs'][j]['qas'][q]['answers'][ans]['text']
                    q_ansSet.add(' '.join(tokenize(answer_q.strip())))

#                 if len(gold_label_q)!=len(paragraph_wordlist):
# #                     print 'len(gold_label_q)!=len(paragraph_wordlist):', len(gold_label_q), len(paragraph_wordlist)
# #                     print 'paragraph:', paragraph
#                     noise+=1
#                     continue
#                     exit(0)

#                 paragraph_wordlist=answer_left_wordlist+answer_q_wordlist+answer_right_wordlist

#                 pos_list, ner_list= pos_and_ner(paragraph_wordlist, ner_tagger, pos2id, ner2id, pos_size, ner_size)


                #write into file
                writefile.write('W:\t'+' '.join(paragraph_wordlist)+'\n')
                writefile.write('P:\t'+' '.join(map(str,pos_list))+'\n')
                writefile.write('N:\t'+' '.join(map(str,ner_list))+'\n')
                writefile.write('A:\t'+'\t'.join(q_ansSet)+'\n')
                writefile.write('Q:\t'+' '.join(question_wordlist)+'\n')



            qa_size+=question_size_j
            print 'pair size:', qa_size#, 'noise:', noise
        para_size+=para_size_i

    writefile.close()
    print 'Store train set', para_size, 'paragraphs,', qa_size, 'question-answer pairs'

def process_tokens(temp_tokens):
    tokens = []
    for token in temp_tokens:
        token=token.replace("''", '"').replace("``", '"')
        flag = False
        l = ("-", "\u2212", "\u2014", "\u2013", "/", "~", '"', "'", "\u201C", "\u2019", "\u201D", "\u2018", "\u00B0")
        # \u2013 is en-dash. Used for number to nubmer
        # l = ("-", "\u2212", "\u2014", "\u2013")
        # l = ("\u2013",)
        tokens.extend(re.split("([{}])".format("".join(l)), token))
    return tokens
def  load_train_AI2(para_len_limit, q_len_limit):
    max_para_len=para_len_limit
    max_Q_len = q_len_limit

    word2id={}
#     read_file=open(path+'train-v1.0.json', 'r')
    with open(path+'train-v1.1.json') as data_file:
        data = json.load(data_file)

#     pprint(data['data'][0]['paragraphs'][0])
    doc_size=len(data['data'])
#     print 'doc_size:', doc_size
    para_size=0
    qa_size=0
    para_list=[]
    Q_list=[]
#     Q_size_list=[]
    start_list=[]
    end_list=[]
    para_mask=[]
    mask=[]
    feature_matrixlist=[]
    stop_words=load_stopwords()
    for i in range(doc_size):#each doc
        para_size_i=len(data['data'][i]['paragraphs'])
        for j in range(para_size_i):#each paragraph
            question_size_j=len(data['data'][i]['paragraphs'][j]['qas'])
#             Q_size_list.append(question_size_j)
            paragraph=data['data'][i]['paragraphs'][j]['context']#.replace("''", '"').replace("``", '"')
#             print 'paragraph:', paragraph
#             paragraph_wordlist=paragraph.strip().split()
#             paragraph_idlist=strs2ids(paragraph_wordlist, word2id)
#             para_len=len(paragraph_wordlist)

#             Q_sublist=[]
#             label_sublist=[]
#             feature_tensor=[]

#             max_q_len=0
            for q in range(question_size_j):
                question_q=data['data'][i]['paragraphs'][j]['qas'][q]['question']#.replace("''", '"').replace("``", '"')
                question_wordlist=tokenize(question_q.strip())


#                 feature_tensor.append(feature_matrix_q)

                question_idlist=strs2ids(question_wordlist, word2id)
                q_len=len(question_idlist)
#                 if len(question_idlist)>max_q_len:
#                     max_q_len=len(question_idlist)
                answer_q=data['data'][i]['paragraphs'][j]['qas'][q]['answers'][0]['text']
                answer_q_wordlist=tokenize(answer_q)
                answer_len=len(answer_q_wordlist)
                answer_start_q=data['data'][i]['paragraphs'][j]['qas'][q]['answers'][0]['answer_start']
#                 while answer_start_q>0 and paragraph[answer_start_q-1]!=' ':
#                     answer_start_q-=1
                answer_left=paragraph[:answer_start_q]
#                 answer_left_wordlist=truncate_by_punct(tokenize(answer_left), True)
                answer_left_wordlist=tokenize(answer_left)
                answer_left_size=len(answer_left_wordlist)
                answer_right=paragraph[answer_start_q+len(answer_q):]
#                 answer_right_wordlist=truncate_by_punct(tokenize(answer_right), False)
                answer_right_wordlist=tokenize(answer_right)
                answer_right_size=len(answer_right_wordlist)
                gold_label_q=[0]*answer_left_size+[1]*answer_len+[0]*answer_right_size

                para_len=answer_left_size+answer_len+answer_right_size
                paragraph_wordlist=answer_left_wordlist+answer_q_wordlist+answer_right_wordlist
                if len(paragraph_wordlist)!=para_len:
                    print 'len(paragraph_wordlist)!=para_len:', len(paragraph_wordlist),para_len
                    exit(0)
#                 print 'paragraph_wordlist:', paragraph_wordlist
#                 print 'question_wordlist:', question_wordlist
#                 exit(0)
                feature_matrix_q=extra_features(stop_words, paragraph_wordlist, question_wordlist)
                paragraph_idlist=strs2ids(paragraph_wordlist, word2id)
                #now, pad paragraph, question, feature_matrix, gold_label
                #first paragraph
                pad_para_len=max_para_len-para_len
                if pad_para_len>0:
                    paded_paragraph_idlist=[0]*pad_para_len+paragraph_idlist
                    paded_para_mask_i=[0.0]*pad_para_len+[1.0]*para_len
                    paded_feature_matrix_q=[[0]*3]*pad_para_len+feature_matrix_q
                    paded_gold_label=[0]*pad_para_len+gold_label_q
                else:
                    paded_paragraph_idlist=paragraph_idlist[:max_para_len]
                    paded_para_mask_i=([1.0]*para_len)[:max_para_len]
                    paded_feature_matrix_q=feature_matrix_q[:max_para_len]
                    paded_gold_label=gold_label_q[:max_para_len]
#                 if 1.0 not in set(paded_gold_label):
#                     print 'numpy.sum(numpy.asarray(paded_gold_label))<1'
#                     exit(0)
                if len(paded_paragraph_idlist)!=300:
                    print 'len(paded_paragraph_idlist)!=300:', len(paded_paragraph_idlist)
                    exit(0)
                para_list.append(paded_paragraph_idlist)
                para_mask.append(paded_para_mask_i)
                feature_matrixlist.append(paded_feature_matrix_q)
                start, end = binaryLabelList2StartEnd(paded_gold_label)
                start_list.append(start)
                end_list.append(end)
                #then question
                pad_q_len=max_Q_len-q_len
                if pad_q_len > 0:
                    paded_question_idlist=[0]*pad_q_len+question_idlist
                    paded_q_mask_i=[0.0]*pad_q_len+[1.0]*q_len
                else:
                    paded_question_idlist=question_idlist[:max_Q_len]
                    paded_q_mask_i=([1.0]*q_len)[:max_Q_len]
                Q_list.append(paded_question_idlist)
                mask.append(paded_q_mask_i)


#             submask=[]
#             Q_sublist_padded=[]
#             for orig_q in Q_sublist: # pad zero at end of sentences
#                 existing_len=len(orig_q)
#                 pad_len=max_q_len-existing_len
#                 if pad_len>0:
#                     orig_q+=[0]*pad_len
#                 Q_sublist_padded.append(orig_q)
#                 submask.append([1.0]*existing_len+[0.0]*pad_len)

#             for orig_q in Q_sublist: # pad zero at mid of sentences
#                 existing_len=len(orig_q)
#                 pad_len=max_q_len-existing_len
#                 if pad_len>0:
#                     mid_place=existing_len/2
#                     orig_q=orig_q[:mid_place]+[0]*pad_len+orig_q[mid_place:]
#                 Q_sublist_padded.append(orig_q)
#                 submask.append([1.0]*mid_place+[0.0]*pad_len+[1.0]*(existing_len-mid_place))


#             Q_list.append(Q_sublist_padded)
#             label_list.append(label_sublist)
#             mask.append(submask)
#             feature_tensorlist.append(feature_tensor)
#             print 'question_size_j:', question_size_j
            qa_size+=question_size_j
#         print 'para_size_i:', para_size_i
        para_size+=para_size_i
#     pprint(len(data['data']))
#     print data['data'][0]['paragraphs'][0]
    print 'Load train set', para_size, 'paragraphs,', qa_size, 'question-answer pairs'
    print 'Train Vocab size:', len(word2id)
    exit(0)
    return para_list, Q_list, start_list, end_list, para_mask, mask, word2id, feature_matrixlist

def  load_dev_or_test_AI2(word2id, para_len_limit, q_len_limit):
#     Dev  max_para_len:, 629 max_q_len: 33
#     read_file=open(path+'train-v1.0.json', 'r')
    max_para_len=para_len_limit
    max_Q_len = q_len_limit
    with open(path+'dev-v1.1.json') as data_file:
        data = json.load(data_file)

#     pprint(data['data'][0]['paragraphs'][0])
    doc_size=len(data['data'])
#     print 'doc_size:', doc_size

#     word2vec=load_word2vec()


    para_size=0
    qa_size=0
    para_list=[]
    Q_list=[]
    Q_list_word=[]
    para_mask=[]
    mask=[]
    feature_matrixlist=[]
    para_text_list=[]
    q_ansSet_list=[]
    stop_words=load_stopwords()
    id_list=[]
    for i in range(doc_size):#each doc
        para_size_i=len(data['data'][i]['paragraphs'])
        for j in range(para_size_i):#each paragraph
            question_size_j=len(data['data'][i]['paragraphs'][j]['qas']) #how many questions for this paragraph
#             Q_size_list.append(question_size_j)
            paragraph=data['data'][i]['paragraphs'][j]['context']#.replace("''", '"').replace("``", '"')
#             print 'paragraph:', paragraph
            paragraph_wordlist=tokenize(paragraph.strip())
#             para_text_list.append(paragraph_wordlist)
#             paragraph_idlist=strs2ids(paragraph_wordlist, word2id)
#             para_len=len(paragraph_wordlist)

#             Q_sublist=[]
#             label_sublist=[]
#             feature_tensor=[]
#             ansSetList=[]
#             max_q_len=0
            for q in range(question_size_j): # for each question
                question_q=data['data'][i]['paragraphs'][j]['qas'][q]['question']#.replace("''", '"').replace("``", '"')
                q_id = data['data'][i]['paragraphs'][j]['qas'][q]['id']
#                 print 'q_id:', q_id
                question_wordlist=tokenize(question_q.strip())
                truncate_paragraph_wordlist=paragraph_wordlist#truncate_paragraph_by_question(word2vec, paragraph_wordlist, question_wordlist, 1)  #paragraph_wordlist#
                truncate_paragraph_idlist=strs2ids(truncate_paragraph_wordlist, word2id)
                truncate_para_len=len(truncate_paragraph_wordlist)
                feature_matrix_q=extra_features(stop_words, truncate_paragraph_wordlist, question_wordlist)
#                 feature_tensor.append(feature_matrix_q)


                question_idlist=strs2ids(question_wordlist, word2id)
                q_len=len(question_idlist)
#                 if len(question_idlist)>max_q_len:
#                     max_q_len=len(question_idlist)

                answer_no=len(data['data'][i]['paragraphs'][j]['qas'][q]['answers'])
                q_ansSet=set()
                for ans in range(answer_no):
                    answer_q=data['data'][i]['paragraphs'][j]['qas'][q]['answers'][ans]['text']
                    q_ansSet.add(' '.join(tokenize(answer_q.strip())))
#                     answer_len=len(answer_q.strip().split())

#                     answer_start_q=data['data'][i]['paragraphs'][j]['qas'][q]['answers'][ans]['answer_start']
#                     while answer_start_q>0 and paragraph[answer_start_q-1]!=' ':
#                         answer_start_q-=1
#                     answer_left=paragraph[:answer_start_q]
#                     answer_left_size=len(answer_left.strip().split())
#                     gold_label_q=[-1.0]*answer_left_size+[1.0]*answer_len+[-1.0]*(para_len-answer_left_size-answer_len)
#                 ansSetList.append(q_ansSet)
#                 Q_sublist.append(question_idlist)
#                 if len(label_sublist)>=1 and len(gold_label_q)!=len(label_sublist[-1]):
#                     print 'wired size'
#                     print len(gold_label_q),len(label_sublist[-1])
#                     exit(0)
#                 label_sublist.append(gold_label_q)
                #now, pad paragraph, question, feature_matrix, gold_label
                #first paragraph
                pad_para_len=max_para_len-truncate_para_len
                if pad_para_len>0:
                    paded_paragraph_idlist=[0]*pad_para_len+truncate_paragraph_idlist
                    paded_para_mask_i=[0.0]*pad_para_len+[1.0]*truncate_para_len
                    paded_feature_matrix_q=[[0]*3]*pad_para_len+feature_matrix_q
                    paded_para_text=['UNK']*pad_para_len+truncate_paragraph_wordlist
                else:
                    paded_paragraph_idlist=truncate_paragraph_idlist[:max_para_len]
                    paded_para_mask_i=([1.0]*truncate_para_len)[:max_para_len]
                    paded_feature_matrix_q=feature_matrix_q[:max_para_len]
                    paded_para_text=truncate_paragraph_wordlist[:max_para_len]

#                 paded_paragraph_idlist=[0]*pad_para_len+paragraph_idlist
#                 paded_para_mask_i=[0.0]*pad_para_len+[1.0]*para_len
#                 paded_feature_matrix_q=[[0]*3]*pad_para_len+feature_matrix_q
#                 paded_para_text=['UNK']*pad_para_len+paragraph_wordlist
                para_list.append(paded_paragraph_idlist)
                para_mask.append(paded_para_mask_i)
                feature_matrixlist.append(paded_feature_matrix_q)
                para_text_list.append(paded_para_text)
                #then question
                pad_q_len=max_Q_len-q_len
                if pad_q_len > 0:
                    paded_question_idlist=[0]*pad_q_len+question_idlist
                    paded_q_mask_i=[0.0]*pad_q_len+[1.0]*q_len
                else:
                    paded_question_idlist=question_idlist[:max_Q_len]
                    paded_q_mask_i=([1.0]*q_len)[:max_Q_len]
#                 paded_question_idlist=[0]*pad_q_len+question_idlist
#                 paded_q_mask_i=[0.0]*pad_q_len+[1.0]*q_len
                Q_list.append(paded_question_idlist)
                Q_list_word.append(question_wordlist)
                mask.append(paded_q_mask_i)
                #then , store answers
                q_ansSet_list.append(q_ansSet)
                id_list.append(q_id)

            qa_size+=question_size_j
#         print 'para_size_i:', para_size_i
        para_size+=para_size_i
#     pprint(len(data['data']))
#     print data['data'][0]['paragraphs'][0]
    print 'Load dev set', para_size, 'paragraphs,', qa_size, 'question-answer pairs'
    print 'Train+Dev Vocab size:', len(word2id)
#     print word2id
    return para_list, Q_list, Q_list_word, para_mask, mask, len(word2id), word2id, para_text_list, q_ansSet_list, feature_matrixlist, id_list

# def load_SQUAD_hinrich_v2(example_no_limit, max_para_len, max_q_len, word2id, fil):
#     line_co=0
#     block_lines=10
#     example_co=0
#     readfile=open(fil, 'r')
#
#     questions=[]
#     questions_mask=[]
#     paras=[]
#     paras_mask=[]
#     c_heads=[]
#     c_tails=[]
#     e_heads=[]
#     e_tails=[]
#     labels=[]
#
#     cand=''
#     extend=''
#     context=''
#     label2co=defaultdict()
#     for line in readfile:
#         line_co+=1
#         if line_co % block_lines==0:
#             example_co+=1
#             cand=''
#             extend=''
#             context=''
#             if example_no_limit is not None and example_co == example_no_limit:
#                 break
#         if line_co%block_lines==1 or line_co%block_lines==3 or line_co%block_lines==7 or line_co%block_lines==8:
#             continue
#         else:
#             if line_co%block_lines==2:#question
#                 q_example=strlist_2_wordidlist(line.strip().split(), word2id)
#                 pad_q_example, q_mask=pad_idlist(q_example, max_q_len)
#                 questions.append(pad_q_example)
#                 questions_mask.append(q_mask)
#             elif line_co%block_lines==4: # cand
#                 cand=line.strip().split()[1:]
#             elif line_co%block_lines==5: # exted
#                 extend=line.strip().split()[1:]
#             elif line_co%block_lines==6: # context
#                 sub_line=line.strip().split()
#                 if len(sub_line)==1:
#                     context=['UNK']
#                 else:
#                     context=sub_line[1:]
#                 side_label=sub_line[0]
#                 if side_label =='L':
#                     paragraph=context+extend+cand
#                     raw_para_len= len(paragraph)
#                     c_head=raw_para_len-len(cand)
#                     c_tail=raw_para_len-1
#                     e_head=raw_para_len-len(extend+cand)
#                     e_tail = c_tail
#                     para_ids=strlist_2_wordidlist(paragraph, word2id)
#                     pad_para, para_mask,para_pad_size=leftpad_idlist_padsize(para_ids, max_para_len)
#                     c_head+=para_pad_size
#                     c_tail+=para_pad_size
#                     e_head+=para_pad_size
#                     e_tail+=para_pad_size
#                     if c_head>=max_para_len or c_tail>=max_para_len or e_head>=max_para_len or e_tail>=max_para_len:
#                         print 'c_head>=max_para_len or c_tail>=max_para_len or e_head>=max_para_len or e_tail>=max_para_len:', c_head,c_tail,e_head,e_tail,max_para_len
#                         exit(0)
#                     if c_head<0 or c_tail<0 or e_head<0 or e_tail<0:
#                         print 'L:'
#                         print 'c_head<0 or c_tail<0 or e_head<0 or e_tail<0:', c_head,c_tail,e_head,e_tail,max_para_len
#                         print 'len(para_ids):',len(para_ids),'para_pad_size:',para_pad_size
#                         print len(context), context
#                         print len(extend), extend
#                         print len(cand), cand
#                         exit(0)
#                 elif side_label =='R':
#                     paragraph=cand+extend+context
#                     raw_para_len= len(paragraph)
#                     c_head=0#raw_para_len-len(cand)
#                     c_tail=len(cand)-1
#                     e_head=0
#                     e_tail = len(cand+extend)-1
#                     para_ids=strlist_2_wordidlist(paragraph, word2id)
#                     pad_para, para_mask,para_pad_size=rightpad_idlist_padsize(para_ids, max_para_len)
#                     if c_head>=max_para_len or c_tail>=max_para_len or e_head>=max_para_len or e_tail>=max_para_len:
#                         print 'c_head>=max_para_len or c_tail>=max_para_len or e_head>=max_para_len or e_tail>=max_para_len:', c_head,c_tail,e_head,e_tail,max_para_len
#                         exit(0)
#                     if c_head<0 or c_tail<0 or e_head<0 or e_tail<0:
#                         print 'R:'
#                         print 'c_head<0 or c_tail<0 or e_head<0 or e_tail<0:', c_head,c_tail,e_head,e_tail,max_para_len
#                         print 'len(para_ids):',len(para_ids),'para_pad_size:',para_pad_size
#                         print len(context), context
#                         print len(extend), extend
#                         print len(cand), cand
#                         exit(0)
#                 else:
#                     print 'unknown label:', line
#                     exit(0)
#                 #load
#                 paras.append(pad_para)
#                 paras_mask.append(para_mask)
#                 c_heads.append(c_head)
#                 c_tails.append(c_tail)
#                 e_heads.append(e_head)
#                 e_tails.append(e_tail)
#
#
#             elif line_co%block_lines ==9:#label
#                 label_str=line.strip().split()[-1]
#                 if label_str =='GOOD':
#                     label=1
#                 elif label_str =='BAD':
#                     label=0
#                 else:
#                     print 'no valid label line:', line
#                     exit(0)
#                 labels.append(label)
#
#         # print line_co
# #     if example_co != example_no_limit:
# #         print 'example_co != example_no_limit:', example_co,example_no_limit
# #         exit(0)
#     readfile.close()
#     print 'load', example_co, 'samples finished'
#
#     return     word2id, questions,questions_mask,paras,paras_mask,c_heads,c_tails,e_heads,e_tails,labels

def load_SQUAD_hinrich_v2(example_no_limit, max_para_len, max_q_len, e_len, c_len, word2id, fil):
    line_co=0
    block_lines=9
    example_co=0
    readfile=open(fil, 'r')

    questions=[]
    questions_mask=[]
    paras=[]
    paras_mask=[]
    e_ids=[]
    e_masks=[]
    c_ids=[]
    c_masks=[]
    c_heads=[]
    c_tails=[]
    l_heads=[]
    l_tails=[]
    e_heads=[]
    e_tails=[]
    labels=[]
    labels_3c=[]

    cand=''
    extend=''
    context=''
    label2co=defaultdict(int)
    for line in readfile:
        if len(line.strip())==0:
            continue
        line_co+=1
#         print line_co, line
        if line_co % block_lines==1 and line_co>1:
            example_co+=1
            cand=''
            extend=''
            context=''
            if example_no_limit is not None and example_co == example_no_limit:
                break
        if line_co%block_lines==1 or line_co%block_lines==3 or line_co%block_lines==7 or line_co%block_lines==8:
            continue
        else:
            if line_co%block_lines==2:#question
                q_example=strlist_2_wordidlist(line.strip().split(), word2id)
                pad_q_example, q_mask=pad_idlist(q_example, max_q_len)
                questions.append(pad_q_example)
                questions_mask.append(q_mask)
            elif line_co%block_lines==4: # cand
                cand=line.strip().split()[1:]
                cand_example=strlist_2_wordidlist(cand, word2id)
                pad_cand_example, cand_mask=pad_idlist(cand_example, c_len)
                c_ids.append(pad_cand_example)
                c_masks.append(cand_mask)
            elif line_co%block_lines==5: # exted
                extend=line.strip().split()[1:]
                extend_example=strlist_2_wordidlist(extend, word2id)
                pad_extend_example, extend_mask=pad_idlist(extend_example, e_len)
                e_ids.append(pad_extend_example)
                e_masks.append(extend_mask)
            elif line_co%block_lines==6: # context
                sub_line=line.strip().split()
                if len(sub_line)==1:
                    context=['UNK']
                else:
                    context=sub_line[1:]
                side_label=sub_line[0]
                if side_label =='L':
                    paragraph=context+extend+cand
                    raw_para_len= len(paragraph)
                    c_head=raw_para_len-len(cand)
                    c_tail=raw_para_len-1
                    l_head=raw_para_len-len(extend+cand)
                    l_tail = c_tail
                    e_head = l_head
                    e_tail = c_head-1
                    para_ids=strlist_2_wordidlist(paragraph, word2id)
                    pad_para, para_mask,para_pad_size=leftpad_idlist_padsize(para_ids, max_para_len)
                    c_head+=para_pad_size
                    c_tail+=para_pad_size
                    l_head+=para_pad_size
                    l_tail+=para_pad_size
                    e_head+=para_pad_size
                    e_tail+=para_pad_size
                    if c_head>=max_para_len or c_tail>=max_para_len or l_head>=max_para_len or l_tail>=max_para_len or e_head>=max_para_len or e_tail>=max_para_len:
                        print 'c_head>=max_para_len or c_tail>=max_para_len or l_head>=max_para_len or l_tail>=max_para_len:', c_head,c_tail,l_head,l_tail,max_para_len
                        exit(0)
                    if c_head<0 or c_tail<0 or l_head<0 or l_tail<0 or e_head<0 or e_tail<0:
                        print 'L:'
                        print 'c_head<0 or c_tail<0 or l_head<0 or l_tail<0:', c_head,c_tail,l_head,l_tail,max_para_len
                        print 'len(para_ids):',len(para_ids),'para_pad_size:',para_pad_size
                        print len(context), context
                        print len(extend), extend
                        print len(cand), cand
                        exit(0)
                elif side_label =='R':
                    paragraph=cand+extend+context
                    raw_para_len= len(paragraph)
                    c_head=0#raw_para_len-len(cand)
                    c_tail=len(cand)-1
                    l_head=0
                    l_tail = len(cand+extend)-1
                    e_head = c_tail+1
                    e_tail = l_tail
                    para_ids=strlist_2_wordidlist(paragraph, word2id)
                    pad_para, para_mask,para_pad_size=rightpad_idlist_padsize(para_ids, max_para_len)
                    if c_head>=max_para_len or c_tail>=max_para_len or l_head>=max_para_len or l_tail>=max_para_len or e_head>=max_para_len or e_tail>=max_para_len:
                        print 'c_head>=max_para_len or c_tail>=max_para_len or l_head>=max_para_len or l_tail>=max_para_len:', c_head,c_tail,l_head,l_tail,max_para_len
                        exit(0)
                    if c_head<0 or c_tail<0 or l_head<0 or l_tail<0 or e_head<0 or e_tail<0:
                        print 'R:'
                        print 'c_head<0 or c_tail<0 or l_head<0 or l_tail<0:', c_head,c_tail,l_head,l_tail,max_para_len
                        print 'len(para_ids):',len(para_ids),'para_pad_size:',para_pad_size
                        print len(context), context
                        print len(extend), extend
                        print len(cand), cand
                        exit(0)
                else:
                    print 'unknown label:', line
                    exit(0)
                #load
                paras.append(pad_para)
                paras_mask.append(para_mask)
                c_heads.append(c_head)
                c_tails.append(c_tail)
                l_heads.append(l_head)
                l_tails.append(l_tail)
                e_heads.append(e_head)
                e_tails.append(e_tail)

            elif line_co%block_lines ==0:#label
                label_str=line.strip().split()[-1]
                if label_str =='GOOD':
                    label=1
                    label_3d=1
                elif label_str =='BAD':
                    label=0
                    label_3d=0
                else:
                    label=0
                    label_3d=2

                labels.append(label)
                labels_3c.append(label_3d)
                label2co[label]+=1
        # print line_co
#     if example_co != example_no_limit:
#         print 'example_co != example_no_limit:', example_co,example_no_limit
#         exit(0)
    readfile.close()
    print 'load', example_co, 'samples finished, majority rate:', label2co.get(1)+label2co.get(0), label2co.get(1)*1.0/(label2co.get(1)+label2co.get(0)), label2co.get(0)*1.0/(label2co.get(1)+label2co.get(0))

    return     word2id, questions,questions_mask,paras,paras_mask,e_ids,e_masks,c_ids,c_masks,c_heads,c_tails,l_heads,l_tails,e_heads,e_tails, labels, labels_3c

def strlist2str(strlist):
    return ' '.join(strlist)

def load_SQUAD_hinrich_v3(example_no_limit, max_para_len, max_q_len, e_len, c_len, char2id, fil):
    line_co=0
    block_lines=9
    example_co=0
    readfile=open(fil, 'r')

    questions=[]
    questions_mask=[]
    paras=[]
    paras_mask=[]
    e_ids=[]
    e_masks=[]
    c_ids=[]
    c_masks=[]
    c_heads=[]
    c_tails=[]
    l_heads=[]
    l_tails=[]
    e_heads=[]
    e_tails=[]
    labels=[]
    labels_3c=[]

    cand=''
    extend=''
    context=''
    label2co=defaultdict(int)
    for line in readfile:
        if len(line.strip())==0:
            continue
        line_co+=1
#         print line_co, line
        if line_co % block_lines==1 and line_co>1:
            example_co+=1
            cand=''
            extend=''
            context=''
            if example_no_limit is not None and example_co == example_no_limit:
                break
        if line_co%block_lines==1 or line_co%block_lines==3 or line_co%block_lines==7 or line_co%block_lines==8:
            continue
        else:
            if line_co%block_lines==2:#question
                q_example=str_2_charidlist(line.strip(), char2id)
                pad_q_example, q_mask=pad_idlist(q_example, max_q_len)
                questions.append(pad_q_example)
                questions_mask.append(q_mask)
            elif line_co%block_lines==4: # cand
                cand=line.strip()[2:]
                cand_example=str_2_charidlist(cand, char2id)
                pad_cand_example, cand_mask=pad_idlist(cand_example, c_len)
                c_ids.append(pad_cand_example)
                c_masks.append(cand_mask)
            elif line_co%block_lines==5: # exted
                extend=line.strip()[2:]
                extend_example=str_2_charidlist(extend, char2id)
                pad_extend_example, extend_mask=pad_idlist(extend_example, e_len)
                e_ids.append(pad_extend_example)
                e_masks.append(extend_mask)
            elif line_co%block_lines==6: # context
                sub_line=line.strip()
                if len(sub_line)==1:
                    context='UNK'
                else:
                    context=sub_line[2:]
                side_label=sub_line[0]
                if side_label =='L':
                    paragraph=context+' '+extend+' '+cand  #str
                    raw_para_len= len(paragraph) # char size
                    c_head=raw_para_len-len(cand)
                    c_tail=raw_para_len-1
                    l_head=raw_para_len-len(extend+' '+cand)
                    l_tail = c_tail
                    e_head = l_head
                    e_tail = c_head-2 # remove the white space
                    para_ids=str_2_charidlist(paragraph, char2id)
                    pad_para, para_mask,para_pad_size=leftpad_idlist_padsize(para_ids, max_para_len)
                    c_head+=para_pad_size
                    c_tail+=para_pad_size
                    l_head+=para_pad_size
                    l_tail+=para_pad_size
                    e_head+=para_pad_size
                    e_tail+=para_pad_size
                    if c_head>=max_para_len or c_tail>=max_para_len or l_head>=max_para_len or l_tail>=max_para_len or e_head>=max_para_len or e_tail>=max_para_len:
                        print 'c_head>=max_para_len or c_tail>=max_para_len or l_head>=max_para_len or l_tail>=max_para_len:', c_head,c_tail,l_head,l_tail,max_para_len
                        exit(0)
                    if c_head<0 or c_tail<0 or l_head<0 or l_tail<0 or e_head<0 or e_tail<0:
                        print 'L:'
                        print 'c_head<0 or c_tail<0 or l_head<0 or l_tail<0:', c_head,c_tail,l_head,l_tail,max_para_len
                        print 'len(para_ids):',len(para_ids),'para_pad_size:',para_pad_size
                        print len(context), context
                        print len(extend), extend
                        print len(cand), cand
                        exit(0)
                elif side_label =='R':
                    paragraph=cand+' '+extend+' '+context
                    raw_para_len= len(paragraph)
                    c_head=0#raw_para_len-len(cand)
                    c_tail=len(cand)-1
                    l_head=0
                    l_tail = len(cand+' '+extend)-1
                    e_head = c_tail+2 # a whole space betwen
                    e_tail = l_tail
                    para_ids=str_2_charidlist(paragraph, char2id)
                    pad_para, para_mask,para_pad_size=rightpad_idlist_padsize(para_ids, max_para_len)
                    if c_head>=max_para_len or c_tail>=max_para_len or l_head>=max_para_len or l_tail>=max_para_len or e_head>=max_para_len or e_tail>=max_para_len:
                        print 'c_head>=max_para_len or c_tail>=max_para_len or l_head>=max_para_len or l_tail>=max_para_len:', c_head,c_tail,l_head,l_tail,max_para_len
                        exit(0)
                    if c_head<0 or c_tail<0 or l_head<0 or l_tail<0 or e_head<0 or e_tail<0:
                        print 'R:'
                        print 'c_head<0 or c_tail<0 or l_head<0 or l_tail<0:', c_head,c_tail,l_head,l_tail,max_para_len
                        print 'len(para_ids):',len(para_ids),'para_pad_size:',para_pad_size
                        print len(context), context
                        print len(extend), extend
                        print len(cand), cand
                        exit(0)
                else:
                    print 'unknown label:', line
                    exit(0)
                #load
                paras.append(pad_para)
                paras_mask.append(para_mask)
                c_heads.append(c_head)
                c_tails.append(c_tail)
                l_heads.append(l_head)
                l_tails.append(l_tail)
                e_heads.append(e_head)
                e_tails.append(e_tail)

            elif line_co%block_lines ==0:#label
                label_str=line.strip().split()[-1]
                if label_str =='GOOD':
                    label=1
                    label_3d=1
                elif label_str =='BAD':
                    label=0
                    label_3d=0
                else:
                    label=0
                    label_3d=2

                labels.append(label)
                labels_3c.append(label_3d)
                label2co[label]+=1
        # print line_co
#     if example_co != example_no_limit:
#         print 'example_co != example_no_limit:', example_co,example_no_limit
#         exit(0)
    readfile.close()
    print 'load', example_co, 'samples finished, majority rate:', label2co.get(1)+label2co.get(0), label2co.get(1)*1.0/(label2co.get(1)+label2co.get(0)), label2co.get(0)*1.0/(label2co.get(1)+label2co.get(0))

    return     char2id, questions,questions_mask,paras,paras_mask,e_ids,e_masks,c_ids,c_masks,c_heads,c_tails,l_heads,l_tails,e_heads,e_tails, labels, labels_3c


def  load_train_reformed_BIO(para_len_limit, q_len_limit):
    max_para_len=para_len_limit
    max_Q_len = q_len_limit
    word2id={}
    read_file=codecs.open(path+'train-reformed.txt', 'r', 'utf-8')


    qa_size=0
    para_list=[]
    Q_list=[]
#     Q_size_list=[]
    label_list=[]
    para_mask=[]
    mask=[]
#     feature_matrixlist=[]
#     pos_matrixlist=[]
#     ner_matrixlist=[]
#     stop_words=load_stopwords()
#     size_control=70000
    past_tag=''
    for line in read_file:
        parts=line.strip().split('\t')
        if parts[0]=='W:':#is paragraph
            paragraph_wordlist=parts[1].split()
            paragraph_idlist=strs2ids(paragraph_wordlist, word2id)
            para_len=len(paragraph_idlist)
            past_tag=''
            continue
        if parts[0]=='P:':#is POS
#             pos_list=map(int,parts[1].split())
            past_tag=''
            continue
        if parts[0]=='N:':#is NER
#             ner_list=map(int,parts[1].split())
            past_tag=''
            continue
        if parts[0]=='L:':#is labels
            gold_label_q=binary_label_2_BIO(map(int,parts[1].split()))
            past_tag=''
        if parts[0]=='Q:':#is question
            question_wordlist=parts[1].split()
            question_idlist=strs2ids(question_wordlist, word2id)
            q_len=len(question_idlist)
            past_tag='Q'

        if past_tag =='Q': #store

            if para_len != len(gold_label_q):
                continue
#             feature_matrix_q=extra_features(stop_words, paragraph_wordlist, question_wordlist)  #(para_len, 3)
#             pos_feature_matrix, ner_feature_matrix= poslist_nerlist_2_featurematrix(pos_list, ner_list, pos_size, ner_size)

            #now, pad paragraph, question, feature_matrix, gold_label
            #first paragraph
            pad_para_len=max_para_len-para_len
            if pad_para_len>0:
                paded_paragraph_idlist=[0]*pad_para_len+paragraph_idlist
                paded_para_mask_i=[0.0]*pad_para_len+[1.0]*para_len

#                 paded_feature_matrix_q=[[0]*3]*pad_para_len+feature_matrix_q
#                 paded_pos_feature_matrix=[[0.0]*pos_size]*pad_para_len+pos_feature_matrix
#                 paded_ner_feature_matrix=[[0.0]*ner_size]*pad_para_len+ner_feature_matrix
                paded_gold_label=[0]*pad_para_len+gold_label_q
            else:
                paded_paragraph_idlist=paragraph_idlist[:max_para_len]
                paded_para_mask_i=[1.0]*max_para_len
#                 paded_feature_matrix_q=feature_matrix_q[:max_para_len]
#                 paded_pos_feature_matrix=pos_feature_matrix[:max_para_len]
#                 paded_ner_feature_matrix=ner_feature_matrix[:max_para_len]
                paded_gold_label=gold_label_q[:max_para_len]
#                 if 1.0 not in set(paded_gold_label):
#                     print 'numpy.sum(numpy.asarray(paded_gold_label))<1'
#                     exit(0)
            para_list.append(paded_paragraph_idlist)
            para_mask.append(paded_para_mask_i)
#             feature_matrixlist.append(paded_feature_matrix_q)
#             pos_matrixlist.append(paded_pos_feature_matrix)
#             ner_matrixlist.append(paded_ner_feature_matrix)
            label_list.append(paded_gold_label)
            #then question
            pad_q_len=max_Q_len-q_len
            if pad_q_len > 0:
                paded_question_idlist=[0]*pad_q_len+question_idlist
                paded_q_mask_i=[0.0]*pad_q_len+[1.0]*q_len
            else:
                paded_question_idlist=question_idlist[:max_Q_len]
                paded_q_mask_i=[1.0]*max_Q_len
            Q_list.append(paded_question_idlist)
            mask.append(paded_q_mask_i)

            qa_size+=1
#             if qa_size == size_control:
#                 break



    print 'Load train set', qa_size, 'question-answer pairs'
    print 'Train Vocab size:', len(word2id)
#     exit(0)
    return para_list, Q_list, label_list, para_mask, mask, word2id#, feature_matrixlist, pos_matrixlist, numpy.asarray(ner_matrixlist)

def  load_dev_reformed_BIO(word2id, para_len_limit, q_len_limit):
#     Dev  max_para_len:, 629 max_q_len: 33
#     read_file=open(path+'train-v1.0.json', 'r')
    max_para_len=para_len_limit
    max_Q_len = q_len_limit
    read_file=codecs.open(path+'dev-reformed.txt', 'r', 'utf-8')

    qa_size=0
    para_list=[]
    Q_list=[]
    para_mask=[]
    mask=[]
    para_text_list=[]
    q_ansSet_list=[]

    past_tag=''
    for line in read_file:
        parts=line.strip().split('\t')
        if parts[0]=='W:':#is paragraph
            paragraph_wordlist=parts[1].split()
#             paragraph_idlist=strs2ids(paragraph_wordlist, word2id)
#             para_len=len(paragraph_idlist)
            past_tag=''
        if parts[0]=='P:':#is POS
#             pos_list=map(int,parts[1].split())
            past_tag=''
            continue
        if parts[0]=='N:':#is NER
#             ner_list=map(int,parts[1].split())
            past_tag=''
            continue
        if parts[0]=='A:':#is labels
#             gold_label_q=map(int,parts[1].split())
            q_ansSet=set()
            for i in range(1, len(parts)):
                q_ansSet.add(parts[i])
            past_tag=''
            continue
        if parts[0]=='Q:':#is question
            question_wordlist=parts[1].split()
            question_idlist=strs2ids(question_wordlist, word2id)
            q_len=len(question_idlist)
            past_tag='Q'

        if past_tag =='Q': #store


#             truncate_paragraph_wordlist, sentB_list=truncate_paragraph_by_question_returnBounary(word2vec, paragraph_wordlist, question_wordlist, 1)
#                 truncate_paragraph_wordlist = paragraph_wordlist
            paragraph_idlist=strs2ids(paragraph_wordlist, word2id)
            para_len=len(paragraph_idlist)

            #first paragraph
            pad_para_len=max_para_len-para_len
            if pad_para_len>0:
                paded_paragraph_idlist=[0]*pad_para_len+paragraph_idlist
                paded_para_mask_i=[0.0]*pad_para_len+[1.0]*para_len
                paded_para_text=['UNK']*pad_para_len+paragraph_wordlist
            else:
                paded_paragraph_idlist=paragraph_idlist[:max_para_len]
                paded_para_mask_i=[1.0]*max_para_len
                paded_para_text=paragraph_wordlist[:max_para_len]

            para_list.append(paded_paragraph_idlist)
            para_mask.append(paded_para_mask_i)
            para_text_list.append(paded_para_text)
            #then question
            pad_q_len=max_Q_len-q_len
            if pad_q_len > 0:
                paded_question_idlist=[0]*pad_q_len+question_idlist
                paded_q_mask_i=[0.0]*pad_q_len+[1.0]*q_len
            else:
                paded_question_idlist=question_idlist[:max_Q_len]
                paded_q_mask_i=[1.0]*max_Q_len
#                 paded_question_idlist=[0]*pad_q_len+question_idlist
#                 paded_q_mask_i=[0.0]*pad_q_len+[1.0]*q_len
            Q_list.append(paded_question_idlist)
            mask.append(paded_q_mask_i)
            #then , store answers
            q_ansSet_list.append(q_ansSet)

            qa_size+=1

    print 'Load dev set', qa_size, 'question-answer pairs'
    print 'Train+Dev Vocab size:', len(word2id)
#     print word2id
    return para_list, Q_list, para_mask, mask, word2id, para_text_list, q_ansSet_list#, feature_matrixlist, pos_matrixlist, ner_matrixlist

def binary_label_2_BIO(label_list):
    meet_1=False
    new_list=[]
    for label in label_list:
        if label ==0:
            new_list.append(label)
        else:#is 1
            if meet_1:
                new_list.append(1)
            else:
                new_list.append(2)
                meet_1=True
    return new_list

def  load_train_reformed_BIO4SpanRank(para_len_limit, q_len_limit):
    max_para_len=para_len_limit
    max_Q_len = q_len_limit
    word2id={}
    read_file=codecs.open(path+'train-reformed.txt', 'r', 'utf-8')


    qa_size=0
    para_list=[]
    Q_list=[]
#     Q_size_list=[]
    label_list=[]
    para_mask=[]
    mask=[]
#     feature_matrixlist=[]
#     pos_matrixlist=[]
#     ner_matrixlist=[]
#     stop_words=load_stopwords()
#     size_control=70000
    past_tag=''
    for line in read_file:
        parts=line.strip().split('\t')
        if parts[0]=='W:':#is paragraph
            paragraph_wordlist=parts[1].split()
            paragraph_idlist=strs2ids(paragraph_wordlist, word2id)
            para_len=len(paragraph_idlist)
            past_tag=''
            continue
        if parts[0]=='P:':#is POS
#             pos_list=map(int,parts[1].split())
            past_tag=''
            continue
        if parts[0]=='N:':#is NER
#             ner_list=map(int,parts[1].split())
            past_tag=''
            continue
        if parts[0]=='L:':#is labels
            gold_label_q=binary_label_2_BIO(map(int,parts[1].split()))
            past_tag=''
        if parts[0]=='Q:':#is question
            question_wordlist=parts[1].split()
            question_idlist=strs2ids(question_wordlist, word2id)
            q_len=len(question_idlist)
            past_tag='Q'

        if past_tag =='Q': #store

            if para_len != len(gold_label_q):
                continue
#             feature_matrix_q=extra_features(stop_words, paragraph_wordlist, question_wordlist)  #(para_len, 3)
#             pos_feature_matrix, ner_feature_matrix= poslist_nerlist_2_featurematrix(pos_list, ner_list, pos_size, ner_size)

            #now, pad paragraph, question, feature_matrix, gold_label
            #first paragraph
            pad_para_len=max_para_len-para_len
            if pad_para_len>0:
                paded_paragraph_idlist=[0]*pad_para_len+paragraph_idlist
                paded_para_mask_i=[0.0]*pad_para_len+[1.0]*para_len

#                 paded_feature_matrix_q=[[0]*3]*pad_para_len+feature_matrix_q
#                 paded_pos_feature_matrix=[[0.0]*pos_size]*pad_para_len+pos_feature_matrix
#                 paded_ner_feature_matrix=[[0.0]*ner_size]*pad_para_len+ner_feature_matrix
                paded_gold_label=[0]*pad_para_len+gold_label_q
            else:
                paded_paragraph_idlist=paragraph_idlist[:max_para_len]
                paded_para_mask_i=[1.0]*max_para_len
#                 paded_feature_matrix_q=feature_matrix_q[:max_para_len]
#                 paded_pos_feature_matrix=pos_feature_matrix[:max_para_len]
#                 paded_ner_feature_matrix=ner_feature_matrix[:max_para_len]
                paded_gold_label=gold_label_q[:max_para_len]
#                 if 1.0 not in set(paded_gold_label):
#                     print 'numpy.sum(numpy.asarray(paded_gold_label))<1'
#                     exit(0)
            para_list.append(paded_paragraph_idlist)
            para_mask.append(paded_para_mask_i)
#             feature_matrixlist.append(paded_feature_matrix_q)
#             pos_matrixlist.append(paded_pos_feature_matrix)
#             ner_matrixlist.append(paded_ner_feature_matrix)
#             label_list.append(paded_gold_label)
            label_list.append(binaryLabelList2Value(paded_gold_label))
            #then question
            pad_q_len=max_Q_len-q_len
            if pad_q_len > 0:
                paded_question_idlist=[0]*pad_q_len+question_idlist
                paded_q_mask_i=[0.0]*pad_q_len+[1.0]*q_len
            else:
                paded_question_idlist=question_idlist[:max_Q_len]
                paded_q_mask_i=[1.0]*max_Q_len
            Q_list.append(paded_question_idlist)
            mask.append(paded_q_mask_i)

            qa_size+=1
#             if qa_size == size_control:
#                 break



    print 'Load train set', qa_size, 'question-answer pairs'
    print 'Train Vocab size:', len(word2id)
#     exit(0)
    return para_list, Q_list, label_list, para_mask, mask, word2id#, feature_matrixlist, pos_matrixlist, numpy.asarray(ner_matrixlist)


def strlist2shapelist(strlist):
    newlist=[]
    for word in strlist:
        word = re.sub(r'[A-Z]', 'X', word)
        word = re.sub(r'[a-z]', 'x', word)
        word = re.sub('\d', '9', word) # digits to 9
        word = ''.join(ch for ch, _ in itertools.groupby(word))
        newlist.append(word)
    return newlist

def str2shape(word):

    word = re.sub(r'[A-Z]', 'X', word)
    word = re.sub(r'[a-z]', 'x', word)
    word = re.sub('\d', '9', word) # digits to 9
    word = ''.join(ch for ch, _ in itertools.groupby(word))
    return word

def wordstr2trichar(wordstr):
    results=[]
    for i in range(len(wordstr)-3):
        results.append(wordstr[i:i+3])
    return results


def wordlist2tricharIDlist(wordlist, trichar2id,max_q_len, max_trichar_len):
    idlist=[]
    masks=[]
    for id, word in enumerate(wordlist):
        if id < max_q_len:#valid word
            trichar_sequence = wordstr2trichar(word)
            trichar_idlist = elelist_2_idlist(trichar_sequence, trichar2id)
            pad_trichar_idlist, trichar_mask=pad_idlist(trichar_idlist, max_trichar_len)
            idlist+=pad_trichar_idlist
            masks+=trichar_mask

    pad_word_size = max_q_len - len(idlist)/max_trichar_len
    if pad_word_size>0:
        padlist = [0]*(max_trichar_len*pad_word_size)
        idlist=padlist + idlist
        masks = padlist + masks
    return idlist, masks




def load_SQUAD_hinrich_v4(example_no_limit, max_para_len, max_q_len, max_trichar_len, word2id, id2word, trichar2id,charLen,fil):
    line_co=0
    block_lines=16
    example_co=0
    readfile=open(fil, 'r')

    questions=[]
    questions_shape=[]
    questions_mask=[]
    paras=[]
    paras_shape=[]
    paras_mask=[]
    types=[]
    types_shape=[]


    labels=[]
    isInQ_paras=[]#is a matrix, left context, cand, right context
    distance_ls=[]
    distance_rs=[]

    question_trichar_ids=[]
    question_trichar_masks=[]
    para_trichar_ids=[]
    para_trichar_masks=[]
    type_trichar_ids=[]
    type_trichar_masks=[]

    question_prefix_ids=[]
    question_prefix_mask=[]
    question_suffix_ids=[]
    question_suffix_mask=[]

    para_prefix_ids=[]
    para_prefix_mask=[]
    para_suffix_ids=[]
    para_suffix_mask=[]


    paragraph=[]#is a matrix, left context, cand, right context
    question=[]
    type=[]
    isInQ_para=[]
    distance_l=[]
    distance_r=[]
    label2co=defaultdict(int)


    for line in readfile:
#         line=readfile.readline()

        if len(line.strip())==0:
            continue
        line_co+=1
#         if example_co >=487859:
#             print 'line_co:', line_co, ':', line.strip()
        if line_co % block_lines==1 and line_co>1:  #store existed data into matrix
            #preprocess
            #question, we need word id, word shape, word trichar, word prefix/suffix
            q_example=strlist_2_wordidlist(question, word2id, id2word)
            q_example_shape = strlist_2_wordidlist(strlist2shapelist(question), word2id, id2word)
            q_trichar_idlist, q_trichar_masks = wordlist2tricharIDlist(question, trichar2id,max_q_len, max_trichar_len)

            pad_q_example, q_mask=pad_idlist(q_example, max_q_len)
            q_prefixIDlist, q_suffixIDlist, q_prefixmasklist, q_suffixmasklist = idlist_2_charShape(pad_q_example, q_mask, id2word, trichar2id, charLen)
            pad_q_example_shape, _ =pad_idlist(q_example_shape, max_q_len)

            questions.append(pad_q_example)
            questions_shape.append(pad_q_example_shape)
            questions_mask.append(q_mask)
            question_trichar_ids.append(q_trichar_idlist)
            question_trichar_masks.append(q_trichar_masks)
            question_prefix_ids.append(q_prefixIDlist)
            question_prefix_mask.append(q_prefixmasklist)
            question_suffix_ids.append(q_suffixIDlist)
            question_suffix_mask.append(q_suffixmasklist)

            #paragraph, we need truncate, word id, shape, word trichar
            context_size= (max_para_len-1)/2

            if len(paragraph[0])+len(paragraph[1])+len(paragraph[2]) != len(isInQ_para[0])+len(isInQ_para[1])+len(isInQ_para[2]):
                print 'len(paragraph[0])+len(paragraph[1])+len(paragraph[2]) != len(isInQ_para[0])+len(isInQ_para[1])+len(isInQ_para[2]):', len(paragraph[0])+len(paragraph[1])+len(paragraph[2]), len(isInQ_para[0])+len(isInQ_para[1])+len(isInQ_para[2])
                exit(0)
            if len(paragraph[0])+len(paragraph[1])+len(paragraph[2]) != len(distance_l[0])+len(distance_l[1])+len(distance_l[2]):
                print 'len(paragraph[0])+len(paragraph[1])+len(paragraph[2]) != len(distance_l[0])+len(distance_l[1])+len(distance_l[2]):', len(paragraph[0])+len(paragraph[1])+len(paragraph[2]), len(distance_l[0])+len(distance_l[1])+len(distance_l[2])
                exit(0)
            leftpad_size=  context_size - len(paragraph[0]) if context_size > len(paragraph[0]) else 0
            rightpad_size=  context_size - len(paragraph[2]) if context_size > len(paragraph[2]) else 0

            new_para=(['UNK']*context_size+paragraph[0])[-context_size:]+paragraph[1]+(paragraph[2]+['UNK']*context_size)[:context_size] #totally 50+1+50 words
            new_isInQ=([0]*context_size+isInQ_para[0])[-context_size:]+isInQ_para[1]+(isInQ_para[2]+[0]*context_size)[:context_size]  #totally 50+1+50 labels
            new_distance_l=([0]*context_size+distance_l[0])[-context_size:]+distance_l[1]+(distance_l[2]+[0]*context_size)[:context_size]  #totally 50+1+50 labels
            new_distance_r=([0]*context_size+distance_r[0])[-context_size:]+distance_r[1]+(distance_r[2]+[0]*context_size)[:context_size]  #totally 50+1+50 labels
            para_mask = [0.0]* leftpad_size+[1.0]*(max_para_len-leftpad_size-rightpad_size) +[0.0]*rightpad_size

            para_ids=strlist_2_wordidlist(new_para, word2id, id2word)
            para_ids_shape=strlist_2_wordidlist(strlist2shapelist(new_para), word2id, id2word)
            para_trichar_idlist, para_trichar_mask = wordlist2tricharIDlist(new_para, trichar2id,max_para_len, max_trichar_len)
            pad_para_shape, _,_=leftpad_idlist_padsize(para_ids_shape, max_para_len)
            para_prefixIDlist, para_suffixIDlist, para_prefixmasklist, para_suffixmasklist = idlist_2_charShape(para_ids, para_mask, id2word, trichar2id, charLen)

            paras.append(para_ids)
            paras_shape.append(pad_para_shape)
            paras_mask.append(para_mask)
            para_trichar_ids.append(para_trichar_idlist)
            para_trichar_masks.append(para_trichar_mask)
            para_prefix_ids.append(para_prefixIDlist)
            para_prefix_mask.append(para_prefixmasklist)
            para_suffix_ids.append(para_suffixIDlist)
            para_suffix_mask.append(para_suffixmasklist)
            isInQ_paras.append(new_isInQ)
            distance_ls.append(new_distance_l)
            distance_rs.append(new_distance_r)

            #type
            type_example=strlist_2_wordidlist(type, word2id, id2word)
            type_example_shape = strlist_2_wordidlist(strlist2shapelist(type), word2id, id2word)
            type_trichar_idlist, type_trichar_mask = wordlist2tricharIDlist(type, trichar2id,2, max_trichar_len)
            types.append(type_example)
            types_shape.append(type_example_shape)
            type_trichar_ids.append(type_trichar_idlist)
            type_trichar_masks.append(type_trichar_mask)
            #reset variables
            example_co+=1
            paragraph=[]
            question=[]
            type=[]
            isInQ_para=[]
            distance_l=[]
            distance_r=[]
            if example_co %50000==0:
                print 'example_co:', example_co
            if example_no_limit is not None and example_co == example_no_limit:
                break
        if line_co%block_lines==3:#ground truth
#             if example_co >=487859:
#                 print 'line_co:', line_co, ':', line.strip()
            continue
        else:
            if line_co%block_lines==1:#question
#                 if example_co >=487859:
#                     print 'line_co:', line_co, ':', line.strip()
                question = line.strip().split()[4:]

            elif line_co%block_lines==2: # type words in question
#                 if example_co >=487859:
#                     print 'line_co:', line_co, ':', line.strip()
                type=line.strip().split()[1:]  #two words
            elif line_co%block_lines==4 or line_co%block_lines==8 or line_co%block_lines==12: # context words
#                 if example_co >=487859:
#                     print 'line_co:', line_co, ':', line.strip()
                if line_co%block_lines==4:
                    paragraph.append(line.strip().split()[2:])
                if line_co%block_lines==8:
                    paragraph.append(line.strip().split()[1:])
                if line_co%block_lines==12:
                    paragraph.append(line.strip().split()[1:-1])
            elif line_co%block_lines==5 or line_co%block_lines==9 or line_co%block_lines==13:
                if line_co%block_lines==5:
                    isInQ_para.append(map(int, line.strip().split()[2:]))
                if line_co%block_lines==9:
                    isInQ_para.append(map(int, line.strip().split()[1:]))
                if line_co%block_lines==13:
                    isInQ_para.append(map(int, line.strip().split()[1:-1]))
            elif line_co%block_lines==6 or line_co%block_lines==10 or line_co%block_lines==14:
                if line_co%block_lines==6:
                    distance_l.append(map(int, line.strip().split()[2:]))
                if line_co%block_lines==10:
                    distance_l.append(map(int, line.strip().split()[1:]))
                if line_co%block_lines==14:
                    distance_l.append(map(int, line.strip().split()[1:-1]))
            elif line_co%block_lines==7 or line_co%block_lines==11 or line_co%block_lines==15:
                if line_co%block_lines==7:
                    distance_r.append(map(int, line.strip().split()[2:]))
                if line_co%block_lines==11:
                    distance_r.append(map(int, line.strip().split()[1:]))
                if line_co%block_lines==15:
                    distance_r.append(map(int, line.strip().split()[1:-1]))
            elif line_co%block_lines ==0:#label
#                 if example_co >=487859:
#                     print 'line_co:', line_co, ':', line.strip()
                label_str=line.strip().split()[-1]

                label = 1 if label_str =='good' else 0

                labels.append(label)
                label2co[label]+=1

    readfile.close()
#     print 'load', example_co, 'samples finished'
    print 'load', example_co, 'samples finished, majority rate:'#, label2co.get(1)+label2co.get(0), label2co.get(1)*1.0/(label2co.get(1)+label2co.get(0)), label2co.get(0)*1.0/(label2co.get(1)+label2co.get(0))




    return     word2id, trichar2id, questions,questions_mask,paras,paras_mask,labels, isInQ_paras, distance_ls, distance_rs, paras_shape, questions_shape, types, types_shape,question_trichar_ids,question_trichar_masks,para_trichar_ids,para_trichar_masks,type_trichar_ids,type_trichar_masks,    question_prefix_ids,question_prefix_mask,question_suffix_ids,question_suffix_mask,para_prefix_ids,para_prefix_mask,para_suffix_ids,para_suffix_mask

def idlist_2_charShape(idlist, masklist, id2word, char2id, charLen):
    prefixIDlist=[]
    prefixmasklist=[]
    suffixIDlist=[]
    suffixmasklist=[]
    for pos, mask in enumerate(masklist):
        if mask == 0.0:
            prefixIDlist+=[0]*charLen
            prefixmasklist+=[0.0]*charLen
            suffixIDlist+=[0]*charLen
            suffixmasklist+=[0.0]*charLen
        else:
            wordstr=id2word.get(idlist[pos])
            word_chars=list(wordstr)
            wordLen=len(word_chars)
            padsize = charLen - wordLen
            if padsize <=0:
                prefix=word_chars[:charLen]
                suffix=word_chars[-charLen:]
                prefixmasklist+=[1.0]*charLen
                suffixmasklist+=[1.0]*charLen
            else: # pad empty space
                prefix=word_chars+[' ']*padsize
                prefixmasklist+=[1.0]*wordLen+[0.0]*padsize
                suffix=[' ']*padsize+word_chars
                suffixmasklist+=[0.0]*padsize+[1.0]*wordLen
            #fix to char ids
            prefixIDlist+=str_2_ids(prefix, char2id)
            suffixIDlist+=str_2_ids(suffix, char2id)
    return      prefixIDlist   , suffixIDlist, prefixmasklist, suffixmasklist

def str_2_ids(charlist, char2id):
    idlist=[]
    for char in charlist:
        id=char2id.get(char)
        if id is None: # if word was not in the vocabulary
            id=len(char2id)  # id of true words starts from 1, leaving 0 to "pad id"
            char2id[char]=id
        idlist.append(id)
    return idlist


def filt_data():
    line_co=0
    block_lines=16
    readfile=open('/mounts/Users/cisintern/hs/l/workhs/yin/20170328/dev.big.20170328.txt', 'r')



    found=False
    newtestfile=open('newtestfile20170531.txt', 'w')

    for line in readfile:
#         line=readfile.readline()

        if len(line.strip())==0:
            continue
        line_co+=1
#         if example_co >=487859:
#             print 'line_co:', line_co, ':', line.strip()
        if line_co % block_lines==1 and found is True:  #store existed data into matrix
            found=False
            print 'store over'
            newtestfile.close()
            exit(0)
        if line_co%block_lines==3:#ground truth
            if found:
                newtestfile.write(line.strip()+'\n')
            continue
        else:
            if line_co%block_lines==1:#question
#                 if example_co >=487859:
#                     print 'line_co:', line_co, ':', line.strip()
                question = line.strip().split()[4:]
                if ' '.join(question).find('What is the major US city that the is the university located') >=0:
                    found=True
                    newtestfile.write(line.strip()+'\n')



            elif line_co%block_lines==2: # type words in question
                if found:
                    newtestfile.write(line.strip()+'\n')
#
            elif line_co%block_lines==4 or line_co%block_lines==8 or line_co%block_lines==12: # context words
                if found:
                    newtestfile.write(line.strip()+'\n')
            elif line_co%block_lines==5 or line_co%block_lines==9 or line_co%block_lines==13:
                if found:
                    newtestfile.write(line.strip()+'\n')
            elif line_co%block_lines==6 or line_co%block_lines==10 or line_co%block_lines==14:
                if found:
                    newtestfile.write(line.strip()+'\n')
            elif line_co%block_lines==7 or line_co%block_lines==11 or line_co%block_lines==15:
                if found:
                    newtestfile.write(line.strip()+'\n')
            elif line_co%block_lines ==0:#label
                if found:
                    newtestfile.write(line.strip()+'\n')

    readfile.close()

def get_pred(wordlist, scorelist):
    max_id=numpy.argmax(scorelist)
    left_id = max_id
    right_id =  max_id
    while numpy.abs(scorelist[left_id]-scorelist[left_id-1]) < numpy.abs(scorelist[left_id-1]-scorelist[left_id-2])/5:
        left_id-=1
    while numpy.abs(scorelist[right_id]-scorelist[right_id+1]) < numpy.abs(scorelist[right_id+1]-scorelist[right_id+2])/5:
        right_id+=1

    return ' '.join(wordlist[left_id:right_id+1])



def construct_standard_answers():
    readfile=open('/mounts/Users/cisintern/hs/l/workhs/yin/20170602/input4wenpeng0.txt', 'r')
    writefile=open('/mounts/data/proj/wenpeng/Dataset/SQuAD/input4wenpeng0_standard20170608.txt', 'w')
    start=False
    question=''
    Goldan=''
    Predan=''
    grounds=[]
    preds=[]
    wordlist=[]
    scorelist=[]
    for line in readfile:
        if line.strip()=='NEW RECORD':
            start=True
            wordlist=[]
            scorelist=[]
            question=''
            Goldan=''
            Predan=''
            continue
        if len(line.strip())==0:
            start=False
            Predan = get_pred(wordlist, scorelist)
            #write
            grounds.append(Goldan)
            preds.append(Predan)
            writefile.write(Goldan+'\t'+';'+'\t'+Predan+'\t'+';'+'\t'+question+'\n')
            continue
        if start:
            tokenlist=line.strip().split()
            if len(tokenlist)==1:
                continue
            elif len(tokenlist)==2 and tokenlist[0]!='G':
#                 print line
                wordlist.append(tokenlist[0])
                scorelist.append(float(tokenlist[1]))
            else:
                if tokenlist[0]=='M':
                    question=' '.join(tokenlist[4:])
                elif tokenlist[0]=='G':
                    Goldan=' '.join(tokenlist[1:])

    print 'over'
    print evaluate_groundtruthlist_predlist(grounds,preds)
    readfile.close()
    writefile.close()


def transfer_wordlist_2_idlist_with_maxlen(token_list, vocab_map, maxlen):
    idlist=[]
    for word in token_list:
        id=vocab_map.get(word)
        if id is None: # if word was not in the vocabulary
            id=len(vocab_map)+1  # id of true words starts from 1, leaving 0 to "pad id"
            vocab_map[word]=id
        idlist.append(id)
    mask_list=[1.0]*len(idlist) # mask is used to indicate each word is a true word or a pad word
    pad_size=maxlen-len(idlist)
    if pad_size>0:
        idlist=[0]*pad_size+idlist
        mask_list=[0.0]*pad_size+mask_list
    else: # if actual sentence len is longer than the maxlen, truncate
        idlist=idlist[:maxlen]
        mask_list=mask_list[:maxlen]
    return idlist, mask_list

def transfer_wordlist_2_idlist_with_maxlen_return_wordlist(token_list, vocab_map, maxlen):
    pad_size = maxlen - len(token_list)
    if pad_size > 0:
        token_list=['uuuuuu']*pad_size+token_list
    else:
        token_list = token_list[:maxlen]
    idlist=[]
    mask_list=[]
    if pad_size > 0:
        idlist+=[0]*pad_size
        mask_list+=[0.0]*pad_size
        valid_token_list = token_list[pad_size:]
    else:
        valid_token_list = token_list
    for word in valid_token_list:
        id=vocab_map.get(word)
        if id is None: # if word was not in the vocabulary
            id=len(vocab_map)+1  # id of true words starts from 1, leaving 0 to "pad id"
            vocab_map[word]=id
        idlist.append(id)
        mask_list.append(1.0)
    return idlist, mask_list, token_list
def load_Qtype_dataset(maxlen=40):
    root="/mounts/Users/cisintern/hs/l/workhs/yin/20170623/"
    files=['trn20170623.txt', 'dev20170623.txt', 'big.dev20170623.txt']
    word2id={}  # store vocabulary, each word map to a id
    wordset=set()
    all_sentences_l=[]
    all_masks_l=[]
    all_sentences_r=[]
    all_masks_r=[]
    all_labels=[]
    extra_f = []
    max_sen_len=0
    for i in range(len(files)):
        print 'loading file:', root+files[i], '...'

        sents_l=[]
        sents_masks_l=[]
        sents_r=[]
        sents_masks_r=[]
        labels=[]
        f=[]
        readfile=open(root+files[i], 'r')
#         one_block_finished = False
        q=''
        a=''
        l=''
        positive_size = 0
        for line in readfile:
            if len(line.strip())==0:
#                 one_block_finished = True
                if q !='':
#                     print 'l:', l
                    if l=='good':
                        label=1
                        positive_size+=1
                    else:
                        label =0
#                     print label
#                     exit(0)
                    labels.append(label)

                    sent_idlist_l, sent_masklist_l=transfer_wordlist_2_idlist_with_maxlen(q, word2id, maxlen)
                    sent_idlist_r, sent_masklist_r=transfer_wordlist_2_idlist_with_maxlen(a, word2id, maxlen)
                    sents_l.append(sent_idlist_l)
                    sents_masks_l.append(sent_masklist_l)
                    sents_r.append(sent_idlist_r)
                    sents_masks_r.append(sent_masklist_r)

                    if len(set([x.lower() for x in q]) & set([x.lower() for x in a]))>0:
                        f.append(1.0)
                    else:
                        f.append(0.0)
                    q=''
                    a=''
                    l=''
            else:
#                 one_block_finished = False
                parts=line.strip().split() #lowercase all tokens, as we guess this is not important for sentiment task
                wordset |= set(parts)
                if parts[0]=='q':
                    q=parts[1:]
                if parts[0]=='a':
                    a=parts[1:]
                if parts[0]=='t':
                    l=parts[1]

        all_sentences_l.append(sents_l)
        all_sentences_r.append(sents_r)
        all_masks_l.append(sents_masks_l)
        all_masks_r.append(sents_masks_r)
        all_labels.append(labels)
        extra_f.append(f)
        print '\t\t\t size:', len(labels), 'pairs, positive rato:', positive_size*1.0/len(labels)
    print 'dataset loaded over, totally ', len(word2id), 'words'
    print 'len(wordset):', len(wordset)
    return all_sentences_l, all_masks_l, all_sentences_r, all_masks_r,all_labels, extra_f, word2id


def wordidlist2wordlist(idlist,mask,id2word):
    sent=[]
    for i, value in enumerate(mask):
        if value != 0.0:
            sent.append(id2word.get(idlist[i]))
    return ' '.join(sent)

def wordList_2_charIdList(word_list, word_size_limit, char_len, char2id):
    sent_len = len(word_list)
    pad_size = word_size_limit - sent_len
    if pad_size > 0:
        word_list = ['u'*char_len]*pad_size + word_list
    else:
        word_list = word_list[:word_size_limit]
    char_idlist=[]
    mask=[]
    for word in word_list:
        sub_char_idlist=[]
        word_len = len(word)
        for char in word:
            char_id = char2id.get(char)
            if char_id is None:
                char_id = len(char2id)+1
                char2id[char]=char_id
            sub_char_idlist.append(char_id)
        char_pad_size = char_len - len(sub_char_idlist)
        if char_pad_size > 0:
            sub_char_idlist = [0]*char_pad_size + sub_char_idlist
            sub_char_mask = [0.0]*char_pad_size + [1.0]*word_len
        else:
            sub_char_idlist=sub_char_idlist[:char_len]
            sub_char_mask = [1.0]*char_len
        char_idlist+=sub_char_idlist
        mask+=sub_char_mask
    return char_idlist, mask

def wordList_to_charIdList(word_list, char_len, char2id):
#     sent_len = len(word_list)
#     pad_size = word_size_limit - sent_len
#     if pad_size > 0:
#         word_list = ['u'*char_len]*pad_size + word_list
#     else:
#         word_list = word_list[:word_size_limit]
    char_idlist=[]
    mask=[]
    for word in word_list:
        sub_char_idlist=[]
        word_len = len(word)
        for char in word:
            char_id = char2id.get(char)
            if char_id is None:
                char_id = len(char2id)+1
                char2id[char]=char_id
            sub_char_idlist.append(char_id)
        char_pad_size = char_len - len(sub_char_idlist)
        if char_pad_size > 0:
            sub_char_idlist = [0]*char_pad_size + sub_char_idlist
            sub_char_mask = [0.0]*char_pad_size + [1.0]*word_len
        else:
            sub_char_idlist=sub_char_idlist[:char_len]
            sub_char_mask = [1.0]*char_len
        char_idlist+=sub_char_idlist
        mask+=sub_char_mask
    return char_idlist, mask

def wordlist_2_extralist(wordlist, refer_wordlist):
#     ner_para = ne_chunk(pos_tag(wordlist))
#     iob_tagged_para = tree2conlltags(ner_para)
    refer_str = ' '.join(refer_wordlist).lower()
    refer_len = len(refer_wordlist)
    wh_word_index = 0
#     for j, entry in enumerate(refer_wordlist):
#         if entry[:2]=='wh' or entry[:2]=='Wh':
#             wh_word_index = j
#             break
    qtype_vec = [0.0]*wh_word_size
    for qtype, idd in wh_word_dict.items():
        position = refer_str.find(qtype)
        if position>=0:
            qtype_vec[idd]=1.0
            wh_word_index = len(refer_str[:position].split())

    ref_vocab = set([x.lower() for x in refer_wordlist])
    extralist=[]
    for i, word in enumerate(wordlist):
        extra=[0.0]*5  #uppercase, digit, isInRefer, pos/q_len, distance_to_wh/q_len, year, month
        if word[0].isupper():
            extra[0]=1.0
        if bool(_digits.search(word)):
            extra[1]=1.0
#             if len(word)==4:
#                 extra[5]=1.0
        if word.lower() in ref_vocab:
            extra[2]=1.0
        
#         co_times=wordlist.count(word)
#         extra[3]=1.0/co_times
        word_index_in_q=0
        for j, entry in enumerate(refer_wordlist):
            if word.lower() == entry.lower():
                word_index_in_q = j
                break
        extra[3] = (word_index_in_q+1)*1.0/refer_len
        extra[4] = (refer_len - word_index_in_q + wh_word_index)*1.0/refer_len
#         if word.lower() in months:
#             extra[6]=1.0

#         print 'qtype_vec: ', qtype_vec
#         print      'refer_str: ', refer_str
#         exit(0)
        extra+=  qtype_vec  
#         if word in  string.punctuation:
#             extra[3]=1.0
#         postag = iob_tagged_para[i][1]
#         postag_index = postag_dict.get(postag)
#         if postag_index is None:
#             print postag, ' is not in postag_dict'
#             exit(0)
#         nertag = iob_tagged_para[i][2]
#         if nertag[0]=='B' or nertag[0]=='I':
#             nertag = nertag[2:]
#         ner_index = nertag_dict.get(nertag)
#         if ner_index is None:
#             print nertag, ' is not in nertag_dict'
#             exit(0)
#         pos_vec = [0.0]*postag_size
#         pos_vec[postag_index]=1.0
#         ner_vec = [0.0]*nertag_size
#         ner_vec[ner_index]=1.0
#         extra += pos_vec+ner_vec

        extralist.append(extra)
    return extralist

def load_squad_cnn_rank_span_train(word2id, char2id, p_len_limit, q_len_limit, char_len):
    readfile=open(path+'train-TwoStageRanking-SpanLevel.txt', 'r')

    questions=[]
    paragraphs=[]
    q_masks=[]
    p_masks=[]

    char_questions=[]
    char_paragraphs=[]
    char_q_masks=[]
    char_p_masks=[]

    labels=[]
    line_co=0
    for line in readfile:
        parts=line.strip().split('\t')
        question_wordlist=parts[0].split()
        para_wordlist=parts[1].split()
        label = int(parts[2])

        q_idlist, q_mask=transfer_wordlist_2_idlist_with_maxlen(question_wordlist, word2id, q_len_limit)
        p_idlist, p_mask=transfer_wordlist_2_idlist_with_maxlen(para_wordlist, word2id, p_len_limit)
        q_char_idlist, q_char_mask = wordList_2_charIdList(question_wordlist, q_len_limit, char_len, char2id)
        p_char_idlist, p_char_mask = wordList_2_charIdList(para_wordlist, p_len_limit, char_len, char2id)
        questions.append(q_idlist)
        paragraphs.append(p_idlist)
        q_masks.append(q_mask)
        p_masks.append(p_mask)

        char_questions.append(q_char_idlist)
        char_paragraphs.append(p_char_idlist)
        char_q_masks.append(q_char_mask)
        char_p_masks.append(p_char_mask)

        labels.append(label)
        line_co+=1
    print 'load train over, ', line_co, ' question-para pairs'
    return     questions,paragraphs,q_masks,p_masks,char_questions, char_paragraphs, char_q_masks,char_p_masks, labels, word2id, char2id

def load_squad_cnn_rank_word_train(word2id, char2id, p_len_limit, q_len_limit, char_len):
    readfile=open(path+'train-TwoStageRanking-SpanLevel.txt', 'r')

    questions=[]
    paragraphs=[]
    q_masks=[]
    p_masks=[]

    char_questions=[]
    char_paragraphs=[]
    char_q_masks=[]
    char_p_masks=[]

    labels=[]
    line_co=0
    for line in readfile:
        parts=line.strip().split('\t')
        question_wordlist=parts[0].split()
        para_wordlist=parts[1].split()
        start_label = int(parts[3])
        end_label = int(parts[4])

        q_idlist, q_mask=transfer_wordlist_2_idlist_with_maxlen(question_wordlist, word2id, q_len_limit)
        p_idlist, p_mask=transfer_wordlist_2_idlist_with_maxlen(para_wordlist, word2id, p_len_limit)
        q_char_idlist, q_char_mask = wordList_2_charIdList(question_wordlist, q_len_limit, char_len, char2id)
        p_char_idlist, p_char_mask = wordList_2_charIdList(para_wordlist, p_len_limit, char_len, char2id)
        questions.append(q_idlist)
        paragraphs.append(p_idlist)
        q_masks.append(q_mask)
        p_masks.append(p_mask)

        char_questions.append(q_char_idlist)
        char_paragraphs.append(p_char_idlist)
        char_q_masks.append(q_char_mask)
        char_p_masks.append(p_char_mask)

        labels.append([start_label, end_label])
        line_co+=1
    print 'load train over, ', line_co, ' question-para pairs'
    return     questions,paragraphs,q_masks,p_masks,char_questions, char_paragraphs, char_q_masks,char_p_masks, labels, word2id, char2id

def wordlist_2_wordOrShapeList(wordlist, vocab):
    newlist=[]
    for word in wordlist:
        if word not in vocab:
            shape = str2shape(word)
            newlist.append(shape)
        else:
            newlist.append(word)
    return newlist

def load_squad_cnn_rank_span_word_train(word2id, char2id, p_len_limit, q_len_limit, char_len):
    readfile=open(path+'train-TwoStageRanking-SpanLevel.txt', 'r')

    questions=[]
    paragraphs=[]
    q_masks=[]
    p_masks=[]

    char_questions=[]
    char_paragraphs=[]
    char_q_masks=[]
    char_p_masks=[]

    para_extras=[]

    span_labels=[]
    word_labels=[]
    line_co=0
    for line in readfile:
        parts=line.strip().split('\t')
        question_wordlist=parts[0].split()
        para_wordlist=parts[1].split()
        label = int(parts[2])
        start_label = int(parts[3])
        end_label = int(parts[4])
        if question_wordlist[-1]=='?':
            question_wordlist=question_wordlist[:-1]



#         q_idlist, q_mask=transfer_wordlist_2_idlist_with_maxlen(question_wordlist, word2id, q_len_limit)
#         p_idlist, p_mask=transfer_wordlist_2_idlist_with_maxlen(para_wordlist, word2id, p_len_limit)
        #transfer_wordlist_2_idlist_with_maxlen_return_wordlist
        q_idlist, q_mask, trunc_q=transfer_wordlist_2_idlist_with_maxlen_return_wordlist(question_wordlist, word2id, q_len_limit)
        p_idlist, p_mask, trunc_p=transfer_wordlist_2_idlist_with_maxlen_return_wordlist(para_wordlist, word2id, p_len_limit)
        q_char_idlist, q_char_mask = wordList_to_charIdList(trunc_q, char_len, char2id)
        p_char_idlist, p_char_mask = wordList_to_charIdList(trunc_p, char_len, char2id)

        p_extra =  wordlist_2_extralist(trunc_p, question_wordlist)
#         print 'p_extra:', p_extra
#         exit(0)

        questions.append(q_idlist)
        paragraphs.append(p_idlist)
        q_masks.append(q_mask)
        p_masks.append(p_mask)

        char_questions.append(q_char_idlist)
        char_paragraphs.append(p_char_idlist)
        char_q_masks.append(q_char_mask)
        char_p_masks.append(p_char_mask)

        para_extras.append(p_extra)
        span_labels.append(label)
        word_labels.append([start_label, end_label])
        line_co+=1
        if line_co%10000==0:
            print line_co, '...'
    print 'load train over, ', line_co, ' question-para pairs'
    return     questions,paragraphs,q_masks,p_masks,char_questions, char_paragraphs, char_q_masks,char_p_masks, span_labels, word_labels, para_extras,word2id, char2id

def load_squad_cnn_rank_span_dev(word2id, char2id, p_len_limit, q_len_limit, char_len):
    readfile=open(path+'dev-TwoStageRanking-SpanLevel.txt', 'r')

    questions=[]
    paragraphs=[]
    para_wordlists=[]
    q_masks=[]
    p_masks=[]

    char_questions=[]
    char_paragraphs=[]
    char_q_masks=[]
    char_p_masks=[]

    q_ids=[] #used to store question ids
    line_co=0
    for line in readfile:
        parts=line.strip().split('\t')
        q_id = parts[0]
        question_wordlist=parts[1].split()
        para_wordlist=parts[2].split()

        q_idlist, q_mask=transfer_wordlist_2_idlist_with_maxlen(question_wordlist, word2id, q_len_limit)
        p_idlist, p_mask=transfer_wordlist_2_idlist_with_maxlen(para_wordlist, word2id, p_len_limit)

        q_char_idlist, q_char_mask = wordList_2_charIdList(question_wordlist, q_len_limit, char_len, char2id)
        p_char_idlist, p_char_mask = wordList_2_charIdList(para_wordlist, p_len_limit, char_len, char2id)

        if p_len_limit > len(para_wordlist):
            para_wordlists.append(['UNK']*(p_len_limit - len(para_wordlist))+para_wordlist)
        else:
            para_wordlists.append(para_wordlist[:p_len_limit])

        questions.append(q_idlist)
        paragraphs.append(p_idlist)
        q_masks.append(q_mask)
        p_masks.append(p_mask)

        char_questions.append(q_char_idlist)
        char_paragraphs.append(p_char_idlist)
        char_q_masks.append(q_char_mask)
        char_p_masks.append(p_char_mask)

        q_ids.append(q_id)
        line_co+=1
    print 'load dev over, ', line_co, ' question-sent pairs'
    return     questions,paragraphs,q_masks,p_masks,char_questions, char_paragraphs, char_q_masks,char_p_masks, q_ids, word2id, char2id, para_wordlists


def load_squad_cnn_rank_word_dev(word2id, char2id, p_len_limit, q_len_limit, char_len):
    readfile=open(path+'dev-TwoStageRanking-SpanLevel.txt', 'r')

    questions=[]
    paragraphs=[]
    para_wordlists=[]
    q_masks=[]
    p_masks=[]

    char_questions=[]
    char_paragraphs=[]
    char_q_masks=[]
    char_p_masks=[]

    labels=[] #used to store starts
    q_ids=[] #used to store question ids
    line_co=0

    ground_truth_ans_outside=0
    max_para_len =0
    for line in readfile:
        parts=line.strip().split('\t')
        q_id = parts[0]
        question_wordlist=parts[1].split()
        para_wordlist=parts[2].split()
        if len(para_wordlist)> max_para_len:
            max_para_len = len(para_wordlist)
        corr_sent_label = parts[3].strip()


        q_idlist, q_mask=transfer_wordlist_2_idlist_with_maxlen(question_wordlist, word2id, q_len_limit)
        p_idlist, p_mask=transfer_wordlist_2_idlist_with_maxlen(para_wordlist, word2id, p_len_limit)

        q_char_idlist, q_char_mask = wordList_2_charIdList(question_wordlist, q_len_limit, char_len, char2id)
        p_char_idlist, p_char_mask = wordList_2_charIdList(para_wordlist, p_len_limit, char_len, char2id)

        if p_len_limit > len(para_wordlist):
            truncate_para_wordlist = ['UNK']*(p_len_limit - len(para_wordlist))+para_wordlist
            # para_wordlists.append(['UNK']*(p_len_limit - len(para_wordlist))+para_wordlist)
        else:
            half_index = len(para_wordlist)/2
            truncate_para_wordlist = para_wordlist[half_index-p_len_limit/2:half_index-p_len_limit/2+p_len_limit]
        para_wordlists.append(truncate_para_wordlist)
        startlist=set()
        if corr_sent_label == '1':
            truncate_q_str = ' '.join(truncate_para_wordlist)
            ans_list = parts[4].split(' || ')
            for ans in ans_list:
                ans_id = truncate_q_str.find(ans)
                if ans_id < 0:
                    # ground_truth_ans_outside+=1
                    startlist.add(-1)
                else:
                    startlist.add(len(truncate_q_str[:ans_id].split()))
            if len(startlist)==1 and list(startlist)[0]==-1:
                ground_truth_ans_outside+=1
        else:
            startlist.add(-1)


        questions.append(q_idlist)
        paragraphs.append(p_idlist)
        q_masks.append(q_mask)
        p_masks.append(p_mask)

        char_questions.append(q_char_idlist)
        char_paragraphs.append(p_char_idlist)
        char_q_masks.append(q_char_mask)
        char_p_masks.append(p_char_mask)
        q_ids.append(q_id)
        labels.append(list(startlist))
        line_co+=1
    print 'load dev over, ', line_co, ' question-sent pairs, ground_truth_ans_outside: ', ground_truth_ans_outside, ' max_para_len:', max_para_len
    return     questions,paragraphs,q_masks,p_masks,char_questions, char_paragraphs, char_q_masks,char_p_masks, labels, q_ids, word2id, char2id, para_wordlists

def load_squad_cnn_rank_span_word_dev(word2id, char2id, p_len_limit, q_len_limit, char_len):
    #current it is the same with load_squad_cnn_rank_span_dev
    readfile=open(path+'dev-TwoStageRanking-SpanLevel.txt', 'r')

    questions=[]
    paragraphs=[]
    para_wordlists=[]
    q_masks=[]
    p_masks=[]

    char_questions=[]
    char_paragraphs=[]
    char_q_masks=[]
    char_p_masks=[]

    para_extras=[]

    q_ids=[] #used to store question ids
    line_co=0
    for line in readfile:
        parts=line.strip().split('\t')
        q_id = parts[0]
        question_wordlist=parts[1].split()
        para_wordlist=parts[2].split()
        if question_wordlist[-1]=='?':
            question_wordlist=question_wordlist[:-1]


#         q_idlist, q_mask=transfer_wordlist_2_idlist_with_maxlen(question_wordlist, word2id, q_len_limit)
#         p_idlist, p_mask=transfer_wordlist_2_idlist_with_maxlen(para_wordlist, word2id, p_len_limit)
#
#         q_char_idlist, q_char_mask = wordList_2_charIdList(question_wordlist, q_len_limit, char_len, char2id)
#         p_char_idlist, p_char_mask = wordList_2_charIdList(para_wordlist, p_len_limit, char_len, char2id)

        q_idlist, q_mask, trunc_q=transfer_wordlist_2_idlist_with_maxlen_return_wordlist(question_wordlist, word2id, q_len_limit)
        p_idlist, p_mask, trunc_p=transfer_wordlist_2_idlist_with_maxlen_return_wordlist(para_wordlist, word2id, p_len_limit)
        q_char_idlist, q_char_mask = wordList_to_charIdList(trunc_q, char_len, char2id)
        p_char_idlist, p_char_mask = wordList_to_charIdList(trunc_p, char_len, char2id)

        p_extra =  wordlist_2_extralist(trunc_p, question_wordlist)

#         if p_len_limit > len(para_wordlist):
#             para_wordlists.append(['UNK']*(p_len_limit - len(para_wordlist))+para_wordlist)
#         else:
#             para_wordlists.append(para_wordlist[:p_len_limit])
        para_wordlists.append(trunc_p)

        questions.append(q_idlist)
        paragraphs.append(p_idlist)
        q_masks.append(q_mask)
        p_masks.append(p_mask)

        char_questions.append(q_char_idlist)
        char_paragraphs.append(p_char_idlist)
        char_q_masks.append(q_char_mask)
        char_p_masks.append(p_char_mask)

        para_extras.append(p_extra)

        q_ids.append(q_id)
        line_co+=1
        if line_co%3000==0:
            print line_co, '...'
    print 'load dev over, ', line_co, ' question-sent pairs'
    return     questions,paragraphs,q_masks,p_masks,char_questions, char_paragraphs, char_q_masks,char_p_masks, q_ids, para_extras, word2id, char2id, para_wordlists

if __name__ == '__main__':
#     store_SQUAD_train()
#     store_SQUAD_dev()
#     filt_data()
    construct_standard_answers()
