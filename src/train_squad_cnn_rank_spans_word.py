import cPickle
import gzip
import os
import sys
sys.setrecursionlimit(6000)
import time

import numpy
import theano
import theano.tensor as T
import theano.sandbox.neighbours as TSN
import time
import random

from logistic_sgd import LogisticRegression
from cis.deep.utils.theano import debug_print
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from load_SQUAD import load_squad_cnn_rank_span_word_train, load_glove, decode_predict_id, load_squad_cnn_rank_span_word_dev, extract_ansList_attentionList, extract_ansList_attentionList_maxlen5, MacroF1, load_word2vec, load_word2vec_to_init
from word2embeddings.nn.util import zero_value, random_value_normal
from common_functions import squad_cnn_rank_spans_word,add_HLs_2_tensor3, load_model_from_file, store_model_to_file, create_LSTM_para, Bd_LSTM_Batch_Tensor_Input_with_Mask, Bd_GRU_Batch_Tensor_Input_with_Mask, create_ensemble_para, create_GRU_para, normalize_matrix, create_conv_para, Matrix_Bit_Shift, Conv_with_input_para, L2norm_paraList
from random import shuffle
from utils_pg import *
from evaluate import standard_eval
import codecs
import json




#need to try
'''
above 35 percent epoch 3
'''

def evaluate_lenet5(learning_rate=0.01, n_epochs=3, batch_size=100, emb_size=300, char_emb_size=20, hidden_size=300,
                    L2_weight=0.0001, p_len_limit=400, test_p_len_limit=100, q_len_limit=20, char_len=15, filter_size = [5,5],
                    char_filter_size=5, margin=0.85):
    test_batch_size=batch_size*10
    model_options = locals().copy()
    print "model options", model_options
    rootPath='/mounts/data/proj/wenpeng/Dataset/SQuAD/';
    rng = numpy.random.RandomState(23455)


    word2id={}
    char2id={}
    #questions,paragraphs,q_masks,p_masks,labels, word2id
    train_Q_list,train_para_list, train_Q_mask, train_para_mask, train_Q_char_list,train_para_char_list, train_Q_char_mask, train_para_char_mask, train_span_label_list, train_word_label_list, word2id, char2id=load_squad_cnn_rank_span_word_train(word2id, char2id, p_len_limit, q_len_limit, char_len)
    train_size=len(train_para_list)

    test_Q_list, test_para_list,  test_Q_mask, test_para_mask,test_Q_char_list, test_para_char_list,  test_Q_char_mask, test_para_char_mask, q_idlist, word2id, char2id, test_para_wordlist_list= load_squad_cnn_rank_span_word_dev(word2id, char2id, test_p_len_limit, q_len_limit, char_len)
    test_size=len(test_para_list)

    train_Q_list = numpy.asarray(train_Q_list, dtype='int32')
    train_para_list = numpy.asarray(train_para_list, dtype='int32')
    train_Q_mask = numpy.asarray(train_Q_mask, dtype=theano.config.floatX)
    train_para_mask = numpy.asarray(train_para_mask, dtype=theano.config.floatX)

    train_Q_char_list = numpy.asarray(train_Q_char_list, dtype='int32')
    train_para_char_list = numpy.asarray(train_para_char_list, dtype='int32')
    train_Q_char_mask = numpy.asarray(train_Q_char_mask, dtype=theano.config.floatX)
    train_para_char_mask = numpy.asarray(train_para_char_mask, dtype=theano.config.floatX)

    train_span_label_list = numpy.asarray(train_span_label_list, dtype='int32')
    train_word_label_list = numpy.asarray(train_word_label_list, dtype='int32')

    test_Q_list = numpy.asarray(test_Q_list, dtype='int32')
    test_para_list = numpy.asarray(test_para_list, dtype='int32')
    test_Q_mask = numpy.asarray(test_Q_mask, dtype=theano.config.floatX)
    test_para_mask = numpy.asarray(test_para_mask, dtype=theano.config.floatX)

    test_Q_char_list = numpy.asarray(test_Q_char_list, dtype='int32')
    test_para_char_list = numpy.asarray(test_para_char_list, dtype='int32')
    test_Q_char_mask = numpy.asarray(test_Q_char_mask, dtype=theano.config.floatX)
    test_para_char_mask = numpy.asarray(test_para_char_mask, dtype=theano.config.floatX)



    vocab_size = len(word2id)
    print 'vocab size: ', vocab_size
    rand_values=random_value_normal((vocab_size+1, emb_size), theano.config.floatX, rng)
    rand_values[0]=numpy.array(numpy.zeros(emb_size),dtype=theano.config.floatX)
    id2word = {y:x for x,y in word2id.iteritems()}
    word2vec=load_glove()
    rand_values=load_word2vec_to_init(rand_values, id2word, word2vec)
    embeddings=theano.shared(value=rand_values, borrow=True)

    char_size = len(char2id)
    print 'char size: ', char_size
    char_rand_values=random_value_normal((char_size+1, char_emb_size), theano.config.floatX, rng)
    char_embeddings=theano.shared(value=char_rand_values, borrow=True)


    # allocate symbolic variables for the data
#     index = T.lscalar()
    paragraph = T.imatrix('paragraph')
    questions = T.imatrix('questions')
    span_indices= T.ivector() #batch
    word_indices = T.imatrix() #(batch, 2)
    ans_indices = T.ivector() # for one batch, the length is dynamic
    para_mask=T.fmatrix('para_mask')
    q_mask=T.fmatrix('q_mask')

    char_paragraph = T.imatrix() #(batch, char_len*p_len)
    char_questions = T.imatrix()
    char_para_mask=T.fmatrix()
    char_q_mask=T.fmatrix()

    true_p_len = T.iscalar()



    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    true_batch_size = paragraph.shape[0]

    common_input_p=embeddings[paragraph.flatten()].reshape((true_batch_size,true_p_len, emb_size)) #the input format can be adapted into CNN or GRU or LSTM
    common_input_q=embeddings[questions.flatten()].reshape((true_batch_size,q_len_limit, emb_size))


    char_common_input_p=char_embeddings[char_paragraph.flatten()].reshape((true_batch_size*true_p_len, char_len, char_emb_size)) #the input format can be adapted into CNN or GRU or LSTM
    char_common_input_q=char_embeddings[char_questions.flatten()].reshape((true_batch_size*q_len_limit, char_len, char_emb_size))

    char_p_masks = char_para_mask.reshape((true_batch_size*true_p_len, char_len))
    char_q_masks = char_q_mask.reshape((true_batch_size*q_len_limit, char_len))

    conv_W_char, conv_b_char=create_conv_para(rng, filter_shape=(char_emb_size, 1, char_emb_size, char_filter_size))
    conv_W_1, conv_b_1=create_conv_para(rng, filter_shape=(hidden_size, 1, emb_size+char_emb_size, filter_size[0]))
    conv_W_2, conv_b_2=create_conv_para(rng, filter_shape=(hidden_size, 1, hidden_size, filter_size[1]))

    conv_W_1_q, conv_b_1_q=create_conv_para(rng, filter_shape=(hidden_size, 1, emb_size+char_emb_size, filter_size[0]))
    conv_W_2_q, conv_b_2_q=create_conv_para(rng, filter_shape=(hidden_size, 1, hidden_size, filter_size[1]))
    NN_para=[conv_W_1, conv_b_1,conv_W_2, conv_b_2,conv_W_1_q, conv_b_1_q, conv_W_2_q, conv_b_2_q, conv_W_char, conv_b_char]

    span_input4score, word_input4score = squad_cnn_rank_spans_word(rng, common_input_p, common_input_q, char_common_input_p, char_common_input_q,batch_size, p_len_limit,q_len_limit,
                         emb_size, char_emb_size,char_len,filter_size,char_filter_size,hidden_size,
                         conv_W_1, conv_b_1,conv_W_2, conv_b_2,conv_W_1_q, conv_b_1_q, conv_W_2_q, conv_b_2_q,conv_W_char,conv_b_char,
                         para_mask, q_mask, char_p_masks,char_q_masks)

    test_span_input4score, test_word_input4score = squad_cnn_rank_spans_word(rng, common_input_p, common_input_q, char_common_input_p, char_common_input_q,test_batch_size, test_p_len_limit,q_len_limit,
                         emb_size, char_emb_size,char_len,filter_size,char_filter_size,hidden_size,
                         conv_W_1, conv_b_1,conv_W_2, conv_b_2, conv_W_1_q, conv_b_1_q, conv_W_2_q, conv_b_2_q,conv_W_char,conv_b_char,
                         para_mask, q_mask, char_p_masks,char_q_masks)  #(batch, hidden, gram_size)

    gram_size = 5*true_p_len-(0+1+2+3+4)

#     U_a = create_ensemble_para(rng, 1, 4*hidden_size)
#     norm_U_a=normalize_matrix(U_a)
#     span_scores_matrix=T.dot(span_input4score.dimshuffle(0,2,1), norm_U_a).reshape((batch_size, gram_size))  #(batch, 13*para_len-78, 1)

    span_HL_1_para = create_ensemble_para(rng, hidden_size, 4*hidden_size)
    span_HL_2_para = create_ensemble_para(rng, hidden_size, hidden_size)
    span_HL_3_para = create_ensemble_para(rng, hidden_size, hidden_size)
    span_HL_4_para = create_ensemble_para(rng, hidden_size, hidden_size)
    span_U_a = create_ensemble_para(rng, 1, hidden_size)
    norm_span_U_a=normalize_matrix(span_U_a)
    norm_span_HL_1_para=normalize_matrix(span_HL_1_para)
    norm_span_HL_2_para=normalize_matrix(span_HL_2_para)
    norm_span_HL_3_para=normalize_matrix(span_HL_3_para)
    norm_span_HL_4_para=normalize_matrix(span_HL_4_para)

    span_scores_matrix = add_HLs_2_tensor3(span_input4score, norm_span_HL_1_para,norm_span_HL_2_para,norm_span_HL_3_para,norm_span_HL_4_para, norm_span_U_a, batch_size,gram_size)

    span_scores=T.nnet.softmax(span_scores_matrix) #(batch, 7*para_len-21)
    loss_neg_likelihood=-T.mean(T.log(span_scores[T.arange(batch_size), span_indices]))

    #ranking loss
    tanh_span_scores_matrix = span_scores#T.tanh(span_scores_matrix) #(batch, gram_size)

    index_matrix = T.zeros((batch_size, gram_size), dtype=theano.config.floatX)
    new_index_matrix = T.set_subtensor(index_matrix[T.arange(batch_size), span_indices], 1.0)


    prob_batch_posi = tanh_span_scores_matrix[new_index_matrix.nonzero()]
    prob_batch_nega = tanh_span_scores_matrix[(1.0-new_index_matrix).nonzero()]

    repeat_posi = T.extra_ops.repeat(prob_batch_posi, prob_batch_nega.shape[0], axis=0)
    repeat_nega = T.extra_ops.repeat(prob_batch_nega.dimshuffle('x',0), prob_batch_posi.shape[0], axis=0).flatten()
    loss_rank = T.mean(T.maximum(0.0, margin-repeat_posi+repeat_nega))

    span_loss = loss_neg_likelihood + loss_rank

#     test_span_scores_matrix=T.dot(test_span_input4score.dimshuffle(0,2,1), norm_U_a).reshape((true_batch_size, gram_size))  #(batch, 13*para_len-78)
    test_span_scores_matrix = add_HLs_2_tensor3(test_span_input4score, norm_span_HL_1_para,norm_span_HL_2_para,norm_span_HL_3_para,norm_span_HL_4_para, norm_span_U_a, true_batch_size,gram_size)



    #word
    HL_1_para = create_ensemble_para(rng, hidden_size, 2*hidden_size)
    HL_2_para = create_ensemble_para(rng, hidden_size, hidden_size)
    HL_3_para = create_ensemble_para(rng, hidden_size, hidden_size)
    HL_4_para = create_ensemble_para(rng, hidden_size, hidden_size)
    start_U_a = create_ensemble_para(rng, 1, hidden_size)
    norm_start_U_a=normalize_matrix(start_U_a)
    norm_HL_1_para=normalize_matrix(HL_1_para)
    norm_HL_2_para=normalize_matrix(HL_2_para)
    norm_HL_3_para=normalize_matrix(HL_3_para)
    norm_HL_4_para=normalize_matrix(HL_4_para)

    end_HL_1_para = create_ensemble_para(rng, hidden_size, 2*hidden_size)
    end_HL_2_para = create_ensemble_para(rng, hidden_size, hidden_size)
    end_HL_3_para = create_ensemble_para(rng, hidden_size, hidden_size)
    end_HL_4_para = create_ensemble_para(rng, hidden_size, hidden_size)
    end_U_a = create_ensemble_para(rng, 1, hidden_size)
    end_norm_U_a=normalize_matrix(end_U_a)
    end_norm_HL_1_para=normalize_matrix(end_HL_1_para)
    end_norm_HL_2_para=normalize_matrix(end_HL_2_para)
    end_norm_HL_3_para=normalize_matrix(end_HL_3_para)
    end_norm_HL_4_para=normalize_matrix(end_HL_4_para)

    start_scores_matrix = add_HLs_2_tensor3(word_input4score, norm_HL_1_para,norm_HL_2_para,norm_HL_3_para,norm_HL_4_para, norm_start_U_a, batch_size,true_p_len)
    start_scores=T.nnet.softmax(start_scores_matrix) #(batch, para_len)
    end_scores_matrix = add_HLs_2_tensor3(word_input4score, end_norm_HL_1_para,end_norm_HL_2_para,end_norm_HL_3_para,end_norm_HL_4_para, end_norm_U_a, batch_size,true_p_len)
    end_scores=T.nnet.softmax(end_scores_matrix) #(batch, para_len)
    start_loss_neg_likelihood=-T.mean(T.log(start_scores[T.arange(batch_size), word_indices[:,0]]))
    end_loss_neg_likelihood=-T.mean(T.log(end_scores[T.arange(batch_size), word_indices[:,1]]))

    #ranking loss start
    tanh_start_scores_matrix = start_scores#T.tanh(span_scores_matrix) #(batch, gram_size)
    start_index_matrix = T.zeros((batch_size, p_len_limit), dtype=theano.config.floatX)
    start_new_index_matrix = T.set_subtensor(start_index_matrix[T.arange(batch_size), word_indices[:,0]], 1.0)
    start_prob_batch_posi = tanh_start_scores_matrix[start_new_index_matrix.nonzero()]
    start_prob_batch_nega = tanh_start_scores_matrix[(1.0-start_new_index_matrix).nonzero()]
    start_repeat_posi = T.extra_ops.repeat(start_prob_batch_posi, start_prob_batch_nega.shape[0], axis=0)
    start_repeat_nega = T.extra_ops.repeat(start_prob_batch_nega.dimshuffle('x',0), start_prob_batch_posi.shape[0], axis=0).flatten()
    start_loss_rank = T.mean(T.maximum(0.0, margin-start_repeat_posi+start_repeat_nega))

    #ranking loss END
    end_tanh_scores_matrix = end_scores#T.tanh(span_scores_matrix) #(batch, gram_size)
    end_index_matrix = T.zeros((batch_size, p_len_limit), dtype=theano.config.floatX)
    end_new_index_matrix = T.set_subtensor(end_index_matrix[T.arange(batch_size), word_indices[:,1]], 1.0)
    end_prob_batch_posi = end_tanh_scores_matrix[end_new_index_matrix.nonzero()]
    end_prob_batch_nega = end_tanh_scores_matrix[(1.0-end_new_index_matrix).nonzero()]
    end_repeat_posi = T.extra_ops.repeat(end_prob_batch_posi, end_prob_batch_nega.shape[0], axis=0)
    end_repeat_nega = T.extra_ops.repeat(end_prob_batch_nega.dimshuffle('x',0), end_prob_batch_posi.shape[0], axis=0).flatten()
    end_loss_rank = T.mean(T.maximum(0.0, margin-end_repeat_posi+end_repeat_nega))

    word_loss = start_loss_neg_likelihood +end_loss_neg_likelihood+start_loss_rank+end_loss_rank

    #test
    test_start_scores_matrix = add_HLs_2_tensor3(test_word_input4score, norm_HL_1_para,norm_HL_2_para,norm_HL_3_para,norm_HL_4_para,norm_start_U_a, true_batch_size,true_p_len) #(batch, test_p_len)
    mask_test_start_return=test_start_scores_matrix*para_mask #(batch, p_len)

    end_test_scores_matrix = add_HLs_2_tensor3(test_word_input4score, end_norm_HL_1_para,end_norm_HL_2_para,end_norm_HL_3_para,end_norm_HL_4_para,end_norm_U_a, true_batch_size,true_p_len) #(batch, test_p_len)
    end_mask_test_return=end_test_scores_matrix*para_mask  #(batch, p_len)

    word_gram_1 = mask_test_start_return+end_mask_test_return
    word_gram_2 = mask_test_start_return[:,:-1]+end_mask_test_return[:,1:] #(batch* hidden_size, maxsenlen-1)
    word_gram_3 = mask_test_start_return[:,:-2]+end_mask_test_return[:,2:] #(batch* hidden_size, maxsenlen-2)
    word_gram_4 = mask_test_start_return[:,:-3]+end_mask_test_return[:,3:] #(batch* hidden_size, maxsenlen-3)
    word_gram_5 = mask_test_start_return[:,:-4]+end_mask_test_return[:,4:] #(batch* hidden_size, maxsenlen-4)
    word_pair_scores=T.concatenate([word_gram_1, word_gram_2,word_gram_3,word_gram_4,word_gram_5], axis=1)#(batch_size, gram_size)


    #ans words train
    ans_HL_1_para = create_ensemble_para(rng, hidden_size, 2*hidden_size)
    ans_HL_2_para = create_ensemble_para(rng, hidden_size, hidden_size)
    ans_HL_3_para = create_ensemble_para(rng, hidden_size, hidden_size)
    ans_HL_4_para = create_ensemble_para(rng, hidden_size, hidden_size)
    ans_U_a = create_ensemble_para(rng, 1, hidden_size)
    norm_ans_U_a=normalize_matrix(ans_U_a)
    norm_ans_HL_1_para=normalize_matrix(ans_HL_1_para)
    norm_ans_HL_2_para=normalize_matrix(ans_HL_2_para)
    norm_ans_HL_3_para=normalize_matrix(ans_HL_3_para)
    norm_ans_HL_4_para=normalize_matrix(ans_HL_4_para)

    ans_scores_matrix = add_HLs_2_tensor3(word_input4score, norm_ans_HL_1_para,norm_ans_HL_2_para,norm_ans_HL_3_para,norm_ans_HL_4_para, norm_ans_U_a, batch_size,true_p_len)
    ans_scores_vec=T.nnet.softmax(ans_scores_matrix).flatten() #(batch, para_len)
    ans_loss_neg_likelihood=-T.mean(T.log(ans_scores_vec[ans_indices]))

    ans_index_vec = T.zeros((batch_size, p_len_limit), dtype=theano.config.floatX).flatten()
    ans_new_index = T.set_subtensor(ans_index_vec[ans_indices], 1.0)
    ans_prob_batch_posi = ans_scores_vec[ans_new_index.nonzero()]
    ans_prob_batch_nega = ans_scores_vec[(1.0-ans_new_index).nonzero()]
    ans_repeat_posi = T.extra_ops.repeat(ans_prob_batch_posi, ans_prob_batch_nega.shape[0], axis=0)
    ans_repeat_nega = T.extra_ops.repeat(ans_prob_batch_nega.dimshuffle('x',0), ans_prob_batch_posi.shape[0], axis=0).flatten()
    ans_loss_rank = T.mean(T.maximum(0.0, margin-ans_repeat_posi+ans_repeat_nega))

    ans_loss = ans_loss_neg_likelihood+ans_loss_rank
    #ans words test
    test_ans_scores_matrix = add_HLs_2_tensor3(test_word_input4score, norm_ans_HL_1_para,norm_ans_HL_2_para,norm_ans_HL_3_para,norm_ans_HL_4_para, norm_ans_U_a, true_batch_size,true_p_len)
    test_ans_scores_matrix=test_ans_scores_matrix*para_mask#T.nnet.softmax(test_ans_scores_matrix) #(batch, para_len)
    ans_gram_1 = test_ans_scores_matrix
    ans_gram_2 = (test_ans_scores_matrix[:,:-1]+test_ans_scores_matrix[:,1:])/2.0 #(batch* hidden_size, maxsenlen-1)
    ans_gram_3 = (test_ans_scores_matrix[:,:-2]+test_ans_scores_matrix[:,1:-1]+test_ans_scores_matrix[:,2:])/3.0 #(batch* hidden_size, maxsenlen-2)
    ans_gram_4 = (test_ans_scores_matrix[:,:-3]+test_ans_scores_matrix[:,1:-2]+test_ans_scores_matrix[:,2:-1]+test_ans_scores_matrix[:,3:])/4.0 #(batch* hidden_size, maxsenlen-3)
    ans_gram_5 = (test_ans_scores_matrix[:,:-4]+test_ans_scores_matrix[:,1:-3]+test_ans_scores_matrix[:,2:-2]+test_ans_scores_matrix[:,3:-1]+test_ans_scores_matrix[:,4:])/5.0 #(batch* hidden_size, maxsenlen-4)
    ans_word_scores=T.concatenate([ans_gram_1, ans_gram_2,ans_gram_3,ans_gram_4,ans_gram_5], axis=1)#(batch, hidden_size, maxsenlen-(0+1+2+3+4))


    test_span_word_scores_matrix=test_span_scores_matrix+word_pair_scores+ans_word_scores
    test_return=T.argmax(test_span_word_scores_matrix, axis=1) #batch

#     params = [embeddings,char_embeddings]+NN_para+[U_a]
    params = [embeddings,char_embeddings]+NN_para+[span_U_a,span_HL_1_para,span_HL_2_para,span_HL_3_para,span_HL_4_para]+[start_U_a, HL_1_para,HL_2_para,HL_3_para,HL_4_para]+[end_U_a,end_HL_1_para,end_HL_2_para,end_HL_3_para,end_HL_4_para]+[ans_U_a,ans_HL_1_para,ans_HL_2_para,ans_HL_3_para,ans_HL_4_para]

    L2_reg =L2norm_paraList([embeddings,char_embeddings,
    conv_W_1,conv_W_2,conv_W_1_q, conv_W_2_q, conv_W_char,
    span_U_a,span_HL_1_para,span_HL_2_para,span_HL_3_para,span_HL_4_para,
    start_U_a, HL_1_para,HL_2_para,HL_3_para,HL_4_para,
    end_U_a,end_HL_1_para,end_HL_2_para,end_HL_3_para,end_HL_4_para,
    ans_U_a,ans_HL_1_para,ans_HL_2_para,ans_HL_3_para,ans_HL_4_para
    ])
    #L2_reg = L2norm_paraList(params)
    cost=span_loss+word_loss+ans_loss+L2_weight*L2_reg


    accumulator=[]
    for para_i in params:
        eps_p=numpy.zeros_like(para_i.get_value(borrow=True),dtype=theano.config.floatX)
        accumulator.append(theano.shared(eps_p, borrow=True))

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    updates = []
    for param_i, grad_i, acc_i in zip(params, grads, accumulator):
#         print grad_i.type
        acc = acc_i + T.sqr(grad_i)
        updates.append((param_i, param_i - learning_rate * grad_i / (T.sqrt(acc)+1e-8)))   #AdaGrad
        updates.append((acc_i, acc))

#     updates=Adam(cost, params, lr=0.0001)

    train_model = theano.function([paragraph, questions,span_indices, word_indices,ans_indices, para_mask, q_mask,    char_paragraph, #(batch, char_len*p_len)
        char_questions, char_para_mask, char_q_mask, true_p_len], cost, updates=updates,on_unused_input='ignore')

    test_model = theano.function([paragraph, questions,para_mask, q_mask,
        char_paragraph,
        char_questions,
        char_para_mask,
        char_q_mask,
                true_p_len], test_return, on_unused_input='ignore')




    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 500000000000000  # look as this many examples regardless


    best_params = None
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.time()
    mid_time = start_time
    past_time= mid_time
    epoch = 0
    done_looping = False


    #para_list, Q_list, label_list, mask, vocab_size=load_train()
    n_train_batches=train_size/batch_size
#     remain_train=train_size%batch_size
    train_batch_start=list(numpy.arange(n_train_batches)*batch_size)+[train_size-batch_size]


    n_test_batches=test_size/test_batch_size
#     remain_test=test_size%batch_size
    test_batch_start=list(numpy.arange(n_test_batches)*test_batch_size)+[test_size-test_batch_size]


    max_F1_acc=0.0
    max_exact_acc=0.0
    cost_i=0.0
    train_ids = range(train_size)
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1

        random.shuffle(train_ids)
        iter_accu=0
        for para_id in train_batch_start:
            # iter means how many batches have been runed, taking into loop
            iter = (epoch - 1) * n_train_batches + iter_accu +1
            iter_accu+=1
            train_id_batch = train_ids[para_id:para_id+batch_size]
            boundary_labels_batch = train_word_label_list[train_id_batch]
            ans_label_list = []
            for i in range(batch_size):
                start = boundary_labels_batch[i][0]+i*p_len_limit
                end = boundary_labels_batch[i][1]+i*p_len_limit
                ans_label_list+=range(start, end+1)
            ans_label_list = numpy.asarray(ans_label_list, dtype='int32')

            cost_i+= train_model(
                                 train_para_list[train_id_batch],
                                 train_Q_list[train_id_batch],
                                 train_span_label_list[train_id_batch],
                                 boundary_labels_batch,
                                 ans_label_list,
                                 train_para_mask[train_id_batch],
                                 train_Q_mask[train_id_batch],
                                 train_para_char_list[train_id_batch],
                                 train_Q_char_list[train_id_batch],
                                 train_para_char_mask[train_id_batch],
                                 train_Q_char_mask[train_id_batch],
                                 p_len_limit)


            #print iter
            if iter%100==0:
                print 'Epoch ', epoch, 'iter '+str(iter)+' average cost: '+str(cost_i/iter), 'uses ', (time.time()-past_time)/60.0, 'min'
                print 'Testing...'
                past_time = time.time()
                pred_dict={}
                q_amount=0
                for test_para_id in test_batch_start:
                    batch_predict_ids=test_model(
                                                 test_para_list[test_para_id:test_para_id+test_batch_size],
                                                 test_Q_list[test_para_id:test_para_id+test_batch_size],
                                                 test_para_mask[test_para_id:test_para_id+test_batch_size],
                                                 test_Q_mask[test_para_id:test_para_id+test_batch_size],
                                                 test_para_char_list[test_para_id:test_para_id+test_batch_size],
                                                 test_Q_char_list[test_para_id:test_para_id+test_batch_size],
                                                 test_para_char_mask[test_para_id:test_para_id+test_batch_size],
                                                 test_Q_char_mask[test_para_id:test_para_id+test_batch_size],
                                                 test_p_len_limit)
                    test_para_wordlist_batch=test_para_wordlist_list[test_para_id:test_para_id+test_batch_size]
                    q_ids_batch=q_idlist[test_para_id:test_para_id+test_batch_size]
                    q_amount+=test_batch_size

                    for q in range(test_batch_size): #for each question
                        pred_ans=decode_predict_id(batch_predict_ids[q], test_para_wordlist_batch[q])
                        q_id=q_ids_batch[q]
                        pred_dict[q_id]=pred_ans
#                         print q_id, test_para_wordlist_batch[q],'\t',pred_ans
                with codecs.open(rootPath+'predictions.txt', 'w', 'utf-8') as outfile:
                    json.dump(pred_dict, outfile)
                F1_acc, exact_acc = standard_eval(rootPath+'dev-v1.1.json', rootPath+'predictions.txt')
                if F1_acc> max_F1_acc:
                    max_F1_acc=F1_acc
                if exact_acc> max_exact_acc:
                    max_exact_acc=exact_acc
#                     if max_exact_acc > max_EM:
#                         store_model_to_file(rootPath+'Best_Paras_google_'+str(max_exact_acc), params)
#                         print 'Finished storing best  params at:', max_exact_acc
                print 'current average F1:', F1_acc, '\t\tmax F1:', max_F1_acc, 'current  exact:', exact_acc, '\t\tmax exact_acc:', max_exact_acc






            if patience <= iter:
                done_looping = True
                break

        print 'Epoch ', epoch, 'uses ', (time.time()-mid_time)/60.0, 'min'
        mid_time = time.time()

        #print 'Batch_size: ', update_freq
    end_time = time.time()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i,'\
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    return max_exact_acc


if __name__ == '__main__':
    '''
learning_rate=0.01, n_epochs=3, batch_size=100, emb_size=300, char_emb_size=20, hidden_size=300,
                    L2_weight=0.0001, p_len_limit=400, test_p_len_limit=100, q_len_limit=20, char_len=15, filter_size = [5,5],
                    char_filter_size=5, margin=0.85
    '''
    # evaluate_lenet5()
    lr_list=[0.01,0.005,0.001,0.02,0.03,0.05]
    hidden_list=[300,250,200,150,350,400]
    batch_list=[100,80,60,40,120,150]

    best_acc=0.0
    best_lr=0.01
    for lr in lr_list:
        acc_test= evaluate_lenet5(learning_rate=lr)
        if acc_test>best_acc:
            best_lr=lr
            best_acc=acc_test
        print '\t\t\t\tcurrent best_acc:', best_acc

    best_hidden=300
    for hidden in hidden_list:
        acc_test= evaluate_lenet5(learning_rate=best_lr, hidden_size=hidden)
        if acc_test>best_acc:
            best_hidden=hidden
            best_acc=acc_test
        print '\t\t\t\tcurrent best_acc:', best_acc

    best_batch=100
    for batch in batch_list:
        acc_test= evaluate_lenet5(learning_rate=best_lr,  hidden_size=best_hidden,   batch_size=batch)
        if acc_test>best_acc:
            best_batch=batch
            best_acc=acc_test
        print '\t\t\t\tcurrent best_acc:', best_acc
    #
    # best_maxlen=60
    # for maxlen in maxlen_list:
    #     acc_test= evaluate_lenet5(learning_rate=best_lr,  emb_size=best_emb,   batch_size=best_batch, maxSentLen=maxlen)
    #     if acc_test>best_acc:
    #         best_maxlen=maxlen
    #         best_acc=acc_test
    #     print '\t\t\t\tcurrent best_acc:', best_acc
    print 'Hyper tune finished, best test acc: ', best_acc, ' by  lr: ', best_lr, ' hidden: ', best_hidden, ' batch: ', best_batch
