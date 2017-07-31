import numpy
import theano

from theano.scalar import Composite
from theano.scalar import add, sub, true_div, mul

import theano.tensor as T
from theano.tensor.nnet import conv
from cis.deep.utils.theano import debug_print
from WPDefined import repeat_whole_matrix, repeat_whole_tensor
from logistic_sgd import LogisticRegression
import numpy as np
from scipy.spatial.distance import cosine
import cPickle



def cosine_matrix1_matrix2_rowwise(M1, M2):
    #assume both matrix are in shape (batch, hidden)
    dot_prod=T.sum(M1*M2, axis=1) #batch
    norm1=T.sqrt(1e-20+T.sum(M1**2,axis=1)) #batch
    norm2=T.sqrt(1e-20+T.sum(M2**2,axis=1)) #batch
    return dot_prod/(norm1*norm2+1e-20)

def cosine_tensor3(T1, T2, dim):
    #for example dim=2
    dot_prod = T.sum(T1*T2, axis=dim)
    norm1 = T.sqrt(1e-20+T.sum(T1**2,axis=dim))
    norm2 = T.sqrt(1e-20+T.sum(T2**2,axis=dim))
    return dot_prod/(norm1*norm2+1e-20)
def create_AttentionMatrix_para(rng, n_in, n_out):

    W1_values = numpy.asarray(rng.uniform(
            low=-numpy.sqrt(6. / (n_in + n_out)),
            high=numpy.sqrt(6. / (n_in + n_out)),
            size=(n_in, n_out)), dtype=theano.config.floatX)  # @UndefinedVariable
    W1 = theano.shared(value=W1_values, name='W1', borrow=True)
    W2_values = numpy.asarray(rng.uniform(
            low=-numpy.sqrt(6. / (n_in + n_out)),
            high=numpy.sqrt(6. / (n_in + n_out)),
            size=(n_in, n_out)), dtype=theano.config.floatX)  # @UndefinedVariable
    W2 = theano.shared(value=W2_values, name='W2', borrow=True)

#     b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)  # @UndefinedVariable
    w_values = numpy.asarray(rng.uniform(
            low=-numpy.sqrt(6. / (n_out+1)),
            high=numpy.sqrt(6. / (n_out+1)),
            size=(n_out,)), dtype=theano.config.floatX)  # @UndefinedVariable
    w = theano.shared(value=w_values, name='w', borrow=True)
    return W1,W2, w


def create_HiddenLayer_para(rng, n_in, n_out):

    W_values = numpy.asarray(rng.uniform(
            low=-numpy.sqrt(6. / (n_in + n_out)),
            high=numpy.sqrt(6. / (n_in + n_out)),
            size=(n_in, n_out)), dtype=theano.config.floatX)  # @UndefinedVariable
    W = theano.shared(value=W_values, name='W', borrow=True)

    b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)  # @UndefinedVariable
    b = theano.shared(value=b_values, name='b', borrow=True)
    return W,b

def create_Bi_GRU_para(rng, word_dim, hidden_dim):
        # Initialize the network parameters
        U = numpy.random.uniform(-numpy.sqrt(1./hidden_dim), numpy.sqrt(1./hidden_dim), (3, hidden_dim, word_dim))
        W = numpy.random.uniform(-numpy.sqrt(1./hidden_dim), numpy.sqrt(1./hidden_dim), (3, hidden_dim, hidden_dim))
        b = numpy.zeros((3, hidden_dim))
        # Theano: Created shared variables
        U = debug_print(theano.shared(name='U', value=U.astype(theano.config.floatX), borrow=True), 'U')
        W = debug_print(theano.shared(name='W', value=W.astype(theano.config.floatX), borrow=True), 'W')
        b = debug_print(theano.shared(name='b', value=b.astype(theano.config.floatX), borrow=True), 'b')

        Ub = numpy.random.uniform(-numpy.sqrt(1./hidden_dim), numpy.sqrt(1./hidden_dim), (3, hidden_dim, word_dim))
        Wb = numpy.random.uniform(-numpy.sqrt(1./hidden_dim), numpy.sqrt(1./hidden_dim), (3, hidden_dim, hidden_dim))
        bb = numpy.zeros((3, hidden_dim))
        # Theano: Created shared variables
        Ub = debug_print(theano.shared(name='Ub', value=Ub.astype(theano.config.floatX), borrow=True), 'Ub')
        Wb = debug_print(theano.shared(name='Wb', value=Wb.astype(theano.config.floatX), borrow=True), 'Wb')
        bb = debug_print(theano.shared(name='bb', value=bb.astype(theano.config.floatX), borrow=True), 'bb')
        return U, W, b, Ub, Wb, bb

def tensor_svd_each_slice(tensor3):
    svd_slices=[]
    for i in range(len(tensor3)):
        matrix_i=tensor3[i]
        svd_slices.append(matrix_svd(matrix_i))
    return numpy.asarray(svd_slices)

def matrix_svd(matrix):
    u_i, s_i, v_i = numpy.linalg.svd(matrix)
    return u_i

def create_GRU_para(rng, word_dim, hidden_dim):
        # Initialize the network parameters
        U = numpy.random.uniform(-0.01, 0.01, (3, hidden_dim, word_dim))
#         U=rng.normal(0.0, 0.01, (3, hidden_dim, word_dim))
#         U=rng.uniform(-0.01, 0.01, (3, hidden_dim, word_dim))
        W = numpy.random.uniform(-0.01, 0.01, (3, hidden_dim, hidden_dim))
#         W=rng.normal(0.0, 0.01, (3, hidden_dim, hidden_dim))
#         W=rng.uniform(-0.01, 0.01, (3, hidden_dim, hidden_dim))
        b = numpy.zeros((3, hidden_dim))
        # Theano: Created shared variables
        U = theano.shared(name='U', value=U.astype(theano.config.floatX), borrow=True)
        W =theano.shared(name='W', value=W.astype(theano.config.floatX), borrow=True)
        b = theano.shared(name='b', value=b.astype(theano.config.floatX), borrow=True)
        return U, W, b
def create_LSTM_para(rng, word_dim, hidden_dim):
    params={}
    #W play with input dimension
    W = rng.normal(0.0, 0.01, (word_dim, 4*hidden_dim))
    params['W'] = theano.shared(name='W', value=W.astype(theano.config.floatX), borrow=True)
    #U play with hidden states
    U = rng.normal(0.0, 0.01, (hidden_dim, 4*hidden_dim))
    params['U'] = theano.shared(name='U', value=U.astype(theano.config.floatX), borrow=True)
    b = numpy.zeros((4 * hidden_dim,))
    params['b'] = theano.shared(name='b', value=b.astype(theano.config.floatX), borrow=True)

    return params
def create_ensemble_para(rng, fan_in, fan_out):
#         W=rng.normal(0.0, 0.01, (fan_out,fan_in))
#
#         W =theano.shared(name='W', value=W.astype(theano.config.floatX), borrow=True)



        # initialize weights with random weights
        W_bound = numpy.sqrt(6. /(fan_in + fan_out))
        W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=(fan_out,fan_in)),
            dtype=theano.config.floatX),
                               borrow=True)

        return W

def create_highw_para(rng, fan_in, fan_out):

        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=(fan_out,fan_in)),
            dtype=theano.config.floatX),
                               borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((fan_out,), dtype=theano.config.floatX)
        b = theano.shared(value=b_values, borrow=True)
        return W, b

def create_conv_para(rng, filter_shape):
        fan_in = numpy.prod(filter_shape[1:])
        fan_out = filter_shape[0] * numpy.prod(filter_shape[2:])
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=theano.config.floatX),
                               borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        b = theano.shared(value=b_values, borrow=True)
        return W, b

def create_rnn_para(rng, dim):
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (2*dim + dim))
#         Whh = theano.shared(numpy.asarray(
#             rng.uniform(low=-W_bound, high=W_bound, size=(dim, dim)),
#             dtype=theano.config.floatX),
#                                borrow=True)
#         Wxh = theano.shared(numpy.asarray(
#             rng.uniform(low=-W_bound, high=W_bound, size=(dim, dim)),
#             dtype=theano.config.floatX),
#                                borrow=True)
        W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=(2*dim, dim)),
            dtype=theano.config.floatX),
                               borrow=True)
        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((dim,), dtype=theano.config.floatX)
        b = theano.shared(value=b_values, borrow=True)
        return W, b


def Wide_Conv_Narrow(rng, input_tensor3, batch, emb, length, hidden, width, W_wi, b_wi, W_na, b_na):
    input_tensor4=input_tensor3.dimshuffle(0,'x',1,2) #(batch, 1, emb_size, len)
    wide_filter_shape=(hidden, 1, emb, width)
    wide_image_shape = (batch, 1, emb, length)
    wide_layer=Wide_Conv_with_input_para(rng, input_tensor4, wide_filter_shape, wide_image_shape, W_wi, b_wi)
    wide_output=wide_layer.wide_conv_out_tensor4 #(batch, 1, hidden, len+width-1)

    na_filter_shape=(hidden, 1, hidden, width)
    na_image_shape = (batch, 1, hidden, length+width-1)
    na_layer=Conv_with_input_para(rng, wide_output, na_filter_shape, na_image_shape, W_na, b_na)
    na_output_tensor3=na_layer.narrow_conv_out_tensor3 #(batch, hidden, len)

    return na_output_tensor3

class Wide_Conv_with_input_para(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, W, b):
        #input: (batch, 1, hidden ,len)
        assert image_shape[1] == filter_shape[1]
        self.input = input
        self.input_hidden=filter_shape[2]
        self.W = W
        self.b = b

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape, border_mode='full')    #(batch, kernels, 2*hidden-1, len+filter_width-1)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        conv_with_bias = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        extract_part=conv_with_bias[:,:,self.input_hidden-1:self.input_hidden,:] #(batch, kernels, 1, len+filter-width-1)
        wide_conv_out = extract_part.reshape((image_shape[0], 1, filter_shape[0], image_shape[3]+filter_shape[3]-1)) #(batch, 1, kernerl, ishape[1]-filter_size1[1]+1)

        self.wide_conv_out_tensor4=wide_conv_out  #(batch, 1, kernels, len+width-1)
        self.wide_conv_out_tensor3=extract_part.reshape((image_shape[0], filter_shape[0], image_shape[3]+filter_shape[3]-1))#(batch,  kernels, len+width-1)


class Conv_with_Mask(object):
    """we define CNN by input tensor3 and output tensor3, like RNN, filter width must by 3,5,7..."""

    def __init__(self, rng, input_tensor3, mask_matrix, filter_shape, image_shape, W, b):
        assert image_shape[1] == filter_shape[1]
        zero_pad_tensor4_1 = T.zeros((input_tensor3.shape[0], 1, input_tensor3.shape[1], filter_shape[3]/2), dtype=theano.config.floatX)+1e-8  # to get rid of nan in CNN gradient
        input = T.concatenate([zero_pad_tensor4_1,input_tensor3.dimshuffle(0,'x',1,2),
                    zero_pad_tensor4_1], axis=3)        #(batch_size, 1, emb_size, maxsenlen+width-1)

        self.input = input
        self.W = W
        self.b = b

        pad_images_shape=(image_shape[0], image_shape[1], image_shape[2], image_shape[3]+filter_shape[3]-1)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W,
                filter_shape=filter_shape, image_shape=pad_images_shape, border_mode='valid')    #here, we should pad enough zero padding for input

        conv_with_bias = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        conv_output_tensor3=conv_with_bias.reshape((image_shape[0], filter_shape[0], image_shape[3])) #(batch, 1, kernerl, ishape[1]-filter_size1[1]+1)

        self.masked_conv_out=conv_output_tensor3*mask_matrix.dimshuffle(0,'x',1) #(batch, hidden_size, len)

        mask_for_conv_output=T.repeat(mask_matrix.dimshuffle(0,'x',1), filter_shape[0], axis=1) #(batch_size, emb_size, maxSentLen-filter_size+1)
        mask_for_conv_output=(1.0-mask_for_conv_output)*(mask_for_conv_output-10)
        masked_conv_output=self.masked_conv_out+mask_for_conv_output      #mutiple mask with the conv_out to set the features by UNK to zero
        self.maxpool_vec=T.max(masked_conv_output, axis=2) #(batch_size, hidden_size) # each sentence then have an embedding of length hidden_size

        self.params = [self.W, self.b]

class Conv_with_input_para(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, W, b):
        assert image_shape[1] == filter_shape[1]
        self.input = input
        self.W = W
        self.b = b

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape, border_mode='valid')    #here, we should pad enough zero padding for input

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        conv_with_bias = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        narrow_conv_out=conv_with_bias.reshape((image_shape[0], 1, filter_shape[0], image_shape[3]-filter_shape[3]+1)) #(batch, 1, kernerl, ishape[1]-filter_size1[1]+1)

        self.narrow_conv_out=narrow_conv_out
        self.narrow_conv_out_tensor3=conv_with_bias.reshape((image_shape[0], filter_shape[0], image_shape[3]-filter_shape[3]+1))

        #pad filter_size-1 zero embeddings at both sides
        left_padding = T.zeros((image_shape[0], 1, filter_shape[0], filter_shape[3]-1), dtype=theano.config.floatX)
        right_padding = T.zeros((image_shape[0], 1, filter_shape[0], filter_shape[3]-1), dtype=theano.config.floatX)
        self.output = T.concatenate([left_padding, narrow_conv_out, right_padding], axis=3)
        self.output_max_pooling_vec=T.max(narrow_conv_out.reshape((narrow_conv_out.shape[2], narrow_conv_out.shape[3])), axis=1)

        # store parameters of this layer
        self.params = [self.W, self.b]

class RNN_with_input_para(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, rnn_Whh, rnn_Wxh, rnn_b, dim):
        self.input = input.transpose(1,0) #iterate over first dim
        self.Whh = rnn_Whh
        self.Wxh=rnn_Wxh
        self.b = rnn_b
        self.h0 = theano.shared(name='h0',
                                value=numpy.zeros(dim,
                                dtype=theano.config.floatX))
        def recurrence(x_t, h_tm1):
            w_t = T.nnet.sigmoid(T.dot(x_t, self.Wxh)
                                 + T.dot(h_tm1, self.Whh) + self.b)
            h_t=h_tm1*w_t+x_t*(1-w_t)
#             s_t = T.nnet.softmax(T.dot(h_t, self.w) + self.b)
            return h_t

        h, _ = theano.scan(fn=recurrence,
                                sequences=self.input,
                                outputs_info=self.h0,#[self.h0, None],
                                n_steps=self.input.shape[0])
        self.output=h.reshape((self.input.shape[0], self.input.shape[1])).transpose(1,0)


        # store parameters of this layer
        self.params = [self.Whh, self.Wxh, self.b]

def Matrix_Bit_Shift(input_matrix): # shit each column
    input_matrix=debug_print(input_matrix, 'input_matrix')

    def shift_at_t(t):
        shifted_matrix=debug_print(T.concatenate([input_matrix[:,t:], input_matrix[:,:t]], axis=1), 'shifted_matrix')
        return shifted_matrix

    tensor,_ = theano.scan(fn=shift_at_t,
                            sequences=T.arange(input_matrix.shape[1]),
                            n_steps=input_matrix.shape[1])

    return tensor

class Bi_GRU_Matrix_Input(object):
    def __init__(self, X, word_dim, hidden_dim, U, W, b, U_b, W_b, b_b, bptt_truncate):
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate

        def forward_prop_step(x_t, s_t1_prev):
            # GRU Layer 1
            z_t1 =T.nnet.sigmoid(U[0].dot(x_t) + W[0].dot(s_t1_prev) + b[0])
            r_t1 = T.nnet.sigmoid(U[1].dot(x_t) + W[1].dot(s_t1_prev) + b[1])
            c_t1 = T.tanh(U[2].dot(x_t) + W[2].dot(s_t1_prev * r_t1) + b[2])
            s_t1 = (T.ones_like(z_t1) - z_t1) * c_t1 + z_t1 * s_t1_prev
            return s_t1

        s, updates = theano.scan(
            forward_prop_step,
            sequences=X.transpose(1,0),
            truncate_gradient=self.bptt_truncate,
            outputs_info=dict(initial=T.zeros(self.hidden_dim)))

#         self.output_matrix=debug_print(s.transpose(), 'GRU_Matrix_Input.output_matrix')
#         self.output_vector_mean=T.mean(self.output_matrix, axis=1)
#         self.output_vector_max=T.max(self.output_matrix, axis=1)
#         self.output_vector_last=self.output_matrix[:,-1]
        #backward
        X_b=X[:,::-1]
        def backward_prop_step(x_t_b, s_t1_prev_b):
            # GRU Layer 1
            z_t1_b =T.nnet.sigmoid(U_b[0].dot(x_t_b) + W_b[0].dot(s_t1_prev_b) + b_b[0])
            r_t1_b = T.nnet.sigmoid(U_b[1].dot(x_t_b) + W_b[1].dot(s_t1_prev_b) + b_b[1])
            c_t1_b = T.tanh(U_b[2].dot(x_t_b) + W_b[2].dot(s_t1_prev_b * r_t1_b) + b_b[2])
            s_t1_b = (T.ones_like(z_t1_b) - z_t1_b) * c_t1_b + z_t1_b * s_t1_prev_b
            return s_t1_b

        s_b, updates_b = theano.scan(
            backward_prop_step,
            sequences=X_b.transpose(1,0),
            truncate_gradient=self.bptt_truncate,
            outputs_info=dict(initial=T.zeros(self.hidden_dim)))
        #dim: hidden_dim*2
#         output_matrix=T.concatenate([s.transpose(), s_b.transpose()[:,::-1]], axis=0)
        output_matrix=s.transpose()+s_b.transpose()[:,::-1]
        self.output_matrix=output_matrix+X # add input feature maps


        self.output_vector_mean=T.mean(self.output_matrix, axis=1)
        self.output_vector_max=T.max(self.output_matrix, axis=1)
        #dim: hidden_dim*4
        self.output_vector_last=T.concatenate([self.output_matrix[:,-1], self.output_matrix[:,0]], axis=0)

class Bi_GRU_Tensor3_Input(object):
    def __init__(self, T, lefts, rights, hidden_dim, U, W, b, Ub,Wb,bb):
        T=debug_print(T,'T')
        lefts=debug_print(lefts, 'lefts')
        rights=debug_print(rights, 'rights')
        def recurrence(matrix, left, right):
            sub_matrix=debug_print(matrix[:,left:-right], 'sub_matrix')
            GRU_layer=Bi_GRU_Matrix_Input(sub_matrix, sub_matrix.shape[0], hidden_dim,U,W,b, Ub,Wb,bb, -1)
            return GRU_layer.output_vector_mean
        new_M, updates = theano.scan(recurrence,
                                     sequences=[T, lefts, rights],
                                     outputs_info=None)
        self.output=debug_print(new_M.transpose(), 'Bi_GRU_Tensor3_Input.output')

class GRU_Matrix_Input(object):
    def __init__(self, X, word_dim, hidden_dim, U, W, b, bptt_truncate):
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate

        def forward_prop_step(x_t, s_t1_prev):
            # GRU Layer 1
            z_t1 =debug_print( T.nnet.sigmoid(U[0].dot(x_t) + W[0].dot(s_t1_prev) + b[0]), 'z_t1')
            r_t1 = debug_print(T.nnet.sigmoid(U[1].dot(x_t) + W[1].dot(s_t1_prev) + b[1]), 'r_t1')
            c_t1 = debug_print(T.tanh(U[2].dot(x_t) + W[2].dot(s_t1_prev * r_t1) + b[2]), 'c_t1')
            s_t1 = debug_print((T.ones_like(z_t1) - z_t1) * c_t1 + z_t1 * s_t1_prev, 's_t1')
            return s_t1

        s, updates = theano.scan(
            forward_prop_step,
            sequences=X.transpose(1,0),
            truncate_gradient=self.bptt_truncate,
            outputs_info=dict(initial=T.zeros(self.hidden_dim)))

        self.output_matrix=debug_print(s.transpose(), 'GRU_Matrix_Input.output_matrix')
        self.output_vector_mean=T.mean(self.output_matrix, axis=1)
        self.output_vector_max=T.max(self.output_matrix, axis=1)
        self.output_vector_last=self.output_matrix[:,-1]

class GRU_Tensor3_Input(object):
    def __init__(self, T, lefts, rights, hidden_dim, U, W, b):
        T=debug_print(T,'T')
        lefts=debug_print(lefts, 'lefts')
        rights=debug_print(rights, 'rights')
        def recurrence(matrix, left, right):
            sub_matrix=debug_print(matrix[:,left:-right], 'sub_matrix')
            GRU_layer=GRU_Matrix_Input(sub_matrix, sub_matrix.shape[0], hidden_dim,U,W,b, -1)
            return GRU_layer.output_vector_mean
        new_M, updates = theano.scan(recurrence,
                                     sequences=[T, lefts, rights],
                                     outputs_info=None)
        self.output=debug_print(new_M.transpose(), 'GRU_Tensor3_Input.output')

def create_params_WbWAE(input_dim, output_dim):
    W = numpy.random.uniform(-numpy.sqrt(1./output_dim), numpy.sqrt(1./output_dim), (6, output_dim, input_dim))
    w = numpy.random.uniform(-numpy.sqrt(1./output_dim), numpy.sqrt(1./output_dim), (1,output_dim))

    W = theano.shared(name='W', value=W.astype(theano.config.floatX))
    w = theano.shared(name='w', value=w.astype(theano.config.floatX))

    return W, w

class Word_by_Word_Attention_EntailmentPaper(object):
    def __init__(self, l_hidden_M, r_hidden_M, W_y,W_h,W_r, w, W_t, W_p, W_x, r_dim):
        self.Y=l_hidden_M
        self.H=r_hidden_M
        self.attention_dim=r_dim
        self.r0 = theano.shared(name='r0', value=numpy.zeros(self.attention_dim, dtype=theano.config.floatX))
        def loop(h_t, r_t_1):
            M_t=T.tanh(W_y.dot(self.Y)+(W_h.dot(h_t)+W_r.dot(r_t_1)).dimshuffle(0,'x'))
            alpha_t=T.nnet.softmax(w.dot(M_t))
            r_t=self.Y.dot(alpha_t.reshape((self.Y.shape[1],1)))+T.tanh(W_t.dot(r_t_1))

            r_t=T.sum(M_t, axis=1)
            return r_t

        r, updates= theano.scan(loop,
                                sequences=self.H.transpose(),
                                outputs_info=self.r0
                                )

        H_star=T.tanh(W_p.dot(r[-1]+W_x.dot(self.H[:,-1])))
        self.output=H_star
class Bd_GRU_Batch_Tensor_Input_with_Mask_with_MatrixInit(object):
    # Bidirectional GRU Layer.
    def __init__(self, X, Mask, MatrixInit, hidden_dim, U, W, b, Ub, Wb, bb):
        fwd = GRU_Batch_Tensor_Input_with_Mask_with_MatrixInit(X, Mask, MatrixInit, hidden_dim, U, W, b)
        bwd = GRU_Batch_Tensor_Input_with_Mask_with_MatrixInit(X[:,:,::-1], Mask[:,::-1], MatrixInit, hidden_dim, Ub, Wb, bb)

#         output_tensor=T.concatenate([fwd.output_tensor, bwd.output_tensor[:,:,::-1]], axis=1)
        #for word level rep
        output_tensor=fwd.output_tensor+bwd.output_tensor[:,:,::-1]
        self.output_tensor=output_tensor+X[:,:output_tensor.shape[1],:] # add initialized emb

        #for final sentence rep
#         sent_output_tensor=fwd.output_tensor+bwd.output_tensor
#         self.output_tensor=output_tensor+X # add initialized emb
#         self.output_sent_rep=self.output_tensor[:,:,-1]
        self.output_sent_rep_maxpooling=fwd.output_tensor[:,:,-1]+bwd.output_tensor[:,:,-1]
#         self.output_sent_rep_maxpooling=T.concatenate([fwd.output_tensor[:,:,-1], bwd.output_tensor[:,:,-1]], axis=1)

class GRU_Batch_Tensor_Input_with_Mask_with_MatrixInit(object):
    def __init__(self, X, Mask, MatrixInit, hidden_dim, U, W, b):
        #now, X is (batch, emb_size, sentlength)
        #Mask is a matrix with (batch, sentlength)
        #MatrixInit (batch, hidden)
        self.hidden_dim = hidden_dim
#         self.bptt_truncate = bptt_truncate
        self.M=Mask.T

        new_tensor=X.dimshuffle(2,1,0)

        def forward_prop_step(x_t, mask, s_t1_prev):
            # GRU Layer 1
            z_t1 =T.nnet.sigmoid(U[0].dot(x_t) + W[0].dot(s_t1_prev) + T.repeat(b[0].reshape((hidden_dim,1)), X.shape[0], axis=1)) #maybe here has a bug, as b is vector while dot product is matrix
            r_t1 = T.nnet.sigmoid(U[1].dot(x_t) + W[1].dot(s_t1_prev) + T.repeat(b[1].reshape((hidden_dim,1)), X.shape[0], axis=1))
            c_t1 = T.tanh(U[2].dot(x_t) + W[2].dot(s_t1_prev * r_t1) + T.repeat(b[2].reshape((hidden_dim,1)), X.shape[0], axis=1))
            s_t1 = (T.ones_like(z_t1) - z_t1) * c_t1 + z_t1 * s_t1_prev

            s_t1_m=s_t1*mask[None,:]+(1.0-mask[None,:])*s_t1_prev

            return s_t1_m

        s, updates = theano.scan(
            forward_prop_step,
            sequences=[new_tensor, self.M],
            outputs_info=MatrixInit.T)

#         self.output_matrix=debug_print(s.transpose(), 'GRU_Matrix_Input.output_matrix')
        self.output_tensor=s.dimshuffle(2,1,0)  #(batch, emb_size, sentlength) again

        self.output_sent_rep=self.output_tensor[:,:,-1]
class Bd_GRU_Batch_Tensor_Input_with_Mask(object):
    # Bidirectional GRU Layer.
    def __init__(self, X, Mask, hidden_dim, U, W, b, Ub, Wb, bb):
        fwd = GRU_Batch_Tensor_Input_with_Mask(X, Mask, hidden_dim, U, W, b)
        bwd = GRU_Batch_Tensor_Input_with_Mask(X[:,:,::-1], Mask[:,::-1], hidden_dim, Ub, Wb, bb)

        self.output_tensor_conc=T.concatenate([fwd.output_tensor, bwd.output_tensor[:,:,::-1]], axis=1) #(batch, 2*hidden , len)
        #for word level rep
        output_tensor=fwd.output_tensor+bwd.output_tensor[:,:,::-1]
        self.output_tensor=output_tensor#+X[:,:output_tensor.shape[1],:] # add initialized emb

        #for final sentence rep
#         sent_output_tensor=fwd.output_tensor+bwd.output_tensor
#         self.output_tensor=output_tensor+X # add initialized emb
#         self.output_sent_rep=self.output_tensor[:,:,-1]
        self.output_sent_rep_maxpooling=fwd.output_tensor[:,:,-1]+bwd.output_tensor[:,:,-1] #(batch, hidden)
        self.output_sent_rep_conc=T.concatenate([fwd.output_tensor[:,:,-1], bwd.output_tensor[:,:,-1]], axis=1) #(batch, 2*hidden)

class Bd_GRU_Batch_Tensor_Input_with_Mask_Concate(object):
    # Bidirectional GRU Layer.
    def __init__(self, X, Mask, hidden_dim, U, W, b, Ub, Wb, bb):
        fwd = GRU_Batch_Tensor_Input_with_Mask(X, Mask, hidden_dim, U, W, b)
        bwd = GRU_Batch_Tensor_Input_with_Mask(X[:,:,::-1], Mask[:,::-1], hidden_dim, Ub, Wb, bb)

        output_tensor=T.concatenate([fwd.output_tensor, bwd.output_tensor[:,:,::-1]], axis=1) #(batch, 2*hidden, len)
        self.output_tensor=output_tensor  ##(batch, 2*hidden, len)
        #for final sentence rep
        self.output_sent_rep_maxpooling=T.concatenate([fwd.output_tensor[:,:,-1], bwd.output_tensor[:,:,0]], axis=1) #(batch, 2*hidden)
class GRU_Batch_Tensor_Input_with_Mask(object):
    def __init__(self, X, Mask, hidden_dim, U, W, b):
        #now, X is (batch, emb_size, sentlength)
        #Mask is a matrix with (batch, sentlength)
        self.hidden_dim = hidden_dim
#         self.bptt_truncate = bptt_truncate
        self.M=Mask.T

        new_tensor=X.dimshuffle(2,1,0)

        def forward_prop_step(x_t, mask, s_t1_prev):
            # GRU Layer 1
            z_t1 =T.nnet.sigmoid(U[0].dot(x_t) + W[0].dot(s_t1_prev) + T.repeat(b[0].reshape((hidden_dim,1)), X.shape[0], axis=1)) #maybe here has a bug, as b is vector while dot product is matrix
            r_t1 = T.nnet.sigmoid(U[1].dot(x_t) + W[1].dot(s_t1_prev) + T.repeat(b[1].reshape((hidden_dim,1)), X.shape[0], axis=1))
            c_t1 = T.tanh(U[2].dot(x_t) + W[2].dot(s_t1_prev * r_t1) + T.repeat(b[2].reshape((hidden_dim,1)), X.shape[0], axis=1))
            s_t1 = (T.ones_like(z_t1) - z_t1) * c_t1 + z_t1 * s_t1_prev

            s_t1_m=s_t1*mask[None,:]+(1.0-mask[None,:])*s_t1_prev

            return s_t1_m

        s, updates = theano.scan(
            forward_prop_step,
            sequences=[new_tensor, self.M],
            outputs_info=dict(initial=T.zeros((self.hidden_dim, X.shape[0]))))

#         self.output_matrix=debug_print(s.transpose(), 'GRU_Matrix_Input.output_matrix')
        self.output_tensor=s.dimshuffle(2,1,0)  #(batch, emb_size, sentlength) again

        self.output_sent_rep=self.output_tensor[:,:,-1]

class Bd_LSTM_Batch_Tensor_Input_with_Mask(object):
    # Bidirectional GRU Layer.
    def __init__(self, X, Mask, hidden_dim, fwd_params, bwd_params):
        fwd = LSTM_Batch_Tensor_Input_with_Mask(X, Mask, hidden_dim, fwd_params)
        bwd = LSTM_Batch_Tensor_Input_with_Mask(X[:,:,::-1], Mask[:,::-1], hidden_dim, bwd_params)

        self.output_tensor_conc=T.concatenate([fwd.output_tensor, bwd.output_tensor[:,:,::-1]], axis=1) #(batch, 2*hidden, len)
        self.output_sent_rep_conc=T.concatenate([fwd.output_sent_rep, bwd.output_sent_rep], axis=1) #(batch, 2*hidden)

class Bd_LSTM_Batch_Tensor_Input_with_Mask_Concate(object):
    # Bidirectional GRU Layer.
    def __init__(self, X, Mask, hidden_dim, fwd_tparams, bwd_tparams):
        fwd = LSTM_Batch_Tensor_Input_with_Mask(X, Mask, hidden_dim, fwd_tparams)
        bwd = LSTM_Batch_Tensor_Input_with_Mask(X[:,:,::-1], Mask[:,::-1], hidden_dim, bwd_tparams)

        output_tensor=T.concatenate([fwd.output_tensor, bwd.output_tensor[:,:,::-1]], axis=1)
        #for word level rep
#         output_tensor=fwd.output_tensor+bwd.output_tensor[:,:,::-1]
        self.output_tensor=output_tensor#(batch, 2*hidden ,len)
        self.forward_output=fwd.output_tensor
        self.backward_output = bwd.output_tensor[:,:,::-1]
        #for final sentence rep
#         self.output_sent_rep_maxpooling=fwd.output_tensor[:,:,-1]+bwd.output_tensor[:,:,-1]
        self.output_sent_rep_maxpooling=T.concatenate([fwd.output_tensor[:,:,-1], bwd.output_tensor[:,:,0]], axis=1) #(batch, 2*hidden)
class LSTM_Batch_Tensor_Input_with_Mask(object):
#     def __init__(self, X, Mask, hidden_dim, U, W, b):
    def __init__(self, X, Mask, hidden_size, tparams ):
        #X (batch, emb_size, senLen), Mask (batch, senLen)
        state_below=X.dimshuffle(2,0,1)
        mask=Mask.T
        # state_below, (senLen, batch_size, emb_size)
        nsteps = state_below.shape[0] #sentence length, as LSTM or GRU needs to know how many words/slices to deal with sequentially
        if state_below.ndim == 3:
            n_samples = state_below.shape[1] #batch_size
        else:
            n_samples = 1

        assert mask is not None

        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n * dim:(n + 1) * dim]
            return _x[:, n * dim:(n + 1) * dim]

        def _step(m_, x_, h_, c_): #mask, x_ current word embedding, h_ and c_ are hidden states in preceding step of two hidden layers
            preact = T.dot(h_, tparams['U'])
            preact += x_
            '''
            already remember that variables generated by "sigmoid" below are "gates", they are not hidden states, they are used to control how much information of
            some hidden states to be used for other steps, so "i", "f", "o" below are gates
            '''
            i = T.nnet.sigmoid(_slice(preact, 0, hidden_size))
            f = T.nnet.sigmoid(_slice(preact, 1, hidden_size))
            o = T.nnet.sigmoid(_slice(preact, 2, hidden_size))
            c = T.tanh(_slice(preact, 3, hidden_size))

            c = f * c_ + i * c
            c = m_[:, None] * c + (1. - m_)[:, None] * c_

            h = o * T.tanh(c)
            h = m_[:, None] * h + (1. - m_)[:, None] * h_

            return h, c

        state_below = (T.dot(state_below, tparams['W']) + tparams['b'])

        dim_proj = hidden_size

        '''
        pls understand this theano.scan by referring to the description of GRU's theano.scan
        '''

        rval, updates = theano.scan(_step,
                                    sequences=[mask, state_below],
                                    outputs_info=[T.alloc(numpy.asarray(0., dtype=theano.config.floatX),n_samples,dim_proj),
                                                  T.alloc(numpy.asarray(0., dtype=theano.config.floatX),n_samples,dim_proj)],
                                    n_steps=nsteps)
        self.output_tensor = rval[0].dimshuffle(1,2,0) # rval[0]: (nsamples, batch, hidden_size), out_tensor: (batch, hidden, nsamples)
        self.output_sent_rep=self.output_tensor[:,:,-1] # (batch, hidden_size)



class GRU_Batch_Tensor_Input(object):
    def __init__(self, X, hidden_dim, U, W, b, bptt_truncate):
        #now, X is (batch, emb_size, sentlength)
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate

        new_tensor=debug_print(X.dimshuffle(2,1,0), 'new_tensor')

        def forward_prop_step(x_t, s_t1_prev):
            # GRU Layer 1
            z_t1 =debug_print( T.nnet.sigmoid(U[0].dot(x_t) + W[0].dot(s_t1_prev) + T.repeat(b[0].reshape((hidden_dim,1)), X.shape[0], axis=1)), 'z_t1')  #maybe here has a bug, as b is vector while dot product is matrix
            r_t1 = debug_print(T.nnet.sigmoid(U[1].dot(x_t) + W[1].dot(s_t1_prev) + T.repeat(b[1].reshape((hidden_dim,1)), X.shape[0], axis=1)), 'r_t1')
            c_t1 = debug_print(T.tanh(U[2].dot(x_t) + W[2].dot(s_t1_prev * r_t1) + T.repeat(b[2].reshape((hidden_dim,1)), X.shape[0], axis=1)), 'c_t1')
            s_t1 = debug_print((T.ones_like(z_t1) - z_t1) * c_t1 + z_t1 * s_t1_prev, 's_t1')
            return s_t1

        s, updates = theano.scan(
            forward_prop_step,
            sequences=new_tensor,
            truncate_gradient=self.bptt_truncate,
            outputs_info=dict(initial=T.zeros((self.hidden_dim, X.shape[0]))))

#         self.output_matrix=debug_print(s.transpose(), 'GRU_Matrix_Input.output_matrix')
        self.output_tensor=debug_print(s.dimshuffle(2,1,0), 'self.output_tensor')

#         d0s, d2s=Dim_Align(self.output_tensor.shape[0])
#         d0s=debug_print(d0s, 'd0s')
#         d2s=debug_print(d2s, 'd2s')
#         self.output_matrix=debug_print(self.output_tensor[d0s,:,d2s].transpose(), 'self.output_matrix')  # before transpose, its (dim, hidden_size), each row is a hidden state

        d0s=Dim_Align_new(self.output_tensor.shape[0])
        self.output_matrix=self.output_tensor.transpose(0,2,1).reshape((self.output_tensor.shape[0]*self.output_tensor.shape[2], self.output_tensor.shape[1]))[d0s].transpose()
        self.dim=debug_print(self.output_tensor.shape[0]*(self.output_tensor.shape[0]+1)/2, 'self.dim')
        self.output_sent_rep=self.output_tensor[0,:,-1]
        self.output_sent_hiddenstates=self.output_tensor[0]
        self.ph_lengths=lenghs_phrases(self.output_tensor.shape[0])





def Dim_Align(x):
#     x = tt.lscalar()
    def series_sum(n):
        return n * (n + 1) / 2
    yz = T.zeros((series_sum(x),), dtype='int64')
#     yz = T.zeros((series_sum(x),), dtype='int32')#for gpu

    def step(x1, y1, y2):
        i = series_sum(x1)
        j = series_sum(x1 + 1)
        z1 = T.arange(x1 + 1)
        z2 = z1[::-1]
        y1 = T.set_subtensor(y1[i:j], z1)
        y2 = T.set_subtensor(y2[i:j], z2)
        return y1, y2

    (r1, r2), _ = theano.scan(step, sequences=[T.arange(x)], outputs_info=[yz, yz])
#     return theano.function([x], [y1[-1], y2[-1]])
    return r1[-1], r2[-1]

def Dim_Align_new(x):
    #there is a bug, when input x=1, namely the sentence has only one word
#     x = tt.lscalar()
    def series_sum(n):
        return n * (n + 1) / 2
    yz = T.zeros((series_sum(x),), dtype='int64')
#     yz = T.zeros((series_sum(x),), dtype='int32')#for gpu

    def step(x1, y1):
        i = series_sum(x1)
        j = series_sum(x1 + 1)
#         z1 = T.arange(x1 + 1)
        z1=T.arange(x1,x1*x+1,x-1)[::-1]
        y1 = T.set_subtensor(y1[i:j], z1)
        return y1

    r1, _ = theano.scan(step, sequences=[T.arange(x)], outputs_info=yz)
#     return theano.function([x], [y1[-1], y2[-1]])
    return r1[-1]

def lenghs_phrases(x):
#     x = tt.lscalar()
    def series_sum(n):
        return n * (n + 1) / 2
    yz = T.zeros((series_sum(x),), dtype='int64')
#     yz = T.zeros((series_sum(x),), dtype='int32')#for gpu

    def step(x1, y1):
        i = series_sum(x1)
        j = series_sum(x1 + 1)
#         z1 = T.arange(x1 + 1)
        z1=T.arange(1,x1+2)
        y1 = T.set_subtensor(y1[i:j], z1)
        return y1

    r1, _ = theano.scan(step, sequences=[T.arange(x)], outputs_info=yz) #[0,1,2,3]
#     return theano.function([x], [y1[-1], y2[-1]])
    return r1[-1]
class biRNN_with_input_para(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, rnn_W, rnn_b, rnn_W_r, rnn_b_r, dim):
        self.input = debug_print(input.transpose(1,0), 'self.input') #iterate over first dim
        self.rnn_W=rnn_W
        self.b = rnn_b

        self.Wr = rnn_W_r
        self.b_r = rnn_b_r
        self.h0 = theano.shared(name='h0',
                                value=numpy.zeros(dim,
                                dtype=theano.config.floatX))
        self.h0_r = theano.shared(name='h0',
                                value=numpy.zeros(dim,
                                dtype=theano.config.floatX))
        def recurrence(x_t, h_tm1):
            concate=T.concatenate([x_t,h_tm1], axis=0)
#             w_t = T.nnet.sigmoid(T.dot(x_t, self.Wxh)
#                                  + T.dot(h_tm1, self.Whh) + self.b)
            w_t = T.nnet.sigmoid(T.dot(concate, self.rnn_W) + self.b)
            h_t=h_tm1*w_t+x_t*(1-w_t)
#             s_t = T.nnet.softmax(T.dot(h_t, self.w) + self.b)
            return h_t

        h, _ = theano.scan(fn=recurrence,
                                sequences=self.input,
                                outputs_info=self.h0,#[self.h0, None],
                                n_steps=self.input.shape[0])
        self.output_one=debug_print(h.reshape((self.input.shape[0], self.input.shape[1])).transpose(1,0), 'self.output_one')
        #reverse direction
        self.input_two=debug_print(input[:,::-1].transpose(1,0), 'self.input_two')
        def recurrence_r(x_t_r, h_tm1_r):
            concate=T.concatenate([x_t_r,h_tm1_r], axis=0)
#             w_t = T.nnet.sigmoid(T.dot(x_t, self.Wxh)
#                                  + T.dot(h_tm1, self.Whh) + self.b)
            w_t = T.nnet.sigmoid(T.dot(concate, self.Wr) + self.b_r)
#             h_t=h_tm1*w_t+x_t*(1-w_t)
# #             s_t = T.nnet.softmax(T.dot(h_t, self.w) + self.b)
#
#
#             w_t = T.nnet.sigmoid(T.dot(x_t_r, self.Wxh_r)
#                                  + T.dot(h_tm1_r, self.Whh_r) + self.b_r)
            h_t=h_tm1_r*w_t+x_t_r*(1-w_t)
#             s_t = T.nnet.softmax(T.dot(h_t, self.w) + self.b)
            return h_t
        h_r, _ = theano.scan(fn=recurrence_r,
                                sequences=self.input_two,
                                outputs_info=self.h0_r,#[self.h0, None],
                                n_steps=self.input_two.shape[0])
        self.output_two=debug_print(h_r.reshape((self.input_two.shape[0], self.input_two.shape[1])).transpose(1,0)[:,::-1], 'self.output_two')
        self.output=debug_print(self.output_one+self.output_two, 'self.output')
#         # store parameters of this layer
#         self.params = [self.Whh, self.Wxh, self.b]
class Conv_with_input_para_one_col_featuremap(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, W, b):
        assert image_shape[1] == filter_shape[1]
        self.input = input
        self.W = W
        self.b = b

        input=debug_print(input, 'input_Conv_with_input_para_one_col_featuremap')
        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape, border_mode='full')    #here, we should pad enough zero padding for input

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        conv_out=debug_print(conv_out, 'conv_out')
        conv_with_bias = debug_print(T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x')), 'conv_with_bias')
        posi=conv_with_bias.shape[2]/2
        conv_with_bias=conv_with_bias[:,:,posi:(posi+1),:]
        wide_conv_out=debug_print(conv_with_bias.reshape((image_shape[0], 1, filter_shape[0], image_shape[3]+filter_shape[3]-1)), 'wide_conv_out') #(batch, 1, kernerl, ishape[1]+filter_size1[1]-1)


        self.output_tensor = debug_print(wide_conv_out, 'self.output_tensor')
        self.output_matrix=debug_print(wide_conv_out.reshape((filter_shape[0], image_shape[3]+filter_shape[3]-1)), 'self.output_matrix')
        self.output_sent_rep_Dlevel=debug_print(T.max(self.output_matrix, axis=1), 'self.output_sent_rep_Dlevel')


        # store parameters of this layer
        self.params = [self.W, self.b]


class Conv(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights
        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape
        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)
        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)
        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = filter_shape[0] * numpy.prod(filter_shape[2:])
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=theano.config.floatX),
                               borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape, border_mode='valid')    #here, we should pad enough zero padding for input

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        conv_with_bias = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        narrow_conv_out=conv_with_bias.reshape((image_shape[0], 1, filter_shape[0], image_shape[3]-filter_shape[3]+1)) #(batch, 1, kernerl, ishape[1]-filter_size1[1]+1)

        #pad filter_size-1 zero embeddings at both sides
        left_padding = 1e-20+T.zeros((image_shape[0], 1, filter_shape[0], filter_shape[3]-1), dtype=theano.config.floatX)
        right_padding = 1e-20+T.zeros((image_shape[0], 1, filter_shape[0], filter_shape[3]-1), dtype=theano.config.floatX)
        self.output = T.concatenate([left_padding, narrow_conv_out, right_padding], axis=3)


        # store parameters of this layer
        self.params = [self.W, self.b]

class Average_Pooling_for_Top(object):
    """The input is output of Conv: a tensor.  The output here should also be tensor"""

    def __init__(self, rng, input_l, input_r, kern, left_l, right_l, left_r, right_r, length_l, length_r, dim, topk): # length_l, length_r: valid lengths after conv
#     layer3_DQ=Average_Pooling_for_Top(rng, input_l=layer2_DQ.output, input_r=layer2_Q.output_sent_rep_Dlevel, kern=nkerns[1],
#                      left_l=left_D, right_l=right_D, left_r=0, right_r=0,
#                       length_l=len_D+filter_sents[1]-1, length_r=1,
#                        dim=maxDocLength+filter_sents[1]-1, topk=3)


        fan_in = kern #kern numbers
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = kern
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=(kern, kern)),
            dtype=theano.config.floatX),
                               borrow=True) #a weight matrix kern*kern

        input_r_matrix=debug_print(input_r,'input_r_matrix')

        input_l_matrix=debug_print(input_l.reshape((input_l.shape[2], input_l.shape[3])), 'origin_input_l_matrix')
        input_l_matrix=debug_print(input_l_matrix[:, left_l:(input_l_matrix.shape[1]-right_l)],'input_l_matrix')



        simi_matrix=compute_simi_feature_matrix_with_column(input_l_matrix, input_r_matrix, length_l, 1, dim) #(input.shape[0]/2, input.shape[1], input.shape[3], input.shape[3])
        simi_question=debug_print(simi_matrix.reshape((1, length_l)),'simi_question')

        neighborsArgSorted = T.argsort(simi_question, axis=1)
        kNeighborsArg = neighborsArgSorted[:,-topk:]#only average the top 3 vectors
        kNeighborsArgSorted = T.sort(kNeighborsArg, axis=1) # make y indices in acending lie
        jj = kNeighborsArgSorted.flatten()
        sub_matrix=input_l_matrix.transpose(1,0)[jj].reshape((topk, input_l_matrix.shape[0]))
        sub_weights=simi_question.transpose(1,0)[jj].reshape((topk, 1))

        sub_weights =sub_weights/T.sum(sub_weights) #L-1 normalize attentions
        #weights_answer=simi_answer/T.sum(simi_answer)
        #concate=T.concatenate([weights_question, weights_answer], axis=1)
        #reshaped_concate=concate.reshape((input.shape[0], 1, 1, length_last_dim))

        sub_weights=T.repeat(sub_weights, kern, axis=1)
        #weights_answer_matrix=T.repeat(weights_answer, kern, axis=0)

        #with attention
#         output_D_doc_level_rep=debug_print(T.sum(sub_matrix*sub_weights, axis=0), 'output_D_doc_level_rep') # is a column now
        output_D_doc_level_rep=debug_print(T.max(sub_matrix, axis=0), 'output_D_doc_level_rep') # is a column now
        self.output_D_doc_level_rep=output_D_doc_level_rep



        self.params = [self.W]



class Average_Pooling(object):
    """The input is output of Conv: a tensor.  The output here should also be tensor"""

    def __init__(self, rng, input_D, input_r, kern, left_D, right_D,left_D_s, right_D_s, left_r, right_r, length_D_s, length_r, dim, doc_len, topk): # length_l, length_r: valid lengths after conv
#     layer1_DQ=Average_Pooling(rng, input_l=layer0_D_output, input_r=layer0_Q_output, kern=nkerns[0],
#                                       left_D=left_D, right_D=right_D,
#                      left_l=left_D_s, right_l=right_D_s, left_r=left_Q, right_r=right_Q,
#                       length_l=len_D_s+filter_words[1]-1, length_r=len_Q+filter_words[1]-1,
#                        dim=maxSentLength+filter_words[1]-1, doc_len=maxDocLength, topk=3)


        fan_in = kern #kern numbers
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = kern
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=(kern, kern)),
            dtype=theano.config.floatX),
                               borrow=True) #a weight matrix kern*kern
        input_r_matrix=debug_print(input_r.reshape((input_r.shape[2], input_r.shape[3])),'origin_input_r_matrix')
        input_r_matrix=debug_print(input_r_matrix[:, left_r:(input_r_matrix.shape[1]-right_r)],'input_r_matrix')
        valid_D_s=[]
        for i in range(left_D, doc_len-right_D): # only consider valid sentences in doc
            input_l=input_D[i,:,:,:] # order-3 tensor
            left_l=left_D_s[i]
            right_l=right_D_s[i]
            length_l=length_D_s[i]


            input_l_matrix=debug_print(input_l.reshape((input_D.shape[2], input_D.shape[3])), 'origin_input_l_matrix')
            input_l_matrix=debug_print(input_l_matrix[:, left_l:(input_l_matrix.shape[1]-right_l)],'input_l_matrix')



            simi_tensor=compute_simi_feature_batch1_new(input_l_matrix, input_r_matrix, length_l, length_r, self.W, dim) #(input.shape[0]/2, input.shape[1], input.shape[3], input.shape[3])
            simi_question=debug_print(T.max(simi_tensor, axis=1).reshape((1, length_l)),'simi_question')

            neighborsArgSorted = T.argsort(simi_question, axis=1)
            kNeighborsArg = neighborsArgSorted[:,-topk:]#only average the top 3 vectors
            kNeighborsArgSorted = T.sort(kNeighborsArg, axis=1) # make y indices in acending lie
            jj = kNeighborsArgSorted.flatten()
            sub_matrix=input_l_matrix.transpose(1,0)[jj].reshape((topk, input_l_matrix.shape[0]))
            sub_weights=simi_question.transpose(1,0)[jj].reshape((topk, 1))

            sub_weights =sub_weights/T.sum(sub_weights) #L-1 normalize attentions
            #weights_answer=simi_answer/T.sum(simi_answer)
            #concate=T.concatenate([weights_question, weights_answer], axis=1)
            #reshaped_concate=concate.reshape((input.shape[0], 1, 1, length_last_dim))

            sub_weights=T.repeat(sub_weights, kern, axis=1)
            #weights_answer_matrix=T.repeat(weights_answer, kern, axis=0)

            #with attention
            dot_l=debug_print(T.sum(sub_matrix*sub_weights, axis=0).transpose(1,0), 'dot_l') # is a column now
            valid_D_s.append(dot_l)
            #dot_r=debug_print(T.sum(input_r_matrix*weights_answer_matrix, axis=1),'dot_r')
            '''
            #without attention
            dot_l=debug_print(T.sum(input_l_matrix, axis=1), 'dot_l') # first add 1e-20 for each element to make non-zero input for weight gradient
            dot_r=debug_print(T.sum(input_r_matrix, axis=1),'dot_r')
            '''
            '''
            #with attention, then max pooling
            dot_l=debug_print(T.max(input_l_matrix*weights_question_matrix, axis=1), 'dot_l') # first add 1e-20 for each element to make non-zero input for weight gradient
            dot_r=debug_print(T.max(input_r_matrix*weights_answer_matrix, axis=1),'dot_r')
            '''
            #norm_l=debug_print(T.sqrt((dot_l**2).sum()),'norm_l')
            #norm_r=debug_print(T.sqrt((dot_r**2).sum()), 'norm_r')

            #self.output_vector_l=debug_print((dot_l/norm_l).reshape((1, kern)),'output_vector_l')
            #self.output_vector_r=debug_print((dot_r/norm_r).reshape((1, kern)), 'output_vector_r')
        valid_matrix=T.concatenate(valid_D_s, axis=1)
        left_padding = T.zeros((input_l_matrix.shape[0], left_D), dtype=theano.config.floatX)
        right_padding = T.zeros((input_l_matrix.shape[0], right_D), dtype=theano.config.floatX)
        matrix_padded = T.concatenate([left_padding, valid_matrix, right_padding], axis=1)
        self.output_D=matrix_padded
        self.output_D_valid_part=valid_matrix
        self.output_QA_sent_level_rep=T.mean(input_r_matrix, axis=1)

        #now, average pooling by comparing self.output_QA and self.output_D_valid_part
        simi_matrix=compute_simi_feature_matrix_with_column(self.output_D_valid_part, self.output_QA, doc_len-left_D-right_D, 1, doc_len) #(input.shape[0]/2, input.shape[1], input.shape[3], input.shape[3])
        simi_question=debug_print(simi_matrix.reshape((1, doc_len-left_D-right_D)),'simi_question')

        neighborsArgSorted = T.argsort(simi_question, axis=1)
        kNeighborsArg = neighborsArgSorted[:,-topk:]#only average the top 3 vectors
        kNeighborsArgSorted = T.sort(kNeighborsArg, axis=1) # make y indices in acending lie
        jj = kNeighborsArgSorted.flatten()
        sub_matrix=self.output_D_valid_part.transpose(1,0)[jj].reshape((topk, self.output_D_valid_part.shape[0]))
        sub_weights=simi_question.transpose(1,0)[jj].reshape((topk, 1))

        sub_weights =sub_weights/T.sum(sub_weights) #L-1 normalize attentions
        #weights_answer=simi_answer/T.sum(simi_answer)
        #concate=T.concatenate([weights_question, weights_answer], axis=1)
        #reshaped_concate=concate.reshape((input.shape[0], 1, 1, length_last_dim))

        sub_weights=T.repeat(sub_weights, kern, axis=1)
        #weights_answer_matrix=T.repeat(weights_answer, kern, axis=0)

        #with attention
        output_D_sent_level_rep=debug_print(T.sum(sub_matrix*sub_weights, axis=0).transpose(1,0), 'output_D_sent_level_rep') # is a column now
        self.output_D_sent_level_rep=output_D_sent_level_rep



        self.params = [self.W]

class Average_Pooling_Scan(object):
    """The input is output of Conv: a tensor.  The output here should also be tensor"""

    def __init__(self, rng, input_D, input_r, kern, left_D, right_D,left_D_s, right_D_s, left_r, right_r, length_D_s, length_r, dim, doc_len, topk): # length_l, length_r: valid lengths after conv
#     layer1_DQ=Average_Pooling(rng, input_l=layer0_D_output, input_r=layer0_Q_output, kern=nkerns[0],
#                                       left_D=left_D, right_D=right_D,
#                      left_l=left_D_s, right_l=right_D_s, left_r=left_Q, right_r=right_Q,
#                       length_l=len_D_s+filter_words[1]-1, length_r=len_Q+filter_words[1]-1,
#                        dim=maxSentLength+filter_words[1]-1, doc_len=maxDocLength, topk=3)


        fan_in = kern #kern numbers
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = kern
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=(kern, kern)),
            dtype=theano.config.floatX),
                               borrow=True) #a weight matrix kern*kern

#         input_tensor_l=T.dtensor4("input_tensor_l")
#         input_tensor_r=T.dtensor4("input_tensor_r")
#         kern_scan=T.lscalar("kern_scan")
#         length_D_s_scan=T.lvector("length_D_s_scan")
#         left_D_s_scan=T.lvector("left_D_s_scan")
#         right_D_s_scan=T.lvector("right_D_s_scan")
#         length_r_scan=T.lscalar("length_r_scan")
#         left_r_scan=T.lscalar("left_r_scan")
#         right_r_scan=T.lscalar("right_r_scan")
#         dim_scan=T.lscalar("dim_scan")
#         topk_scan=T.lscalar("topk_scan")



        def sub_operation(input_l, length_l, left_l, right_l, input_r, kernn , length_r, left_r, right_r, dim, topk):
            input_l_matrix=debug_print(input_l.reshape((input_l.shape[1], input_l.shape[2])), 'origin_input_l_matrix')#input_l should be order3 tensor now
            input_l_matrix=debug_print(input_l_matrix[:, left_l:(input_l_matrix.shape[1]-right_l)],'input_l_matrix')
#             input_r_matrix=debug_print(input_r.reshape((input_r.shape[2], input_r.shape[3])),'origin_input_r_matrix')#input_r should be order4 tensor still
#             input_r_matrix=debug_print(input_r_matrix[:, left_r:(input_r_matrix.shape[1]-right_r)],'input_r_matrix')
#
#
#             simi_tensor=compute_simi_feature_batch1_new(input_l_matrix, input_r_matrix, length_l, length_r, self.W, dim) #(input.shape[0]/2, input.shape[1], input.shape[3], input.shape[3])
#             simi_question=debug_print(T.max(simi_tensor, axis=1).reshape((1, length_l)),'simi_question')
#
#             neighborsArgSorted = T.argsort(simi_question, axis=1)
#             kNeighborsArg = neighborsArgSorted[:,-topk:]#only average the top 3 vectors
#             kNeighborsArgSorted = T.sort(kNeighborsArg, axis=1) # make y indices in acending lie
#             jj = kNeighborsArgSorted.flatten()
#             sub_matrix=input_l_matrix.transpose(1,0)[jj].reshape((topk, input_l_matrix.shape[0]))
#             sub_weights=simi_question.transpose(1,0)[jj].reshape((topk, 1))
#
#             sub_weights =sub_weights/T.sum(sub_weights) #L-1 normalize attentions
#             sub_weights=T.repeat(sub_weights, kernn, axis=1)
#             dot_l=debug_print(T.sum(sub_matrix*sub_weights, axis=0), 'dot_l') # is a column now
#             dot_l=T.max(sub_matrix, axis=0)
            dot_l=debug_print(T.max(input_l_matrix, axis=1), 'dot_l') # max pooling
            return dot_l



#         results, updates = theano.scan(fn=sub_operation,
#                                        outputs_info=None,
#                                        sequences=[input_tensor_l, length_D_s_scan, left_D_s_scan, right_D_s_scan],
#                                        non_sequences=[input_tensor_r, kern_scan, length_r_scan, left_r_scan, right_r_scan, dim_scan, topk_scan])

        results, updates = theano.scan(fn=sub_operation,
                                       outputs_info=None,
                                       sequences=[input_D[left_D:doc_len-right_D], length_D_s[left_D: doc_len-right_D], left_D_s[left_D: doc_len-right_D], right_D_s[left_D: doc_len-right_D]],
                                       non_sequences=[input_r, kern, length_r, left_r, right_r, dim, topk])

#         scan_function = theano.function(inputs=[input_tensor_l, input_tensor_r, kern_scan, length_D_s_scan, left_D_s_scan, right_D_s_scan, length_r_scan, left_r_scan, right_r_scan, dim_scan, topk_scan],
#                                         outputs=results,
#                                         updates=updates)
#
#
#
#         sents=scan_function(input_D[left_D:doc_len-right_D], input_r, kern,
#                             length_D_s[left_D: doc_len-right_D], left_D_s[left_D: doc_len-right_D], right_D_s[left_D: doc_len-right_D],
#                             length_r,
#                             left_r,
#                             right_r,
#                             dim,
#                             topk)
        sents=results
        input_r_matrix=debug_print(input_r.reshape((input_r.shape[2], input_r.shape[3])),'origin_input_r_matrix')
        input_r_matrix=debug_print(input_r_matrix[:, left_r:(input_r_matrix.shape[1]-right_r)],'input_r_matrix')


        valid_matrix=debug_print(sents.transpose(1,0), 'valid_matrix')
        left_padding = T.zeros((input_D.shape[2], left_D), dtype=theano.config.floatX)
        right_padding = T.zeros((input_D.shape[2], right_D), dtype=theano.config.floatX)
        matrix_padded = T.concatenate([left_padding, valid_matrix, right_padding], axis=1)
        self.output_D=matrix_padded   #it shows the second conv for doc has input of all sentences
        self.output_D_valid_part=valid_matrix
        self.output_QA_sent_level_rep=T.max(input_r_matrix, axis=1)

        #now, average pooling by comparing self.output_QA and self.output_D_valid_part, choose one key sentence
        topk=1
        simi_matrix=debug_print(compute_simi_feature_matrix_with_column(self.output_D_valid_part, self.output_QA_sent_level_rep, doc_len-left_D-right_D, 1, doc_len), 'simi_matrix_matrix_with_column') #(input.shape[0]/2, input.shape[1], input.shape[3], input.shape[3])
        simi_question=debug_print(simi_matrix.reshape((1, doc_len-left_D-right_D)),'simi_question')

        neighborsArgSorted = T.argsort(simi_question, axis=1)
        kNeighborsArg = neighborsArgSorted[:,-topk:]#only average the top 3 vectors
        kNeighborsArgSorted = T.sort(kNeighborsArg, axis=1) # make y indices in acending lie
        jj = kNeighborsArgSorted.flatten()
        sub_matrix=self.output_D_valid_part.transpose(1,0)[jj].reshape((topk, self.output_D_valid_part.shape[0]))
        sub_weights=simi_question.transpose(1,0)[jj].reshape((topk, 1))

        sub_weights =sub_weights/T.sum(sub_weights) #L-1 normalize attentions
        #weights_answer=simi_answer/T.sum(simi_answer)
        #concate=T.concatenate([weights_question, weights_answer], axis=1)
        #reshaped_concate=concate.reshape((input.shape[0], 1, 1, length_last_dim))

        sub_weights=T.repeat(sub_weights, kern, axis=1)
        #weights_answer_matrix=T.repeat(weights_answer, kern, axis=0)

        #with attention
#         output_D_sent_level_rep=debug_print(T.sum(sub_matrix*sub_weights, axis=0), 'output_D_sent_level_rep') # is a column now
        output_D_sent_level_rep=debug_print(T.max(sub_matrix, axis=0), 'output_D_sent_level_rep') # is a column now
        self.output_D_sent_level_rep=output_D_sent_level_rep



        self.params = [self.W]

class GRU_Average_Pooling_Scan(object):
    """The input is output of Conv: a tensor.  The output here should also be tensor"""

    def __init__(self, rng, input_D, input_r, kern, left_D, right_D, dim, doc_len, topk): # length_l, length_r: valid lengths after conv
#     layer1_DQ=Average_Pooling(rng, input_l=layer0_D_output, input_r=layer0_Q_output, kern=nkerns[0],
#                                       left_D=left_D, right_D=right_D,
#                      left_l=left_D_s, right_l=right_D_s, left_r=left_Q, right_r=right_Q,
#                       length_l=len_D_s+filter_words[1]-1, length_r=len_Q+filter_words[1]-1,
#                        dim=maxSentLength+filter_words[1]-1, doc_len=maxDocLength, topk=3)


#         fan_in = kern #kern numbers
#         # each unit in the lower layer receives a gradient from:
#         # "num output feature maps * filter height * filter width" /
#         #   pooling size
#         fan_out = kern
#         # initialize weights with random weights
#         W_bound = numpy.sqrt(6. / (fan_in + fan_out))
#         self.W = theano.shared(numpy.asarray(
#             rng.uniform(low=-W_bound, high=W_bound, size=(kern, kern)),
#             dtype=theano.config.floatX),
#                                borrow=True) #a weight matrix kern*kern

#         input_tensor_l=T.dtensor4("input_tensor_l")
#         input_tensor_r=T.dtensor4("input_tensor_r")
#         kern_scan=T.lscalar("kern_scan")
#         length_D_s_scan=T.lvector("length_D_s_scan")
#         left_D_s_scan=T.lvector("left_D_s_scan")
#         right_D_s_scan=T.lvector("right_D_s_scan")
#         length_r_scan=T.lscalar("length_r_scan")
#         left_r_scan=T.lscalar("left_r_scan")
#         right_r_scan=T.lscalar("right_r_scan")
#         dim_scan=T.lscalar("dim_scan")
#         topk_scan=T.lscalar("topk_scan")



#         def sub_operation(input_l, length_l, left_l, right_l, input_r, kernn , length_r, left_r, right_r, dim, topk):
#             input_l_matrix=debug_print(input_l.reshape((input_l.shape[1], input_l.shape[2])), 'origin_input_l_matrix')#input_l should be order3 tensor now
#             input_l_matrix=debug_print(input_l_matrix[:, left_l:(input_l_matrix.shape[1]-right_l)],'input_l_matrix')
# #             input_r_matrix=debug_print(input_r.reshape((input_r.shape[2], input_r.shape[3])),'origin_input_r_matrix')#input_r should be order4 tensor still
# #             input_r_matrix=debug_print(input_r_matrix[:, left_r:(input_r_matrix.shape[1]-right_r)],'input_r_matrix')
# #
# #
# #             simi_tensor=compute_simi_feature_batch1_new(input_l_matrix, input_r_matrix, length_l, length_r, self.W, dim) #(input.shape[0]/2, input.shape[1], input.shape[3], input.shape[3])
# #             simi_question=debug_print(T.max(simi_tensor, axis=1).reshape((1, length_l)),'simi_question')
# #
# #             neighborsArgSorted = T.argsort(simi_question, axis=1)
# #             kNeighborsArg = neighborsArgSorted[:,-topk:]#only average the top 3 vectors
# #             kNeighborsArgSorted = T.sort(kNeighborsArg, axis=1) # make y indices in acending lie
# #             jj = kNeighborsArgSorted.flatten()
# #             sub_matrix=input_l_matrix.transpose(1,0)[jj].reshape((topk, input_l_matrix.shape[0]))
# #             sub_weights=simi_question.transpose(1,0)[jj].reshape((topk, 1))
# #
# #             sub_weights =sub_weights/T.sum(sub_weights) #L-1 normalize attentions
# #             sub_weights=T.repeat(sub_weights, kernn, axis=1)
# #             dot_l=debug_print(T.sum(sub_matrix*sub_weights, axis=0), 'dot_l') # is a column now
# #             dot_l=T.max(sub_matrix, axis=0)
#             dot_l=debug_print(T.max(input_l_matrix, axis=1), 'dot_l') # max pooling
#             return dot_l
#
#
#
# #         results, updates = theano.scan(fn=sub_operation,
# #                                        outputs_info=None,
# #                                        sequences=[input_tensor_l, length_D_s_scan, left_D_s_scan, right_D_s_scan],
# #                                        non_sequences=[input_tensor_r, kern_scan, length_r_scan, left_r_scan, right_r_scan, dim_scan, topk_scan])
#
#         results, updates = theano.scan(fn=sub_operation,
#                                        outputs_info=None,
#                                        sequences=[input_D[left_D:doc_len-right_D], length_D_s[left_D: doc_len-right_D], left_D_s[left_D: doc_len-right_D], right_D_s[left_D: doc_len-right_D]],
#                                        non_sequences=[input_r, kern, length_r, left_r, right_r, dim, topk])

#         scan_function = theano.function(inputs=[input_tensor_l, input_tensor_r, kern_scan, length_D_s_scan, left_D_s_scan, right_D_s_scan, length_r_scan, left_r_scan, right_r_scan, dim_scan, topk_scan],
#                                         outputs=results,
#                                         updates=updates)
#
#
#
#         sents=scan_function(input_D[left_D:doc_len-right_D], input_r, kern,
#                             length_D_s[left_D: doc_len-right_D], left_D_s[left_D: doc_len-right_D], right_D_s[left_D: doc_len-right_D],
#                             length_r,
#                             left_r,
#                             right_r,
#                             dim,
#                             topk)
#         sents=results
#         input_r_matrix=debug_print(input_r.reshape((input_r.shape[2], input_r.shape[3])),'origin_input_r_matrix')
#         input_r_matrix=debug_print(input_r_matrix[:, left_r:(input_r_matrix.shape[1]-right_r)],'input_r_matrix')


        valid_matrix=debug_print(input_D, 'valid_matrix')
        left_padding = T.zeros((input_D.shape[0], left_D), dtype=theano.config.floatX)
        right_padding = T.zeros((input_D.shape[0], right_D), dtype=theano.config.floatX)
        matrix_padded = T.concatenate([left_padding, valid_matrix, right_padding], axis=1)
        self.output_D=matrix_padded   #it shows the second conv for doc has input of all sentences
        self.output_D_valid_part=valid_matrix
        self.output_QA_sent_level_rep=input_r

        #now, average pooling by comparing self.output_QA and self.output_D_valid_part, choose one key sentence
        topk=1
        simi_matrix=debug_print(compute_simi_feature_matrix_with_column(self.output_D_valid_part, self.output_QA_sent_level_rep, doc_len-left_D-right_D, 1, doc_len), 'simi_matrix_matrix_with_column') #(input.shape[0]/2, input.shape[1], input.shape[3], input.shape[3])
        simi_question=debug_print(simi_matrix.reshape((1, doc_len-left_D-right_D)),'simi_question')

        neighborsArgSorted = T.argsort(simi_question, axis=1)
        kNeighborsArg = neighborsArgSorted[:,-topk:]#only average the top 3 vectors
        kNeighborsArgSorted = T.sort(kNeighborsArg, axis=1) # make y indices in acending lie
        jj = kNeighborsArgSorted.flatten()
        sub_matrix=self.output_D_valid_part.transpose(1,0)[jj].reshape((topk, self.output_D_valid_part.shape[0]))
        sub_weights=simi_question.transpose(1,0)[jj].reshape((topk, 1))

        sub_weights =sub_weights/T.sum(sub_weights) #L-1 normalize attentions
        #weights_answer=simi_answer/T.sum(simi_answer)
        #concate=T.concatenate([weights_question, weights_answer], axis=1)
        #reshaped_concate=concate.reshape((input.shape[0], 1, 1, length_last_dim))

        sub_weights=T.repeat(sub_weights, kern, axis=1)
        #weights_answer_matrix=T.repeat(weights_answer, kern, axis=0)

        #with attention
#         output_D_sent_level_rep=debug_print(T.sum(sub_matrix*sub_weights, axis=0), 'output_D_sent_level_rep') # is a column now
        output_D_sent_level_rep=debug_print(T.max(sub_matrix, axis=0), 'output_D_sent_level_rep') # is a column now
        self.output_D_sent_level_rep=output_D_sent_level_rep



#         self.params = [self.W]

def dropout_standard(is_train, input, p, rng):
    """
    :type input: numpy.array
    :param input: layer or weight matrix on which dropout resp. dropconnect is applied
    :type p: float or double between 0. and 1.
    :param p: p probability of NOT dropping out a unit or connection, therefore (1.-p) is the drop rate.
    """
    srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
    mask = srng.binomial(n = 1, p = 1-p, size = input.shape, dtype = theano.config.floatX)
    return  T.switch(T.eq(is_train, 1), input * mask, input * (1 - p))


class Average_Pooling_RNN(object):
    """The input is output of Conv: a tensor.  The output here should also be tensor"""

    def __init__(self, rng, input_D, input_r, kern, left_D, right_D,doc_len, topk): # length_l, length_r: valid lengths after conv
#     layer1_DQ=Average_Pooling(rng, input_l=layer0_D_output, input_r=layer0_Q_output, kern=nkerns[0],
#                                       left_D=left_D, right_D=right_D,
#                      left_l=left_D_s, right_l=right_D_s, left_r=left_Q, right_r=right_Q,
#                       length_l=len_D_s+filter_words[1]-1, length_r=len_Q+filter_words[1]-1,
#                        dim=maxSentLength+filter_words[1]-1, doc_len=maxDocLength, topk=3)





        self.output_D_valid_part=input_D
        self.output_QA_sent_level_rep=input_r

        #now, average pooling by comparing self.output_QA and self.output_D_valid_part, choose one key sentence
        topk=1
        simi_matrix=debug_print(compute_simi_feature_matrix_with_column(self.output_D_valid_part, self.output_QA_sent_level_rep, doc_len-left_D-right_D, 1, doc_len), 'simi_matrix_matrix_with_column') #(input.shape[0]/2, input.shape[1], input.shape[3], input.shape[3])
        simi_question=debug_print(simi_matrix.reshape((1, doc_len-left_D-right_D)),'simi_question')

        neighborsArgSorted = T.argsort(simi_question, axis=1)
        kNeighborsArg = neighborsArgSorted[:,-topk:]#only average the top 3 vectors
        kNeighborsArgSorted = T.sort(kNeighborsArg, axis=1) # make y indices in acending lie
        jj = kNeighborsArgSorted.flatten()
        sub_matrix=self.output_D_valid_part.transpose(1,0)[jj].reshape((topk, self.output_D_valid_part.shape[0]))
        sub_weights=simi_question.transpose(1,0)[jj].reshape((topk, 1))

        sub_weights =sub_weights/T.sum(sub_weights) #L-1 normalize attentions
        #weights_answer=simi_answer/T.sum(simi_answer)
        #concate=T.concatenate([weights_question, weights_answer], axis=1)
        #reshaped_concate=concate.reshape((input.shape[0], 1, 1, length_last_dim))

        sub_weights=T.repeat(sub_weights, kern, axis=1)
        #weights_answer_matrix=T.repeat(weights_answer, kern, axis=0)

        #with attention
#         output_D_sent_level_rep=debug_print(T.sum(sub_matrix*sub_weights, axis=0), 'output_D_sent_level_rep') # is a column now
        output_D_rep=debug_print(T.max(sub_matrix, axis=0), 'output_D_rep') # is a column now
        self.output_D_sent_level_rep=output_D_rep


def compute_simi_feature_batch1_new(input_l_matrix, input_r_matrix, length_l, length_r, para_matrix, dim):
    #matrix_r_after_translate=debug_print(T.dot(para_matrix, input_r_matrix), 'matrix_r_after_translate')
    matrix_r_after_translate=input_r_matrix

    input_l_tensor=input_l_matrix.dimshuffle('x',0,1)
    input_l_tensor=T.repeat(input_l_tensor, dim, axis=0)[:length_r,:,:]
    input_l_tensor=input_l_tensor.dimshuffle(2,1,0).dimshuffle(0,2,1)
    repeated_1=input_l_tensor.reshape((length_l*length_r, input_l_matrix.shape[0])).dimshuffle(1,0)

    input_r_tensor=matrix_r_after_translate.dimshuffle('x',0,1)
    input_r_tensor=T.repeat(input_r_tensor, dim, axis=0)[:length_l,:,:]
    input_r_tensor=input_r_tensor.dimshuffle(0,2,1)
    repeated_2=input_r_tensor.reshape((length_l*length_r, matrix_r_after_translate.shape[0])).dimshuffle(1,0)



    #cosine attention
    length_1=debug_print(1e-10+T.sqrt(T.sum(T.sqr(repeated_1), axis=0)),'length_1')
    length_2=debug_print(1e-10+T.sqrt(T.sum(T.sqr(repeated_2), axis=0)), 'length_2')

    multi=debug_print(repeated_1*repeated_2, 'multi')
    sum_multi=debug_print(T.sum(multi, axis=0),'sum_multi')

    list_of_simi= debug_print(sum_multi/(length_1*length_2),'list_of_simi')   #to get rid of zero length
    simi_matrix=debug_print(list_of_simi.reshape((length_l, length_r)), 'simi_matrix')


#     #euclid, effective for wikiQA
#     gap=debug_print(repeated_1-repeated_2, 'gap')
#     eucli=debug_print(T.sqrt(1e-10+T.sum(T.sqr(gap), axis=0)),'eucli')
#     simi_matrix=debug_print((1.0/(1.0+eucli)).reshape((length_l, length_r)), 'simi_matrix')


    return simi_matrix#[:length_l, :length_r]

def compute_simi_feature_matrix_with_matrix(input_l_matrix, input_r_matrix, length_l, length_r, dim):
    #this function is the same with "compute_simi_feature_batch1_new", except that this has no input parameters
    matrix_r_after_translate=input_r_matrix

    input_l_tensor=input_l_matrix.dimshuffle('x',0,1)
    input_l_tensor=T.repeat(input_l_tensor, dim, axis=0)[:length_r,:,:]
    input_l_tensor=input_l_tensor.dimshuffle(2,1,0).dimshuffle(0,2,1)
    repeated_1=input_l_tensor.reshape((length_l*length_r, input_l_matrix.shape[0])).dimshuffle(1,0)

    input_r_tensor=matrix_r_after_translate.dimshuffle('x',0,1)
    input_r_tensor=T.repeat(input_r_tensor, dim, axis=0)[:length_l,:,:]
    input_r_tensor=input_r_tensor.dimshuffle(0,2,1)
    repeated_2=input_r_tensor.reshape((length_l*length_r, matrix_r_after_translate.shape[0])).dimshuffle(1,0)



    #cosine attention
    length_1=debug_print(1e-10+T.sqrt(T.sum(T.sqr(repeated_1), axis=0)),'length_1')
    length_2=debug_print(1e-10+T.sqrt(T.sum(T.sqr(repeated_2), axis=0)), 'length_2')

    multi=debug_print(repeated_1*repeated_2, 'multi')
    sum_multi=debug_print(T.sum(multi, axis=0),'sum_multi')

    list_of_simi= debug_print(sum_multi/(length_1*length_2),'list_of_simi')   #to get rid of zero length
    simi_matrix=debug_print(list_of_simi.reshape((length_l, length_r)), 'simi_matrix')


#     #euclid, effective for wikiQA
#     gap=debug_print(repeated_1-repeated_2, 'gap')
#     eucli=debug_print(T.sqrt(1e-10+T.sum(T.sqr(gap), axis=0)),'eucli')
#     simi_matrix=debug_print((1.0/(1.0+eucli)).reshape((length_l, length_r)), 'simi_matrix')


    return simi_matrix#[:length_l, :length_r]

def compute_attention_feature_matrix_with_matrix(input_l_matrix, input_r_matrix, length_l, length_r, dim, W1, W2, w):
    #this function is the same with "compute_simi_feature_batch1_new", except that this has no input parameters
    matrix_r_after_translate=input_r_matrix

    input_l_tensor=input_l_matrix.dimshuffle('x',0,1)
    input_l_tensor=T.repeat(input_l_tensor, dim, axis=0)[:length_r,:,:]
    input_l_tensor=input_l_tensor.dimshuffle(2,1,0).dimshuffle(0,2,1)
    repeated_1=input_l_tensor.reshape((length_l*length_r, input_l_matrix.shape[0])).dimshuffle(1,0)

    input_r_tensor=matrix_r_after_translate.dimshuffle('x',0,1)
    input_r_tensor=T.repeat(input_r_tensor, dim, axis=0)[:length_l,:,:]
    input_r_tensor=input_r_tensor.dimshuffle(0,2,1)
    repeated_2=input_r_tensor.reshape((length_l*length_r, matrix_r_after_translate.shape[0])).dimshuffle(1,0)

    proj_1=W1.dot(repeated_1)
    proj_2=W2.dot(repeated_2)

    attentions=T.tanh(w.dot(proj_1+proj_2))
    attention_matrix=attentions.reshape((length_l, length_r))
#     #cosine attention
#     length_1=debug_print(1e-10+T.sqrt(T.sum(T.sqr(repeated_1), axis=0)),'length_1')
#     length_2=debug_print(1e-10+T.sqrt(T.sum(T.sqr(repeated_2), axis=0)), 'length_2')
#
#     multi=debug_print(repeated_1*repeated_2, 'multi')
#     sum_multi=debug_print(T.sum(multi, axis=0),'sum_multi')
#
#     list_of_simi= debug_print(sum_multi/(length_1*length_2),'list_of_simi')   #to get rid of zero length
#     simi_matrix=debug_print(list_of_simi.reshape((length_l, length_r)), 'simi_matrix')


#     #euclid, effective for wikiQA
#     gap=debug_print(repeated_1-repeated_2, 'gap')
#     eucli=debug_print(T.sqrt(1e-10+T.sum(T.sqr(gap), axis=0)),'eucli')
#     simi_matrix=debug_print((1.0/(1.0+eucli)).reshape((length_l, length_r)), 'simi_matrix')


    return attention_matrix#[:length_l, :length_r]

def compute_simi_feature_matrix_with_column(input_l_matrix, column, length_l, length_r, dim):
    column=column.reshape((column.shape[0],1))
    repeated_2=T.repeat(column, dim, axis=1)[:,:length_l]



    #cosine attention
    length_1=debug_print(1e-10+T.sqrt(T.sum(T.sqr(input_l_matrix), axis=0)),'length_1')
    length_2=debug_print(1e-10+T.sqrt(T.sum(T.sqr(repeated_2), axis=0)), 'length_2')

    multi=debug_print(input_l_matrix*repeated_2, 'multi')
    sum_multi=debug_print(T.sum(multi, axis=0),'sum_multi')

    list_of_simi= debug_print(sum_multi/(length_1*length_2),'list_of_simi')   #to get rid of zero length
    simi_matrix=debug_print(list_of_simi.reshape((length_l, length_r)), 'simi_matrix')


#     #euclid, effective for wikiQA
#     gap=debug_print(input_l_matrix-repeated_2, 'gap')
#     eucli=debug_print(T.sqrt(1e-10+T.sum(T.sqr(gap), axis=0)),'eucli')
#     simi_matrix=debug_print((1.0/(1.0+eucli)).reshape((length_l, length_r)), 'simi_matrix')


    return simi_matrix#[:length_l, :length_r]

def compute_simi_feature(tensor, dim, para_matrix):
    odd_tensor=debug_print(tensor[0:tensor.shape[0]:2,:,:,:],'odd_tensor')
    even_tensor=debug_print(tensor[1:tensor.shape[0]:2,:,:,:], 'even_tensor')
    even_tensor_after_translate=debug_print(T.dot(para_matrix, 1e-20+even_tensor.reshape((tensor.shape[2], dim*tensor.shape[0]/2))), 'even_tensor_after_translate')
    fake_even_tensor=debug_print(even_tensor_after_translate.reshape((tensor.shape[0]/2, tensor.shape[1], tensor.shape[2], tensor.shape[3])),'fake_even_tensor')

    repeated_1=debug_print(T.repeat(odd_tensor, dim, axis=3),'repeated_1')
    repeated_2=debug_print(repeat_whole_matrix(fake_even_tensor, dim, False),'repeated_2')
    #repeated_2=T.repeat(even_tensor, even_tensor.shape[3], axis=2).reshape((tensor.shape[0]/2, tensor.shape[1], tensor.shape[2], tensor.shape[3]**2))
    length_1=debug_print(1e-10+T.sqrt(T.sum(T.sqr(repeated_1), axis=2)),'length_1')
    length_2=debug_print(1e-10+T.sqrt(T.sum(T.sqr(repeated_2), axis=2)), 'length_2')

    multi=debug_print(repeated_1*repeated_2, 'multi')
    sum_multi=debug_print(T.sum(multi, axis=2),'sum_multi')

    list_of_simi= debug_print(sum_multi/(length_1*length_2),'list_of_simi')   #to get rid of zero length

    return list_of_simi.reshape((tensor.shape[0]/2, tensor.shape[1], tensor.shape[3], tensor.shape[3]))

def compute_acc(label_list, scores_list):
    #label_list contains 0/1, 500 as a minibatch, score_list contains score between -1 and 1, 500 as a minibatch
    if len(label_list)%500!=0 or len(scores_list)%500!=0:
        print 'len(label_list)%500: ', len(label_list)%500, ' len(scores_list)%500: ', len(scores_list)%500
        exit(0)
    if len(label_list)!=len(scores_list):
        print 'len(label_list)!=len(scores_list)', len(label_list), ' and ',len(scores_list)
        exit(0)
    correct_count=0
    total_examples=len(label_list)/500
    start_posi=range(total_examples)*500
    for i in start_posi:
        set_1=set()

        for scan in range(i, i+500):
            if label_list[scan]==1:
                set_1.add(scan)
        set_0=set(range(i, i+500))-set_1
        flag=True
        for zero_posi in set_0:
            for scan in set_1:
                if scores_list[zero_posi]> scores_list[scan]:
                    flag=False
        if flag==True:
            correct_count+=1

    return correct_count*1.0/total_examples
#def unify_eachone(tensor, left1, right1, left2, right2, dim, Np):
def top_k_pooling(matrix, sentlength_1, sentlength_2, Np):

    #tensor: (1, feature maps, 66, 66)
    #sentlength_1=dim-left1-right1
    #sentlength_2=dim-left2-right2
    #core=tensor[:,:, left1:(dim-right1),left2:(dim-right2) ]
    '''
    repeat_row=Np/sentlength_1
    extra_row=Np%sentlength_1
    repeat_col=Np/sentlength_2
    extra_col=Np%sentlength_2
    '''
    #repeat core
    matrix_1=repeat_whole_tensor(matrix, 5, True)
    matrix_2=repeat_whole_tensor(matrix_1, 5, False)

    list_values=matrix_2.flatten()
    neighborsArgSorted = T.argsort(list_values)
    kNeighborsArg = neighborsArgSorted[-(Np**2):]
    top_k_values=list_values[kNeighborsArg]


    all_max_value=top_k_values.reshape((1, Np**2))

    return all_max_value
def unify_eachone(matrix, sentlength_1, sentlength_2, Np):

    #tensor: (1, feature maps, 66, 66)
    #sentlength_1=dim-left1-right1
    #sentlength_2=dim-left2-right2
    #core=tensor[:,:, left1:(dim-right1),left2:(dim-right2) ]

    repeat_row=Np/sentlength_1
    extra_row=Np%sentlength_1
    repeat_col=Np/sentlength_2
    extra_col=Np%sentlength_2

    #repeat core
    matrix_1=repeat_whole_tensor(matrix, 5, True)
    matrix_2=repeat_whole_tensor(matrix_1, 5, False)

    new_rows=T.maximum(sentlength_1, sentlength_1*repeat_row+extra_row)
    new_cols=T.maximum(sentlength_2, sentlength_2*repeat_col+extra_col)

    #core=debug_print(core_2[:,:, :new_rows, : new_cols],'core')
    new_matrix=debug_print(matrix_2[:new_rows,:new_cols], 'new_matrix')
    #determine x, y start positions
    size_row=new_rows/Np
    remain_row=new_rows%Np
    size_col=new_cols/Np
    remain_col=new_cols%Np

    xx=debug_print(T.concatenate([T.arange(Np-remain_row+1)*size_row, (Np-remain_row)*size_row+(T.arange(remain_row)+1)*(size_row+1)]),'xx')
    yy=debug_print(T.concatenate([T.arange(Np-remain_col+1)*size_col, (Np-remain_col)*size_col+(T.arange(remain_col)+1)*(size_col+1)]),'yy')

    list_of_maxs=[]
    for i in xrange(Np):
        for j in xrange(Np):
            region=debug_print(new_matrix[xx[i]:xx[i+1], yy[j]:yy[j+1]],'region')
            #maxvalue1=debug_print(T.max(region, axis=2), 'maxvalue1')
            maxvalue=debug_print(T.max(region).reshape((1,1)), 'maxvalue')
            list_of_maxs.append(maxvalue)


    all_max_value=T.concatenate(list_of_maxs, axis=1).reshape((1, Np**2))

    return all_max_value

def softmax_tensor_by_axis(tensor, dim):
    exp_tensor = T.exp(tensor)
    summ = T.sum(exp_tensor, axis=dim)
    if dim == 1:
        return exp_tensor/summ.dimshuffle(0,'x',1)
    elif dim ==2:
        return exp_tensor/summ.dimshuffle(0,1, 'x')
    else:
        print 'softmax_tensor_by_axis wrong by dim:', dim
        exit(0)

class Create_Attention_Input_Cnn(object):
    """The input is output of Conv: a tensor.  The output here should also be tensor"""

    def __init__(self, rng, tensor_l, tensor_r, dim,kern, l_left_pad, l_right_pad, r_left_pad, r_right_pad): # length_l, length_r: valid lengths after conv
        #first reshape into matrix
        matrix_l=tensor_l.reshape((tensor_l.shape[2], tensor_l.shape[3]))
        matrix_r=tensor_r.reshape((tensor_r.shape[2], tensor_r.shape[3]))
        #start
        repeated_1=debug_print(T.repeat(matrix_l, dim, axis=1),'repeated_1') # add 10 because max_sent_length is only input for conv, conv will make size bigger
        repeated_2=debug_print(repeat_whole_tensor(matrix_r, dim, False),'repeated_2')
        '''
        #cosine attention
        length_1=debug_print(1e-10+T.sqrt(T.sum(T.sqr(repeated_1), axis=0)),'length_1')
        length_2=debug_print(1e-10+T.sqrt(T.sum(T.sqr(repeated_2), axis=0)), 'length_2')
        multi=debug_print(repeated_1*repeated_2, 'multi')
        sum_multi=debug_print(T.sum(multi, axis=0),'sum_multi')
        list_of_simi= debug_print(sum_multi/(length_1*length_2),'list_of_simi')   #to get rid of zero length
        simi_matrix=debug_print(list_of_simi.reshape((length_l, length_r)), 'simi_matrix')
        '''
        #euclid, effective for wikiQA
        gap=debug_print(repeated_1-repeated_2, 'gap')
        eucli=debug_print(T.sqrt(1e-10+T.sum(T.sqr(gap), axis=0)),'eucli')
        simi_matrix=debug_print((1.0/(1.0+eucli)).reshape((dim, dim)), 'simi_matrix')
        W_bound = numpy.sqrt(6. / (dim + kern))
        self.W = theano.shared(numpy.asarray(rng.uniform(low=-W_bound, high=W_bound, size=(kern, dim)),dtype=theano.config.floatX),borrow=True) #a weight matrix kern*kern
        matrix_l_attention=debug_print(T.dot(self.W, simi_matrix.T), 'matrix_l_attention')
        matrix_r_attention=debug_print(T.dot(self.W, simi_matrix), 'matrix_r_attention')
        #reset zero at both side
        left_zeros_l=T.set_subtensor(matrix_l_attention[:,:l_left_pad], T.zeros((matrix_l_attention.shape[0], l_left_pad), dtype=theano.config.floatX))
        right_zeros_l=T.set_subtensor(left_zeros_l[:,-l_right_pad:], T.zeros((matrix_l_attention.shape[0], l_right_pad), dtype=theano.config.floatX))
        left_zeros_r=T.set_subtensor(matrix_r_attention[:,:r_left_pad], T.zeros((matrix_r_attention.shape[0], r_left_pad), dtype=theano.config.floatX))
        right_zeros_r=T.set_subtensor(left_zeros_r[:,-r_right_pad:], T.zeros((matrix_r_attention.shape[0], r_right_pad), dtype=theano.config.floatX))
        #combine with original input matrix
        self.new_tensor_l=T.concatenate([matrix_l,right_zeros_l], axis=0).reshape((tensor_l.shape[0], 2*tensor_l.shape[1], tensor_l.shape[2], tensor_l.shape[3]))
        self.new_tensor_r=T.concatenate([matrix_r,right_zeros_r], axis=0).reshape((tensor_r.shape[0], 2*tensor_r.shape[1], tensor_r.shape[2], tensor_r.shape[3]))

        self.params=[self.W]

def Diversify_Reg(W):
    loss=((W.dot(W.T)-T.eye(n=W.shape[0], m=W.shape[0], k=0, dtype=theano.config.floatX))**2).sum()
    return loss



def normalize_matrix(M):
    norm=T.sqrt(T.sum(T.sqr(M)))
    return M/norm
def L2norm_paraList(params):
    sum=0.0
    for x in params:
        sum+=(x**2).sum()
    return sum

# def L2norm_paraList(paralist):
#     summ=0.0
#
#     for para in paralist:
#         summ+=(para** 2).mean()
#     return summ
def constant_param(value=0.0, shape=(0,)):
#     return theano.shared(lasagne.init.Constant(value).sample(shape), borrow=True)
    return theano.shared(numpy.full(shape, value, dtype=theano.config.floatX), borrow=True)


def normal_param(std=0.1, mean=0.0, shape=(0,)):
#     return theano.shared(lasagne.init.Normal(std, mean).sample(shape), borrow=True)
    U=numpy.random.normal(mean, std, shape)
    return theano.shared(name='U', value=U.astype(theano.config.floatX), borrow=True)
def cosine_simi(x, y):
    #this is better
    a = np.array(x)
    b = np.array(y)
    c = 1-cosine(a,b)
    return c

class Conv_then_GRU_then_Classify(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, concate_paragraph_input, Qs_emb, para_len_limit, q_len_limit, input_h_size_1, output_h_size, input_h_size_2, conv_width, batch_size, para_mask, q_mask, labels, layer_scalar):
        self.paragraph_para, self.paragraph_conv_output, self.paragraph_gru_output_tensor, self.paragraph_gru_output_reps=conv_then_gru(rng, concate_paragraph_input, output_h_size, input_h_size_1, conv_width, batch_size, para_len_limit, para_mask)

        self.q_para, self.q_conv_output, self.q_gru_output_tensor, self.q_gru_output_reps=conv_then_gru(rng, Qs_emb, output_h_size, input_h_size_2, conv_width, batch_size, q_len_limit, q_mask)

        LR_mask=para_mask[:,:-1]*para_mask[:,1:]
        self.classify_para, self.error, self.masked_dis=combine_for_LR(rng, output_h_size, self.paragraph_gru_output_tensor, self.q_gru_output_reps, LR_mask, batch_size, labels)
        self.masked_dis_inprediction=self.masked_dis*T.sqrt(layer_scalar)
        self.paras=self.paragraph_para+self.q_para+self.classify_para

def conv_then_gru(rng, input_tensor3, out_h_size, in_h_size, conv_width, batch_size, size_last_dim, mask):
    conv_input=input_tensor3.dimshuffle((0,'x', 1,2)) #(batch_size, 1, emb+3, maxparalen)
    conv_W, conv_b=create_conv_para(rng, filter_shape=(out_h_size, 1, in_h_size, conv_width))
    conv_para=[conv_W, conv_b]
    conv_model = Conv_with_input_para(rng, input=conv_input,
            image_shape=(batch_size, 1, in_h_size, size_last_dim),
            filter_shape=(out_h_size, 1, in_h_size, conv_width), W=conv_W, b=conv_b)
    conv_output=conv_model.narrow_conv_out #(batch, 1, hidden_size, maxparalen-1)

    U, W, b=create_GRU_para(rng, out_h_size, out_h_size)
    U_b, W_b, b_b=create_GRU_para(rng, out_h_size, out_h_size)
    gru_para=[U, W, b, U_b, W_b, b_b]
    gru_input=conv_output.reshape((conv_output.shape[0], conv_output.shape[2], conv_output.shape[3]))
    gru_mask=mask[:,:-1]*mask[:,1:]
    gru_model=Bd_GRU_Batch_Tensor_Input_with_Mask(X=gru_input, Mask=gru_mask, hidden_dim=out_h_size,U=U,W=W,b=b,Ub=U_b,Wb=W_b,bb=b_b)
    gru_output_tensor=gru_model.output_tensor #(batch, hidden_dim, para_len-1)
    gru_output_reps=gru_model.output_sent_rep_maxpooling.reshape((batch_size, 1, out_h_size)) #(batch, 2*out_size)

    overall_para= conv_para + gru_para
    return overall_para, conv_output, gru_output_tensor, gru_output_reps
def combine_for_LR(rng, hidden_size, para_reps, questions_reps, para_mask, batch_size, labels):
    #combine, then classify
    W_a1 = create_ensemble_para(rng, hidden_size, hidden_size)# init_weights((2*hidden_size, hidden_size))
    W_a2 = create_ensemble_para(rng, hidden_size, hidden_size)
    U_a = create_ensemble_para(rng, 2, hidden_size) # 3 extra features

    norm_W_a1=normalize_matrix(W_a1)
    norm_W_a2=normalize_matrix(W_a2)
    norm_U_a=normalize_matrix(U_a)

    LR_b = theano.shared(value=numpy.zeros((2,),
                                                 dtype=theano.config.floatX),  # @UndefinedVariable
                               name='LR_b', borrow=True)

    attention_paras=[W_a1, W_a2, U_a, LR_b]

    transformed_para_reps=T.tanh(T.dot(para_reps.transpose((0, 2,1)), norm_W_a2))
    transformed_q_reps=T.tanh(T.dot(questions_reps, norm_W_a1))
    #transformed_q_reps=T.repeat(transformed_q_reps, transformed_para_reps.shape[1], axis=1)

    add_both=0.5*(transformed_para_reps+transformed_q_reps)

    prior_att=add_both
    combined_size=hidden_size


    #prior_att=T.concatenate([transformed_para_reps, transformed_q_reps], axis=2)
    valid_indices=para_mask.flatten().nonzero()[0]

    layer3=LogisticRegression(rng, input=prior_att.reshape((batch_size*prior_att.shape[1], combined_size)), n_in=combined_size, n_out=2, W=norm_U_a, b=LR_b)
    #error =layer3.negative_log_likelihood(labels.flatten()[valid_indices])
    error = -T.mean(T.log(layer3.p_y_given_x)[valid_indices, labels.flatten()[valid_indices]])#[T.arange(y.shape[0]), y])

    distributions=layer3.p_y_given_x[:,-1].reshape((batch_size, para_mask.shape[1]))
    #distributions=layer3.y_pred.reshape((batch_size, para_mask.shape[1]))
    masked_dis=distributions*para_mask
    return  attention_paras, error, masked_dis

def dropout_layer(state_before, use_noise, trng):
    proj = T.switch(use_noise,
                         (state_before *
                          trng.binomial(state_before.shape,
                                        p=0.5, n=1)),
                         state_before * 0.5)
    return proj


def BatchMatchMatrix_between_2tensors(tensor1, tensor2):
    #assume both are (batch, hidden ,para_len), (batch, hidden ,q_len)
    def example_in_batch(para_matrix, q_matrix):
        #assume both are (hidden, para_len),  (hidden, q_len)
        transpose_para_matrix=para_matrix.T  #(para_len, hidden)
        interaction_matrix=T.dot(transpose_para_matrix, q_matrix) #(para_len, q_len)
        return interaction_matrix
    batch_matrix,_ = theano.scan(fn=example_in_batch,
                                   outputs_info=None,
                                   sequences=[tensor1, tensor2])    #batch_q_reps (batch, hidden, para_len)
    return batch_matrix #(batch, para_len, q_len)

def attention_dot_prod_between_2tensors(tensor1, tensor2):
    #assume both are (batch, hidden ,para_len), (batch, hidden ,q_len)
    def example_in_batch(para_matrix, q_matrix):
        #assume both are (hidden, para_len),  (hidden, q_len)
        transpose_para_matrix=para_matrix.T  #(para_len, hidden)
        interaction_matrix=T.dot(transpose_para_matrix, q_matrix) #(para_len, q_len)
        norm_interaction_matrix=T.nnet.softmax(interaction_matrix)  #(para_len, q_len)
        attended_para= T.dot(q_matrix, norm_interaction_matrix.T) #(hidden, q_len)*(q_len, para_len)=(hidden, para_len)
        norm_interaction_matrix_T=T.nnet.softmax(interaction_matrix.T)#(q_len, para_len)
        attended_q=T.dot(para_matrix, norm_interaction_matrix_T.T)  #(hidden, para_len)*(para_len, q_len)=(hidden, q_len)
        return attended_para, attended_q
    (attended_para_reps, attended_q_reps), _ = theano.scan(fn=example_in_batch,
                                   outputs_info=None,
                                   sequences=[tensor1, tensor2])    #batch_q_reps (batch, hidden, para_len)
    return attended_para_reps, attended_q_reps

def cosine_row_wise_twoMatrix(M1, M2):
    #assume both (batch, hidden))
    dot=T.sum(M1*M2, axis=1) #(batch)
    norm1=T.sqrt(T.sum(M1**2, axis=1))
    norm2=T.sqrt(T.sum(M2**2, axis=1))
    return dot/(norm1*norm2)
def load_model_from_file(file_path, params):
    #save_file = open('/mounts/data/proj/wenpeng/Dataset/WikiQACorpus/Best_Conv_Para')
    save_file = open(file_path)
#     save_file = open('/mounts/data/proj/wenpeng/Dataset/WikiQACorpus/Best_Conv_Para_at_22')

    for para in params:
        para.set_value(cPickle.load(save_file), borrow=True)
    print 'model loaded successfully'
    save_file.close()
def store_model_to_file(file_path, best_params):
    save_file = open(file_path, 'wb')  # this will overwrite current contents
    for para in best_params:
        cPickle.dump(para.get_value(borrow=True), save_file, -1)  # the -1 is for HIGHEST_PROTOCOL
    save_file.close()

class HiddenLayer_with_Para(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
#         if W is None:
#             W_values = numpy.asarray(rng.uniform(
#                     low=-numpy.sqrt(6. / (n_in + n_out)),
#                     high=numpy.sqrt(6. / (n_in + n_out)),
#                     size=(n_in, n_out)), dtype=theano.config.floatX)  # @UndefinedVariable
#             if activation == theano.tensor.nnet.sigmoid:
#                 W_values *= 4
#
#             W = theano.shared(value=W_values, name='W', borrow=True)
#
#         if b is None:
#             b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)  # @UndefinedVariable
#             b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W     #for original one
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None
                       else activation(lin_output))
def Adam(cost, params, lr=0.001, b1=0.1, b2=0.001, e=1e-8):
    updates = []
    grads = T.grad(cost, params)
    i = theano.shared(numpy.float32(0.))
    i_t = i + 1.
    fix1 = 1. - (1. - b1)**i_t
    fix2 = 1. - (1. - b2)**i_t
    lr_t = lr * (T.sqrt(fix2) / fix1)
    for p, g in zip(params, grads):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (T.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))
    return updates

class rmsprop(object):
    """
    RMSProp with nesterov momentum and gradient rescaling
    """
    def __init__(self, params):
        self.running_square_ = [theano.shared(np.zeros_like(p.get_value(), dtype=theano.config.floatX))
                                for p in params]
        self.running_avg_ = [theano.shared(np.zeros_like(p.get_value(), dtype=theano.config.floatX))
                             for p in params]
        self.memory_ = [theano.shared(np.zeros_like(p.get_value(), dtype=theano.config.floatX))
                        for p in params]

    def updates(self, params, grads, learning_rate, momentum, rescale=5.):
        grad_norm = T.sqrt(sum(map(lambda x: T.sqr(x).sum(), grads)))
        not_finite = T.or_(T.isnan(grad_norm), T.isinf(grad_norm))
        grad_norm = T.sqrt(grad_norm)
        scaling_num = rescale
        scaling_den = T.maximum(rescale, grad_norm)
        # Magic constants
        combination_coeff = 0.95
        minimum_grad = 1e-4
        updates = []
        for n, (param, grad) in enumerate(zip(params, grads)):
            grad = T.switch(not_finite, 0.1 * param,
                            grad * (scaling_num / scaling_den))
            old_square = self.running_square_[n]
            new_square = combination_coeff * old_square + (
                1. - combination_coeff) * T.sqr(grad)
#             print 'new_square.type:', new_square.type
            old_avg = self.running_avg_[n]
            new_avg = combination_coeff * old_avg + (
                1. - combination_coeff) * grad
#             print 'new_avg.type:', new_avg.type
            rms_grad = T.sqrt(new_square - new_avg ** 2)
#             print 'rms_grad_0.type', rms_grad.type
            rms_grad = T.maximum(rms_grad, minimum_grad)
#             print 'rms_grad.type', rms_grad.type
            memory = self.memory_[n]
            half1=momentum * memory
#             print 'half1.type', half1.type
#             print 'grad.type', grad.type
            half2=learning_rate * grad / rms_grad
#             print 'half2.type', half2.type
            update = half1 - half2
#             print 'update.type', update.type
            update2 = momentum * momentum * memory - (np.float32(1.0) + momentum) * learning_rate * grad / rms_grad
#             print 'update2.type:', update2.type
            updates.append((old_square, new_square))
            updates.append((old_avg, new_avg))
            updates.append((memory, update))
            updates.append((param, param + update2))
        return updates

def rescale_weights(params, incoming_max):
    incoming_max = np.cast[theano.config.floatX](incoming_max)
    for p in params:
        w = p.get_value()
        w_sum = (w**2).sum(axis=0)
        w[:, w_sum>incoming_max] = w[:, w_sum>incoming_max] * np.sqrt(incoming_max) / w_sum[w_sum>incoming_max]
        p.set_value(w)



class BNComposite(Composite):
    init_param = ('dtype',)

    def __init__(self, dtype):
        self.dtype = dtype
        x = theano.scalar.Scalar(dtype=dtype).make_variable()
        mean = theano.scalar.Scalar(dtype=dtype).make_variable()
        std = theano.scalar.Scalar(dtype=dtype).make_variable()
        gamma = theano.scalar.Scalar(dtype=dtype).make_variable()
        beta = theano.scalar.Scalar(dtype=dtype).make_variable()
        o = add(mul(true_div(sub(x, mean), std), gamma), beta)
        inputs = [x, mean, std, gamma, beta]
        outputs = [o]
        super(BNComposite, self).__init__(inputs, outputs)

    def grad(self, inps, grads):
        x, mean, std, gamma, beta = inps
        top, = grads
        dx = (top * gamma) / std
        dmean = -(top * gamma) / std
        dstd = -(top * gamma * (x - mean)) / (std * std)
        dgamma = top * (x - mean) / std
        return [dx, dmean, dstd, dgamma, top]
def batch_normalization(inputs, gamma, beta, mean, std,
                        mode='low_mem'):
    """
    This function will build the symbolic graph for applying batch normalization
    to a set of activations.
    Also works on GPUs
    .. versionadded:: 0.7.1
    Parameters
    ----------
    inputs : symbolic tensor
        Mini-batch of activations
    gamma: symbolic tensor
        BN scale parameter, must be of same dimensionality as
        inputs and broadcastable against it
    beta: symbolic tensor
        BN shift parameter, must be of same dimensionality as
        inputs and broadcastable against it
    mean: symbolic tensor
        inputs means, must be of same dimensionality as
        inputs and broadcastable against it
    std: symbolic tensor
        inputs standard deviation, must be of same dimensionality as
        inputs and broadcastable against it
    mode: 'low_mem' or 'high_mem'
        Specify which batch_normalization implementation that will be
        used.
        As no intermediate representations are stored for the back-propagation,
        'low_mem' implementation lower the memory usage, however,
        it is 5-10% slower than 'high_mem' implementation. Note that 5-10% computation
        time difference compare the batch_normalization operation only, time difference
        between implementation is likely to be less important on the full model fprop/bprop.
    """
    if mode == 'low_mem':
        elm_bn = theano.tensor.elemwise.Elemwise(scalar_op=BNComposite(dtype=inputs.dtype))
        rval = elm_bn(inputs, mean, std, gamma, beta)
    elif mode == 'high_mem':
        rval = (inputs - mean) * (gamma / std) + beta
    else:
        raise ValueError(
            'mode must be either "low_mem", "high_mem"')
    return rval

def shared(shape, name):
    """
    Create a shared object of a numpy array.
    """
    if len(shape) == 1:
        value = np.zeros(shape)  # bias are initialized with zeros
    else:
        drange = np.sqrt(6. / (np.sum(shape)))
        value = drange * np.random.uniform(low=-1.0, high=1.0, size=shape)
    return theano.shared(value=value.astype(theano.config.floatX), name=name)

def crf_forward(observations, transitions, viterbi=False,
            return_alpha=False, return_best_sequence=False):
    """
    Takes as input:
        - observations, sequence of shape (n_steps, n_classes)
        - transitions, sequence of shape (n_classes, n_classes)
    Probabilities must be given in the log space.
    Compute alpha, matrix of size (n_steps, n_classes), such that
    alpha[i, j] represents one of these 2 values:
        - the probability that the real path at node i ends in j
        - the maximum probability of a path finishing in j at node i (Viterbi)
    Returns one of these 2 values:
        - alpha
        - the final probability, which can be:
            - the sum of the probabilities of all paths
            - the probability of the best path (Viterbi)
    """
    assert not return_best_sequence or (viterbi and not return_alpha)

    def recurrence(obs, previous, transitions):
        previous = previous.dimshuffle(0, 'x')
        obs = obs.dimshuffle('x', 0)
        if viterbi:
            scores = previous + obs + transitions
            out = scores.max(axis=0)
            if return_best_sequence:
                out2 = scores.argmax(axis=0)
                return out, out2
            else:
                return out
        else:
            return log_sum_exp(previous + obs + transitions, axis=0)

    initial = observations[0]
    alpha, _ = theano.scan(
        fn=recurrence,
        outputs_info=(initial, None) if return_best_sequence else initial,
        sequences=[observations[1:]],
        non_sequences=transitions
    )

    if return_alpha:
        return alpha
    elif return_best_sequence:
        sequence, _ = theano.scan(
            fn=lambda beta_i, previous: beta_i[previous],
            outputs_info=T.cast(T.argmax(alpha[0][-1]), 'int32'),
            sequences=T.cast(alpha[1][::-1], 'int32')
        )
        sequence = T.concatenate([sequence[::-1], [T.argmax(alpha[0][-1])]])
        return sequence
    else:
        if viterbi:
            return alpha[-1].max(axis=0)
        else:
            return log_sum_exp(alpha[-1], axis=0)
def log_sum_exp(x, axis=None):
    """
    Sum probabilities in the log-space.
    """
    xmax = x.max(axis=axis, keepdims=True)
    xmax_ = x.max(axis=axis)
    return xmax_ + T.log(T.exp(x - xmax).sum(axis=axis))

def squad_cnn_rank_spans(rng, common_input_p, common_input_q, char_common_input_p, char_common_input_q,batch_size, p_len_limit,q_len_limit,
                         emb_size, char_emb_size,char_len,filter_size,char_filter_size,hidden_size,
                         conv_W_1, conv_b_1,conv_W_2, conv_b_2, conv_W_1_q, conv_b_1_q,conv_W_2_q, conv_b_2_q, conv_W_char,conv_b_char,
                         para_mask, q_mask, char_p_masks,char_q_masks):
    conv_input_p_char = char_common_input_p.dimshuffle(0,2,1)       #(batch_size, emb_size, maxsenlen)
    conv_model_p_char = Conv_with_Mask(rng, input_tensor3=conv_input_p_char,
             mask_matrix = char_p_masks,
             image_shape=(batch_size*p_len_limit, 1, char_emb_size, char_len),
             filter_shape=(char_emb_size, 1, char_emb_size, char_filter_size), W=conv_W_char, b=conv_b_char)
    p_word_char_reps = conv_model_p_char.maxpool_vec.reshape((batch_size, p_len_limit, char_emb_size)).dimshuffle(0,2,1) #(batch_size, char_emb_size,*q_len)

    conv_input_q_char = char_common_input_q.dimshuffle(0,2,1)       #(batch_size, 1, emb_size, maxsenlen+width-1)
    conv_model_q_char = Conv_with_Mask(rng, input_tensor3=conv_input_q_char,
             mask_matrix = char_q_masks,
             image_shape=(batch_size*q_len_limit, 1, char_emb_size, char_len),
             filter_shape=(char_emb_size, 1, char_emb_size, char_filter_size), W=conv_W_char, b=conv_b_char)
    q_word_char_reps = conv_model_q_char.maxpool_vec.reshape((batch_size, q_len_limit, char_emb_size)).dimshuffle(0,2,1) #(batch_size, char_emb_size,*q_len)

    conv_input_p_1 = T.concatenate([common_input_p.dimshuffle((0,2,1)), p_word_char_reps], axis=1)

    conv_model_p_1 = Conv_with_Mask(rng, input_tensor3=conv_input_p_1,
             mask_matrix = para_mask,
             image_shape=(batch_size, 1, emb_size+char_emb_size, p_len_limit),
             filter_shape=(hidden_size, 1, emb_size+char_emb_size, filter_size[0]), W=conv_W_1, b=conv_b_1)
    conv_output_p_1=conv_model_p_1.masked_conv_out #(batch, hidden, len)

    conv_input_q_1 = T.concatenate([common_input_q.dimshuffle((0,2,1)), q_word_char_reps], axis=1)
    conv_model_q_1 = Conv_with_Mask(rng, input_tensor3=conv_input_q_1,
             mask_matrix = q_mask,
             image_shape=(batch_size, 1, emb_size+char_emb_size, q_len_limit),
             filter_shape=(hidden_size, 1, emb_size+char_emb_size, filter_size[0]), W=conv_W_1_q, b=conv_b_1_q)
    conv_output_q_1=conv_model_q_1.masked_conv_out #(batch, 1, hidden_size, maxsenlen)

    #the second layer
    conv_model_p_2 = Conv_with_Mask(rng, input_tensor3=conv_output_p_1,
             mask_matrix = para_mask,
             image_shape=(batch_size, 1, hidden_size, p_len_limit),
             filter_shape=(hidden_size, 1, hidden_size, filter_size[1]), W=conv_W_2, b=conv_b_2)
    conv_output_p_tensor3=conv_model_p_2.masked_conv_out

    conv_model_q_2 = Conv_with_Mask(rng, input_tensor3=conv_output_q_1,
             mask_matrix = q_mask,
             image_shape=(batch_size, 1, hidden_size, q_len_limit),
             filter_shape=(hidden_size, 1, hidden_size, filter_size[1]), W=conv_W_2_q, b=conv_b_2_q)
    q_rep=conv_model_q_2.maxpool_vec #(batch_size, hidden_size) # each sentence then have an embedding of length hidden_size

    p2loop_matrix = conv_output_p_tensor3.reshape((conv_output_p_tensor3.shape[0]*conv_output_p_tensor3.shape[1], conv_output_p_tensor3.shape[2]))#(batch* hidden_size, maxsenlen)
    gram_1 = p2loop_matrix
    gram_2 = T.max(T.concatenate([p2loop_matrix[:,:-1].dimshuffle('x',0,1), p2loop_matrix[:,1:].dimshuffle('x',0,1)], axis=0), axis=0) #(batch* hidden_size, maxsenlen-1)
    gram_3 = T.max(T.concatenate([p2loop_matrix[:,:-2].dimshuffle('x',0,1), p2loop_matrix[:,1:-1].dimshuffle('x',0,1),p2loop_matrix[:,2:].dimshuffle('x',0,1)], axis=0), axis=0) #(batch* hidden_size, maxsenlen-2)
    gram_4 = T.max(T.concatenate([p2loop_matrix[:,:-3].dimshuffle('x',0,1), p2loop_matrix[:,1:-2].dimshuffle('x',0,1),p2loop_matrix[:,2:-1].dimshuffle('x',0,1),p2loop_matrix[:,3:].dimshuffle('x',0,1)], axis=0), axis=0) #(batch* hidden_size, maxsenlen-3)
    gram_5 = T.max(T.concatenate([p2loop_matrix[:,:-4].dimshuffle('x',0,1), p2loop_matrix[:,1:-3].dimshuffle('x',0,1),p2loop_matrix[:,2:-2].dimshuffle('x',0,1),p2loop_matrix[:,3:-1].dimshuffle('x',0,1),p2loop_matrix[:,4:].dimshuffle('x',0,1)], axis=0), axis=0) #(batch* hidden_size, maxsenlen-4)
    gram_size = 5*p_len_limit-(0+1+2+3+4)
    span_reps=T.concatenate([gram_1, gram_2,gram_3,gram_4,gram_5], axis=1).reshape((batch_size, hidden_size, gram_size)) #(batch, hidden_size, maxsenlen-(0+1+2+3+4))
    input4score = T.concatenate([span_reps, T.repeat(q_rep.dimshuffle(0,1,'x'), gram_size, axis=2)], axis=1) #(batch, 2*hidden, 5*p_len_limit-(0+1+2+3+4))

    return input4score

def squad_cnn_rank_word(rng, common_input_p, common_input_q, char_common_input_p, char_common_input_q,batch_size, p_len_limit,q_len_limit,
                         emb_size, char_emb_size,char_len,filter_size,char_filter_size,hidden_size,
                         conv_W_1, conv_b_1,conv_W_2, conv_b_2, conv_W_1_q, conv_b_1_q,conv_W_2_q, conv_b_2_q, conv_W_char,conv_b_char,
                         para_mask, q_mask, char_p_masks,char_q_masks):
    conv_input_p_char = char_common_input_p.dimshuffle(0,2,1)       #(batch_size, emb_size, maxsenlen)
    conv_model_p_char = Conv_with_Mask(rng, input_tensor3=conv_input_p_char,
             mask_matrix = char_p_masks,
             image_shape=(batch_size*p_len_limit, 1, char_emb_size, char_len),
             filter_shape=(char_emb_size, 1, char_emb_size, char_filter_size), W=conv_W_char, b=conv_b_char)
    p_word_char_reps = conv_model_p_char.maxpool_vec.reshape((batch_size, p_len_limit, char_emb_size)).dimshuffle(0,2,1) #(batch_size, char_emb_size,*q_len)

    conv_input_q_char = char_common_input_q.dimshuffle(0,2,1)       #(batch_size, 1, emb_size, maxsenlen+width-1)
    conv_model_q_char = Conv_with_Mask(rng, input_tensor3=conv_input_q_char,
             mask_matrix = char_q_masks,
             image_shape=(batch_size*q_len_limit, 1, char_emb_size, char_len),
             filter_shape=(char_emb_size, 1, char_emb_size, char_filter_size), W=conv_W_char, b=conv_b_char)
    q_word_char_reps = conv_model_q_char.maxpool_vec.reshape((batch_size, q_len_limit, char_emb_size)).dimshuffle(0,2,1) #(batch_size, char_emb_size,*q_len)

    conv_input_p_1 = T.concatenate([common_input_p.dimshuffle((0,2,1)), p_word_char_reps], axis=1)

    conv_model_p_1 = Conv_with_Mask(rng, input_tensor3=conv_input_p_1,
             mask_matrix = para_mask,
             image_shape=(batch_size, 1, emb_size+char_emb_size, p_len_limit),
             filter_shape=(hidden_size, 1, emb_size+char_emb_size, filter_size[0]), W=conv_W_1, b=conv_b_1)
    conv_output_p_1=conv_model_p_1.masked_conv_out #(batch, hidden, len)

    conv_input_q_1 = T.concatenate([common_input_q.dimshuffle((0,2,1)), q_word_char_reps], axis=1)
    conv_model_q_1 = Conv_with_Mask(rng, input_tensor3=conv_input_q_1,
             mask_matrix = q_mask,
             image_shape=(batch_size, 1, emb_size+char_emb_size, q_len_limit),
             filter_shape=(hidden_size, 1, emb_size+char_emb_size, filter_size[0]), W=conv_W_1_q, b=conv_b_1_q)
    conv_output_q_1=conv_model_q_1.masked_conv_out #(batch, 1, hidden_size, maxsenlen)

    #the second layer
    conv_model_p_2 = Conv_with_Mask(rng, input_tensor3=conv_output_p_1,
             mask_matrix = para_mask,
             image_shape=(batch_size, 1, hidden_size, p_len_limit),
             filter_shape=(hidden_size, 1, hidden_size, filter_size[1]), W=conv_W_2, b=conv_b_2)
    conv_output_p_tensor3=T.concatenate([conv_model_p_2.masked_conv_out, conv_output_p_1, conv_input_p_1], axis=1) #(batch, 3*hidden+char_emb)

    conv_model_q_2 = Conv_with_Mask(rng, input_tensor3=conv_output_q_1,
             mask_matrix = q_mask,
             image_shape=(batch_size, 1, hidden_size, q_len_limit),
             filter_shape=(hidden_size, 1, hidden_size, filter_size[1]), W=conv_W_2_q, b=conv_b_2_q)
    q_rep=conv_model_q_2.maxpool_vec + conv_model_q_1.maxpool_vec #(batch_size, hidden_size) # each sentence then have an embedding of length hidden_size

    #a_variable_casted = T.cast(a_variable, 'int32')
    q_head_indices = T.cast(-T.sum(q_mask, axis=1),'int32') #(batch)
    q_rep_tensor3 = conv_output_q_1+conv_model_q_2.masked_conv_out
    q_heads = q_rep_tensor3[T.arange(batch_size),:,q_head_indices]  #(batch, hidden)
    q_tails = q_rep_tensor3[:,:,-1]  #(batch, hidden)

    q_rep_head_tail = T.concatenate([q_rep,q_heads,q_tails], axis=1) #(batch, 3*hidden)

    input4score = T.concatenate([conv_output_p_tensor3, T.repeat(q_rep_head_tail.dimshuffle(0,1,'x'), p_len_limit, axis=2)], axis=1) #(batch, 6*hidden+char_emb, p_len_limit)

    return input4score*para_mask.dimshuffle(0,'x',1)

def squad_cnn_rank_spans_word(rng, common_input_p, common_input_q, char_common_input_p, char_common_input_q,batch_size, p_len_limit,q_len_limit,
                         emb_size, char_emb_size,char_len,filter_size,char_filter_size,hidden_size,
                         conv_W_1, conv_b_1,conv_W_2, conv_b_2, conv_W_1_q, conv_b_1_q,conv_W_2_q, conv_b_2_q, conv_W_char,conv_b_char,
                         conv_W_3, conv_b_3, conv_W_3_q, conv_b_3_q,
#                          conv_W_4, conv_b_4, conv_W_4_q, conv_b_4_q,
#                          conv_W_5, conv_b_5, conv_W_5_q, conv_b_5_q,
                         para_mask, q_mask, char_p_masks,char_q_masks,
                         extra, extra_size):
    conv_input_p_char = char_common_input_p.dimshuffle(0,2,1)       #(batch_size, emb_size, maxsenlen)
    conv_model_p_char = Conv_with_Mask(rng, input_tensor3=conv_input_p_char,
             mask_matrix = char_p_masks,
             image_shape=(batch_size*p_len_limit, 1, char_emb_size, char_len),
             filter_shape=(char_emb_size, 1, char_emb_size, char_filter_size), W=conv_W_char, b=conv_b_char)
    p_word_char_reps = conv_model_p_char.maxpool_vec.reshape((batch_size, p_len_limit, char_emb_size)).dimshuffle(0,2,1) #(batch_size, char_emb_size,*q_len)

    conv_input_q_char = char_common_input_q.dimshuffle(0,2,1)       #(batch_size, 1, emb_size, maxsenlen+width-1)
    conv_model_q_char = Conv_with_Mask(rng, input_tensor3=conv_input_q_char,
             mask_matrix = char_q_masks,
             image_shape=(batch_size*q_len_limit, 1, char_emb_size, char_len),
             filter_shape=(char_emb_size, 1, char_emb_size, char_filter_size), W=conv_W_char, b=conv_b_char)
    q_word_char_reps = conv_model_q_char.maxpool_vec.reshape((batch_size, q_len_limit, char_emb_size)).dimshuffle(0,2,1) #(batch_size, char_emb_size,*q_len)
    
    p_init_input = common_input_p.dimshuffle((0,2,1))  #(batch, emb_size, p_len)
    conv_input_p_1 = T.concatenate([p_init_input, p_word_char_reps, extra.dimshuffle(0,2,1)], axis=1)
#     zero_pad = T.zeros((batch_size,emb_size,1))
#     left_word_emb = T.concatenate([zero_pad, p_init_input[:,:,:-1]], axis=2)#(batch, emb_size, p_len)
#     right_word_emb = T.concatenate([p_init_input[:,:,1:], zero_pad], axis=2)#(batch, emb_size, p_len)
    
    
    
    
    conv_model_p_1 = Conv_with_Mask(rng, input_tensor3=conv_input_p_1,
             mask_matrix = para_mask,
             image_shape=(batch_size, 1, emb_size+char_emb_size+extra_size, p_len_limit),
             filter_shape=(hidden_size, 1, emb_size+char_emb_size+extra_size, filter_size[0]), W=conv_W_1, b=conv_b_1)
    conv_output_p_1=conv_model_p_1.masked_conv_out #(batch, hidden, len)

    conv_input_q_1 = T.concatenate([common_input_q.dimshuffle((0,2,1)), q_word_char_reps], axis=1)
    conv_model_q_1 = Conv_with_Mask(rng, input_tensor3=conv_input_q_1,
             mask_matrix = q_mask,
             image_shape=(batch_size, 1, emb_size+char_emb_size, q_len_limit),
             filter_shape=(hidden_size, 1, emb_size+char_emb_size, filter_size[0]), W=conv_W_1_q, b=conv_b_1_q)
    conv_output_q_1=conv_model_q_1.masked_conv_out #(batch, 1, hidden_size, maxsenlen)

    #the second layer
    conv_model_p_2 = Conv_with_Mask(rng, input_tensor3=conv_output_p_1,
             mask_matrix = para_mask,
             image_shape=(batch_size, 1, hidden_size, p_len_limit),
             filter_shape=(hidden_size, 1, hidden_size, filter_size[1]), W=conv_W_2, b=conv_b_2)
    conv_output_p_2=conv_model_p_2.masked_conv_out# + conv_output_p_1

    conv_model_q_2 = Conv_with_Mask(rng, input_tensor3=conv_output_q_1,
             mask_matrix = q_mask,
             image_shape=(batch_size, 1, hidden_size, q_len_limit),
             filter_shape=(hidden_size, 1, hidden_size, filter_size[1]), W=conv_W_2_q, b=conv_b_2_q)
    conv_output_q_2=conv_model_q_2.masked_conv_out#(batch_size, hidden_size) # each sentence then have an embedding of length hidden_size

    #the third layer
    conv_model_p_3 = Conv_with_Mask(rng, input_tensor3=conv_output_p_2,
             mask_matrix = para_mask,
             image_shape=(batch_size, 1, hidden_size, p_len_limit),
             filter_shape=(hidden_size, 1, hidden_size, filter_size[2]), W=conv_W_3, b=conv_b_3)
#     conv_output_p_3=conv_model_p_3.masked_conv_out# + conv_output_p_1 + conv_output_p_2
    conv_output_p_tensor3=T.concatenate([conv_model_p_3.masked_conv_out,conv_output_p_1,conv_output_p_2],axis=1) #(batch, 3*hidden, p_len)

    conv_model_q_3 = Conv_with_Mask(rng, input_tensor3=conv_output_q_2,
             mask_matrix = q_mask,
             image_shape=(batch_size, 1, hidden_size, q_len_limit),
             filter_shape=(hidden_size, 1, hidden_size, filter_size[2]), W=conv_W_3_q, b=conv_b_3_q)
    q_rep=conv_model_q_3.maxpool_vec + conv_model_q_1.maxpool_vec+ conv_model_q_2.maxpool_vec#(batch_size, hidden_size) # each sentence then have an embedding of length hidden_size
#     conv_output_q_3=conv_model_q_3.masked_conv_out

    #the fourth layer
#     conv_model_p_4 = Conv_with_Mask(rng, input_tensor3=conv_output_p_3,
#              mask_matrix = para_mask,
#              image_shape=(batch_size, 1, hidden_size, p_len_limit),
#              filter_shape=(hidden_size, 1, hidden_size, filter_size[3]), W=conv_W_4, b=conv_b_4)
#     conv_output_p_4=conv_model_p_4.masked_conv_out# + conv_output_p_1 + conv_output_p_2
# 
#     conv_model_q_4 = Conv_with_Mask(rng, input_tensor3=conv_output_q_3,
#              mask_matrix = q_mask,
#              image_shape=(batch_size, 1, hidden_size, q_len_limit),
#              filter_shape=(hidden_size, 1, hidden_size, filter_size[3]), W=conv_W_4_q, b=conv_b_4_q)
#     conv_output_q_4=conv_model_q_4.masked_conv_out

    #the 5th layer
#     conv_model_p_5 = Conv_with_Mask(rng, input_tensor3=conv_output_p_4,
#              mask_matrix = para_mask,
#              image_shape=(batch_size, 1, hidden_size, p_len_limit),
#              filter_shape=(hidden_size, 1, hidden_size, filter_size[4]), W=conv_W_5, b=conv_b_5)
#     conv_output_p_tensor3=T.concatenate([conv_model_p_5.masked_conv_out,conv_output_p_1,conv_output_p_2,conv_output_p_3, conv_output_p_4],axis=1) #(batch, 5*hidden, p_len)
# 
#     conv_model_q_5 = Conv_with_Mask(rng, input_tensor3=conv_output_q_4,
#              mask_matrix = q_mask,
#              image_shape=(batch_size, 1, hidden_size, q_len_limit),
#              filter_shape=(hidden_size, 1, hidden_size, filter_size[4]), W=conv_W_5_q, b=conv_b_5_q)
#     q_rep=conv_model_q_5.maxpool_vec + conv_model_q_1.maxpool_vec+ conv_model_q_2.maxpool_vec + conv_model_q_3.maxpool_vec+ conv_model_q_4.maxpool_vec#(batch_size, hidden_size) # each sentence then have an embedding of length hidden_size


    #a_variable_casted = T.cast(a_variable, 'int32')
    q_head_indices = T.cast(-T.sum(q_mask, axis=1),'int32') #(batch)
    q_rep_tensor3 = conv_output_q_1+conv_output_q_2+conv_model_q_3.masked_conv_out#+conv_output_q_4+conv_model_q_5.masked_conv_out  #(batch, hidden, q_len)
    q_heads = T.concatenate([q_rep_tensor3[T.arange(batch_size),:,q_head_indices], conv_input_q_1[T.arange(batch_size),:,q_head_indices],conv_input_q_1[T.arange(batch_size),:,q_head_indices+1],conv_input_q_1[T.arange(batch_size),:,q_head_indices+2]], axis=1)  #(batch, hidden)
    q_tails = q_rep_tensor3[:,:,-1]  #(batch, hidden)

    q_rep_head_tail = T.concatenate([q_rep,q_heads,q_tails], axis=1) #(batch, 3*hidden)
    q_hidden_size = 3*hidden_size+3*(emb_size+char_emb_size)

    conv_output_p_tensor3 = T.concatenate([conv_output_p_tensor3, conv_input_p_1], axis=1) #(batch, 3*hidden+1*emb+char_emb+extra_size, p_len)
    p_hidden_size = 3*hidden_size+1*emb_size+char_emb_size+extra_size

    p2loop_matrix = conv_output_p_tensor3.reshape((conv_output_p_tensor3.shape[0]*conv_output_p_tensor3.shape[1], conv_output_p_tensor3.shape[2]))#(batch* hidden_size, maxsenlen)
    gram_1 = p2loop_matrix
    concate_2 = T.concatenate([p2loop_matrix[:,:-1].dimshuffle('x',0,1), p2loop_matrix[:,1:].dimshuffle('x',0,1)], axis=0)
    gram_2 = T.max(concate_2, axis=0) #(batch* hidden_size, maxsenlen-1)
    concate_3 = T.concatenate([p2loop_matrix[:,:-2].dimshuffle('x',0,1), p2loop_matrix[:,1:-1].dimshuffle('x',0,1),p2loop_matrix[:,2:].dimshuffle('x',0,1)], axis=0)
    gram_3 = T.max(concate_3, axis=0) #(batch* hidden_size, maxsenlen-2)
    concate_4 = T.concatenate([p2loop_matrix[:,:-3].dimshuffle('x',0,1), p2loop_matrix[:,1:-2].dimshuffle('x',0,1),p2loop_matrix[:,2:-1].dimshuffle('x',0,1),p2loop_matrix[:,3:].dimshuffle('x',0,1)], axis=0)
    gram_4 = T.max(concate_4, axis=0) #(batch* hidden_size, maxsenlen-3)
    concate_5 = T.concatenate([p2loop_matrix[:,:-4].dimshuffle('x',0,1), p2loop_matrix[:,1:-3].dimshuffle('x',0,1),p2loop_matrix[:,2:-2].dimshuffle('x',0,1),p2loop_matrix[:,3:-1].dimshuffle('x',0,1),p2loop_matrix[:,4:].dimshuffle('x',0,1)], axis=0)
    gram_5 = T.max(concate_5, axis=0) #(batch* hidden_size, maxsenlen-4)

    gram_1_max = gram_1.reshape((batch_size, p_hidden_size, p_len_limit))
    gram_1_left = gram_1_max
    gram_1_right = gram_1_max
    gram_1_sum = gram_1_max
    gram_1_comb = T.concatenate([gram_1_max,gram_1_left,gram_1_right, gram_1_sum], axis=1) #(batch, 4*p_hidden_size, p_len)

    gram_2_max = gram_2.reshape((batch_size, p_hidden_size, p_len_limit-1))
    gram_2_left = p2loop_matrix[:,:-1].reshape((batch_size, p_hidden_size, p_len_limit-1))
    gram_2_right = p2loop_matrix[:,1:].reshape((batch_size, p_hidden_size, p_len_limit-1))
    gram_2_sum = gram_2_left + gram_2_right
    gram_2_comb = T.concatenate([gram_2_max,gram_2_left,gram_2_right, gram_2_sum], axis=1) #(batch, 4*p_hidden_size, p_len-1)

    gram_3_max = gram_3.reshape((batch_size, p_hidden_size, p_len_limit-2))
    gram_3_left = p2loop_matrix[:,:-2].reshape((batch_size, p_hidden_size, p_len_limit-2))
    gram_3_right = p2loop_matrix[:,2:].reshape((batch_size, p_hidden_size, p_len_limit-2))
    gram_3_sum = T.sum(concate_3, axis=0).reshape((batch_size, p_hidden_size, p_len_limit-2))
    gram_3_comb = T.concatenate([gram_3_max,gram_3_left,gram_3_right,gram_3_sum], axis=1) #(batch, 4*p_hidden_size, p_len-1)

    gram_4_max = gram_4.reshape((batch_size, p_hidden_size, p_len_limit-3))
    gram_4_left = p2loop_matrix[:,:-3].reshape((batch_size, p_hidden_size, p_len_limit-3))
    gram_4_right = p2loop_matrix[:,3:].reshape((batch_size, p_hidden_size, p_len_limit-3))
    gram_4_sum = T.sum(concate_4, axis=0).reshape((batch_size, p_hidden_size, p_len_limit-3))
    gram_4_comb = T.concatenate([gram_4_max,gram_4_left,gram_4_right,gram_4_sum], axis=1) #(batch, 4*p_hidden_size, p_len-1)

    gram_5_max = gram_5.reshape((batch_size, p_hidden_size, p_len_limit-4))
    gram_5_left = p2loop_matrix[:,:-4].reshape((batch_size, p_hidden_size, p_len_limit-4))
    gram_5_right = p2loop_matrix[:,4:].reshape((batch_size, p_hidden_size, p_len_limit-4))
    gram_5_sum = T.sum(concate_5, axis=0).reshape((batch_size, p_hidden_size, p_len_limit-4))
    gram_5_comb = T.concatenate([gram_5_max,gram_5_left,gram_5_right,gram_5_sum], axis=1) #(batch, 4*p_hidden_size, p_len-1)

    gram_size = 5*p_len_limit-(0+1+2+3+4)
    span_reps=T.concatenate([gram_1_comb, gram_2_comb,gram_3_comb,gram_4_comb,gram_5_comb], axis=2).reshape((batch_size, 4*p_hidden_size, gram_size)) #(batch, hidden_size, maxsenlen-(0+1+2+3+4))
    span_input4score = T.concatenate([span_reps, T.repeat(q_rep_head_tail.dimshuffle(0,1,'x'), gram_size, axis=2)], axis=1) #(batch, 4*p_hidden_size+3*hidden, 5*p_len_limit-(0+1+2+3+4))

    word_input4score = T.concatenate([conv_output_p_tensor3, T.repeat(q_rep_head_tail.dimshuffle(0,1,'x'), p_len_limit, axis=2)], axis=1)*para_mask.dimshuffle(0,'x',1) #(batch, p_hidden_size+3*hidden, p_len_limit)
    
    overall_span_hidden_size = 4*p_hidden_size+q_hidden_size
    overall_word_hidden_size = p_hidden_size+q_hidden_size
    return span_input4score, word_input4score, overall_span_hidden_size,overall_word_hidden_size

def add_HLs_2_tensor3(input4score, HL_1_para,HL_2_para,HL_3_para,HL_4_para,norm_U_a, batch_size,true_p_len):
    HL_input = input4score.dimshuffle(0,2,1)  #(batch, p_len, input_size)
    HL_1_output = T.tanh(T.dot(HL_input, HL_1_para)) #(batch, p_len, hidden_size)
    HL_2_output = T.tanh(T.dot(HL_1_output, HL_2_para))
    HL_3_output = T.tanh(T.dot(HL_2_output+HL_1_output, HL_3_para))
    HL_4_output = T.tanh(T.dot(HL_3_output+HL_2_output+HL_1_output, HL_4_para))
    
    score_step_input = T.concatenate([HL_4_output+HL_3_output+HL_2_output+HL_1_output, HL_input], axis=2) #(batch, p_len, input_size+hidden_size)
    span_scores_matrix=T.dot(score_step_input, norm_U_a).reshape((batch_size, true_p_len))  #(batch, 13*para_len-78, 1)
    return span_scores_matrix #(batch, para_len)
