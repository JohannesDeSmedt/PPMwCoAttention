from tensorflow.keras.layers import Dense, BatchNormalization, TimeDistributed, Add, Reshape
from tensorflow.keras.layers import LSTM, Input, Activation, Multiply, Attention
from tensorflow.keras.layers import Conv2D, Concatenate, GlobalAveragePooling2D, Dropout, GlobalAveragePooling1D
from tensorflow.keras.layers import Flatten, Masking, Layer
from tensorflow.keras.models import Model
from tensorflow.python.ops import math_ops

from tensorflow.keras import backend as K
import tensorflow as tf


class MLB_flat(Layer):

    def __init__(self, output_dim, **kwargs):
        super(MLB_flat, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.supports_masking = True

    def build(self, input_shape):
        self.U = self.add_weight(name='U', shape=(input_shape[0][-1], self.output_dim),
                               initializer='random_normal', trainable=True)
        self.V = self.add_weight(name='V', shape=(input_shape[1][-1], self.output_dim),
                               initializer='random_normal', trainable=True)
        super(MLB_flat, self).build(input_shape)

    def call(self, x, mask=None):
        dot_1 = K.dot(x[0], self.U)
        dot_2 = K.dot(x[1], self.V)
        projection = dot_1 * dot_2
        return projection


class MFB_reverse(Layer):

    def __init__(self, output_dim, k_in, g_in, **kwargs):
        super(MFB_reverse, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.supports_masking = True
        self.k = k_in
        self.g = g_in

    def build(self, input_shape):
        self.W_R = self.add_weight(name='R_l', shape=(1, self.output_dim, input_shape[0][-1] ),
                               initializer='random_normal', trainable=True)
        self.W_Q = self.add_weight(name='Q_l', shape=(1, self.output_dim, input_shape[1][-1]),
                               initializer='random_normal', trainable=True)
        self.br = self.add_weight(name='attention_bias', shape=(self.output_dim, 1),
                               initializer='zeros', trainable=True)
        self.bq = self.add_weight(name='attention_bias', shape=(self.output_dim, 1),
                               initializer='zeros', trainable=True)

        self.U = self.add_weight(name='U', shape=(1, self.output_dim, input_shape[1][-2], self.k),
                                 initializer='random_normal', trainable=True)
        self.V = self.add_weight(name='V', shape=(1, self.output_dim, input_shape[0][-2], self.k),
                                 initializer='random_normal', trainable=True)

        self.W2_R = self.add_weight(name='W_R', shape=(1, self.g, self.output_dim),
                                 initializer='random_normal', trainable=True)
        self.W2_Q = self.add_weight(name='W_Q', shape=(1, self.g, self.output_dim),
                                 initializer='random_normal', trainable=True)
        self.w_mv = self.add_weight(name='w_g_r', shape=(self.g, 1),
                               initializer='random_normal', trainable=True)
        self.w_mq = self.add_weight(name='w_g_r', shape=(self.g, 1),
                               initializer='random_normal', trainable=True)
        super(MFB_reverse, self).build(input_shape)

    def call(self, x, mask=None):
        q_mask = mask[0] if mask else None
        v_mask = mask[1] if mask else None

        in_1 = tf.reshape(x[0], [-1, x[0].shape[2], x[0].shape[1]])
        in_2 = tf.reshape(x[1], [-1, x[1].shape[2], x[1].shape[1]])

        R_l = K.batch_dot(self.W_R, in_1)
        Q_l = K.batch_dot(self.W_Q, in_2)

        padding_mask = math_ops.logical_not(q_mask)
        output_list = []
        for i in range(R_l.shape[1]):
            ones_r = tf.fill((R_l.shape[2], 1), value=1.e9)
            ones_r = K.expand_dims(ones_r, 0)
            ones_r = K.squeeze(ones_r, -1)
            pad = math_ops.cast(padding_mask, dtype=K.floatx())
            pad = ones_r * pad
            output_list.append(pad)
        outputs = tf.stack(output_list, axis=1)

        R_l = tf.subtract(R_l, outputs)

        padding_mask = math_ops.logical_not(v_mask)
        output_list = []
        for i in range(Q_l.shape[1]):
            ones_r = tf.fill((Q_l.shape[2], 1), value=1.e9)
            ones_r = K.expand_dims(ones_r, 0)
            ones_r = K.squeeze(ones_r, -1)
            pad = math_ops.cast(padding_mask, dtype=K.floatx())
            pad = ones_r * pad
            output_list.append(pad)
        outputs = tf.stack(output_list, axis=1)

        Q_l = tf.subtract(Q_l, outputs)

        R_l = K.relu(R_l + self.br)
        Q_l = K.relu(Q_l + self.bq)


        F_sum = []
        for i in range(0, self.k):
            U_i = tf.reshape(self.U[:,:,:,i], [-1, self.U[:,:,:,i].shape[2], self.U[:,:,:,i].shape[1]])
            V_i = tf.reshape(self.V[:,:,:,i], [-1, self.V[:,:,:,i].shape[2], self.V[:,:,:,i].shape[1]])

            dot_1 = K.batch_dot(U_i, R_l)
            dot_2 = K.batch_dot(V_i, Q_l)
            transpose = tf.reshape(dot_1, [-1, dot_1.shape[2], dot_1.shape[1]])
            hada = transpose * dot_2
            F_sum.append(hada)

        F = F_sum[0]
        for i in range(1, len(F_sum)):
            F += F_sum[i]

        F_trans = tf.reshape(F, [-1, F.shape[2], F.shape[1]])

        WR = K.batch_dot(self.W2_R, R_l)
        WQ = K.batch_dot(self.W2_Q, Q_l)

        M_v = K.relu(WR + K.batch_dot(WQ, F_trans))
        M_q = K.relu(WQ + K.batch_dot(WR, F))

        dot_1 = K.batch_dot(K.transpose(self.w_mv), M_v)
        dot_2 = K.batch_dot(K.transpose(self.w_mq), M_q)

        padding_mask = math_ops.logical_not(q_mask)
        dot_1 = tf.subtract(dot_1, 1.e9 * math_ops.cast(padding_mask, dtype=K.floatx()))
        padding_mask = math_ops.logical_not(v_mask)
        dot_2 = tf.subtract(dot_2, 1.e9 * math_ops.cast(padding_mask, dtype=K.floatx()))

        gamma_v = K.softmax(dot_1)
        gamma_q = K.softmax(dot_2)

        R_l2 = tf.reshape(R_l, [-1, R_l.shape[2], R_l.shape[1]])
        Q_l2 = tf.reshape(Q_l, [-1, Q_l.shape[2], Q_l.shape[1]])
        trace_embedding = K.batch_dot(gamma_v, R_l2)
        log_embedding = K.batch_dot(gamma_q, Q_l2)

        final_embedding = trace_embedding * log_embedding

        return final_embedding


def prefix_plain_lstm(no_act, lstm_dim, length_input):
    trace_input = Input(shape=(length_input, no_act), name='trace_input')
    masked_trace_input = Masking(mask_value=-100000000, input_shape=(length_input, no_act))(trace_input)

    l_t_1 = LSTM(lstm_dim, kernel_initializer='glorot_uniform', return_sequences=True, name='trace_lstm_1')(masked_trace_input)
    l_t_2 = LSTM(lstm_dim, kernel_initializer='glorot_uniform', return_sequences=False, name='trace_lstm_2')(l_t_1)

    act_output = Dense(no_act, activation='softmax', kernel_initializer='glorot_uniform', name='act_output')(l_t_2)

    model = Model(inputs=[trace_input], outputs=[act_output])

    return model


def dual_lstm_concat(no_act, lstm_dim, length_input, prefix_model):
    trace_input = Input(shape=(length_input, no_act), name='trace_input')
    masked_trace_input = Masking(mask_value=-100000000, input_shape=(length_input, no_act))(trace_input)

    l_t_1 = LSTM(lstm_dim, kernel_initializer='glorot_uniform', return_sequences=True, name='trace_lstm_1')(masked_trace_input)
    l_t_2 = LSTM(lstm_dim, kernel_initializer='glorot_uniform', return_sequences=False, name='trace_lstm_2')(l_t_1)

    process_input = Input(shape=(prefix_model, no_act), name='process_input')
    masked_process_input = Masking(mask_value=-100000000, input_shape=(prefix_model, no_act))(process_input)

    l_p_1 = LSTM(lstm_dim, kernel_initializer='glorot_uniform', return_sequences=True, name='process_lstm_1')(masked_process_input)
    l_p_2 = LSTM(lstm_dim, kernel_initializer='glorot_uniform', return_sequences=False, name='process_lstm_2')(l_p_1)

    concat = Concatenate()([l_t_2, l_p_2])

    act_output = Dense(no_act, activation='softmax', kernel_initializer='glorot_uniform', name='act_output')(concat)

    model = Model(inputs=[trace_input, process_input], outputs=[act_output])

    return model


def dual_lstm_mul(no_act, lstm_dim, length_input, prefix_model):
    trace_input = Input(shape=(length_input, no_act), name='trace_input')
    masked_trace_input = Masking(mask_value=-100000000, input_shape=(length_input, no_act))(trace_input)

    l_t_1 = LSTM(lstm_dim, kernel_initializer='glorot_uniform', return_sequences=True, name='trace_lstm_1')(masked_trace_input)
    l_t_2 = LSTM(lstm_dim, kernel_initializer='glorot_uniform', return_sequences=False, name='trace_lstm_2')(l_t_1)

    process_input = Input(shape=(prefix_model, no_act), name='process_input')
    masked_process_input = Masking(mask_value=-100000000, input_shape=(prefix_model, no_act))(process_input)

    l_p_1 = LSTM(lstm_dim, kernel_initializer='glorot_uniform', return_sequences=True, name='process_lstm_1')(masked_process_input)
    l_p_2 = LSTM(lstm_dim, kernel_initializer='glorot_uniform', return_sequences=False, name='process_lstm_2')(l_p_1)

    concat = Multiply()([l_t_2, l_p_2])

    act_output = Dense(no_act, activation='softmax', kernel_initializer='glorot_uniform', name='act_output')(concat)

    model = Model(inputs=[trace_input, process_input], outputs=[act_output])

    return model


def dual_lstm_MLB(no_act, lstm_dim, length_input, prefix_model, attention_dim=8):
    trace_input = Input(shape=(length_input, no_act), name='trace_input')
    masked_trace_intput = Masking(mask_value=-100000000, input_shape=(length_input, no_act))(trace_input)

    l_t_1 = LSTM(lstm_dim, kernel_initializer='glorot_uniform', return_sequences=True, name='trace_lstm_1')(masked_trace_intput)
    l_t_2 = LSTM(lstm_dim, kernel_initializer='glorot_uniform', return_sequences=False, name='trace_lstm_2')(l_t_1)

    prefix_input = Input(shape=(prefix_model, no_act), name='process_input')
    masked_process_input = Masking(mask_value=-100000000, input_shape=(prefix_model, no_act))(prefix_input)

    l_p_1 = LSTM(lstm_dim, kernel_initializer='glorot_uniform', return_sequences=True, name='process_lstm_1')(masked_process_input)
    l_p_2 = LSTM(lstm_dim, kernel_initializer='glorot_uniform', return_sequences=False, name='process_lstm_2')(l_p_1)

    mlb_co_attention = MLB_flat(attention_dim)([l_t_2, l_p_2])
    final_emb = Activation('tanh')(mlb_co_attention)

    act_output = Dense(no_act, activation='softmax', kernel_initializer='glorot_uniform', name='act_output')(final_emb)

    model = Model(inputs=[trace_input, prefix_input], outputs=[act_output])

    return model


def dual_lstm_MFB(no_act, lstm_dim, length_input, prefix_model, k, g, attention_dim=8):
    trace_input = Input(shape=(length_input, no_act), name='trace_input')
    masked_trace_intput = Masking(mask_value=-100000000)(trace_input)

    l_t_1 = LSTM(lstm_dim, kernel_initializer='glorot_uniform', return_sequences=True, name='trace_lstm_1')(masked_trace_intput)
    l_t_2 = LSTM(lstm_dim, kernel_initializer='glorot_uniform', return_sequences=True, name='trace_lstm_2')(l_t_1)

    prefix_input = Input(shape=(prefix_model, no_act), name='process_input')
    masked_process_input = Masking(mask_value=-100000000)(prefix_input)

    l_p_1 = LSTM(lstm_dim, kernel_initializer='glorot_uniform', return_sequences=True, name='process_lstm_1')(masked_process_input)
    l_p_2 = LSTM(lstm_dim, kernel_initializer='glorot_uniform', return_sequences=True, name='process_lstm_2')(l_p_1)

    final_emb = MFB_reverse(attention_dim, k, g)([l_t_2, l_p_2])
    flat = Flatten()(final_emb)

    act_output = Dense(no_act, activation='softmax', kernel_initializer='glorot_uniform', name='act_output')(flat)

    model = Model(inputs=[trace_input, prefix_input], outputs=[act_output])

    return model
