import numpy as np
import pm4py

from pm4py.objects.log.importer.xes import importer as xes_importer

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, Input
from tensorflow.keras.layers import Flatten, Masking
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.optimizers import Nadam
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score

from create_inputs import create_case_log_prefix_traces_new


class MFB_reverse(Layer):

    def __init__(self, output_dim, **kwargs):
        super(MFB_reverse, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.supports_masking = True
        self.k = 5
        self.g = 4

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

        return final_embedding, gamma_v, gamma_q, F


def dual_lstm_MFB(lstm_dim, prefix_trace, prefix_log, attention_dim=4):
    trace_input = Input(shape=(prefix_trace, no_act), name='trace_input')
    masked_trace_input = Masking(mask_value=-100000000, input_shape=(prefix_trace, no_act))(trace_input)

    l_t_1 = LSTM(lstm_dim, kernel_initializer='glorot_uniform', return_sequences=True, name='trace_lstm_1')(masked_trace_input)
    l_t_2 = LSTM(lstm_dim, kernel_initializer='glorot_uniform', return_sequences=True, name='trace_lstm_2')(l_t_1)

    prefix_input = Input(shape=(prefix_log, no_act), name='process_input')
    masked_process_input = Masking(mask_value=-100000000, input_shape=(prefix_log, no_act))(prefix_input)

    l_p_1 = LSTM(lstm_dim, kernel_initializer='glorot_uniform', return_sequences=True, name='process_lstm_1')(masked_process_input)
    l_p_2 = LSTM(lstm_dim, kernel_initializer='glorot_uniform', return_sequences=True, name='process_lstm_2')(l_p_1)

    final_emb, gamma2_v, gamma2_q, F = MFB_reverse(attention_dim)([l_t_2, l_p_2])
    flat = Flatten()(final_emb)

    act_output = Dense(no_act, activation='softmax', kernel_initializer='glorot_uniform', name='act_output')(flat)

    model = Model(inputs=[trace_input, prefix_input], outputs=[act_output])
    model_2 = Model(inputs=[trace_input, prefix_input],
                    outputs=[act_output, gamma2_v, gamma2_q])

    return model, model_2


# should mirror the name of a xes file, e.g., 'bpi12.xes'
dataset = 'sepsis'
event_log_file = f'../datasets/{dataset}.xes'
variant = xes_importer.Variants.ITERPARSE
paras = {variant.value.Parameters.MAX_TRACES: 200000}
log = xes_importer.apply(event_log_file, parameters=paras)
activity_names = pm4py.get_event_attribute_values(log, 'concept:name')
no_act = len(activity_names)

# t_t
prefix_length = 5
# t_l
prefix_log = 10
# minimum prefix length
min_length = 2

X_train_t, X_test_t, X_train_p, X_test_p, y_train_ohe, y_test_ohe, reverse_map = \
    create_case_log_prefix_traces_new(log, no_activities=no_act, min_length=min_length,
                                      prefix_length=prefix_length, prefix_model=prefix_log,
                                      shuffle=True, random_state=42)

# hyperparameters of the LSTM
epochs = 50
lstm_dim = 32
act_dim = 16

# model_2 stores the attention weights separate from model_1 which fits the data
opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004, clipvalue=3)
model_1, model_2 = dual_lstm_MFB(lstm_dim, prefix_length, prefix_log, attention_dim=act_dim)

model_1.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model_2.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

model_1.fit([X_train_t, X_train_p], y_train_ohe, validation_split=0.2, epochs=epochs)

predictions = model_1.predict([X_test_t, X_test_p], verbose=1)
y_class = y_test_ohe.argmax(axis=-1)
y_pred = predictions.argmax(axis=-1)

# the visualisation picks random test instances, this can be changed to your liking to include
# different events
random_indices = np.random.randint(0, len(X_test_p), 100)
# gamma_t and gamma_l contain the attention scores
predictions_2, gamma_t, gamma_l = model_2.predict([X_test_t[random_indices], X_test_p[random_indices]], verbose=1)

accuracy = accuracy_score(y_class, y_pred)
precision = precision_score(y_class, y_pred, average='macro')
print("Accuracy test set: %.2f%%" % (accuracy * 100))
print("Precision test set: %.2f%%" % (precision * 100))

for i in range(len(random_indices)):
    label = reverse_map[np.argmax(y_test_ohe[random_indices[i]])]
    pred = reverse_map[np.argmax(predictions_2[i])]

    # convert one-hot encoded traces to activity label sequences
    convert_trace = []
    convert_log = []

    for l in range(0, prefix_length):
        trace = X_test_t[random_indices[i], l]
        if np.sum(trace) < 0:
            convert_trace.append('PADDING_'+str(l))
        else:
            convert_trace.append(reverse_map[np.argmax(trace)]+"_"+str(l))

    for l in range(0, prefix_log):
        model = X_test_p[random_indices[i], l]
        if np.sum(model) < 0:
            convert_log.append('PADDING_'+str(l))
        else:
            convert_log.append(reverse_map[np.argmax(model)]+"_"+str(l))

    # normalize attention scores
    a_t = np.reshape(gamma_t[i], (len(gamma_t[i])))
    a_t = (a_t - np.min(a_t)) / (np.max(a_t) - np.min(a_t))
    a_l = np.reshape(gamma_l[i], (len(gamma_l[i])))
    a_l = (a_l - np.min(a_l)) / (np.max(a_l) - np.min(a_l))

    # visualize with seaborn
    sns.set(font_scale=1)
    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 10))
    sns.barplot(x=convert_trace, y=a_t, ax=ax1)
    ax1.axhline(0, color="k", clip_on=False)
    ax1.set_xticklabels(convert_trace, rotation=45, ha='right', rotation_mode='anchor')
    ax1.set_ylabel("Attention trace")
    sns.barplot(x=convert_log, y=a_l, ax=ax2)
    ax2.set_xticklabels(convert_log, rotation=45, ha='right', rotation_mode='anchor')
    ax2.axhline(0, color="k", clip_on=False)
    ax2.set_ylabel("Attention log")
    sns.despine(bottom=True)
    plt.title('Predicted label: ' + pred + ' vs. actual label: ' + label)
    plt.setp(f.axes, yticks=[])
    plt.tight_layout(h_pad=2)
    # either save or show
    plt.savefig(f'./img/{dataset}_attention_log_trace_step_'+str(i), bbox_inches='tight', dpi=300)
    # plt.show()

    # plt.figure()
    # g = sns.heatmap(F[i], yticklabels=convert_trace)
    # g.set_xticklabels(convert_model, rotation=45, ha='right', rotation_mode='anchor')
    # g.set_yticklabels(convert_trace)
    # plt.title('Prediction: ' + pred + ' vs label: ' + label)
    # g.set_xlabel('Trace attention')
    # g.set_ylabel('Process attention')
    # plt.savefig('./img/attention_correlation_step_' + str(i), bbox_inches='tight', dpi=300)
    # # plt.show()