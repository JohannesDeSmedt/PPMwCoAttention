import numpy as np
from operator import itemgetter

from sklearn.model_selection import train_test_split as tts
from tensorflow.keras.preprocessing import sequence


class ActivityRecord:

    def __init__(self, a1, timestamp, trace_no, event, encoding):
        self.a1 = a1
        self.timestamp = timestamp
        self.trace_no = trace_no
        self.event = event
        self.encoding = encoding

    def __str__(self):
        return self.a1 + ' at ' + str(self.timestamp) + ' in trace ' + str(self.trace_no)

    def __gt__(self, other):
        if self.timestamp > other.timestamp:
            return True
        else:
            return False


def create_case_log_prefix_traces_new(log, no_activities, min_length, prefix_length, prefix_model, shuffle, random_state):
    activity_records = []
    event_map = {}
    act_map = {}
    reverse_map = {}

    # loop through the traces and store activity mappings (activity -> number)
    # store activity records which are sortable activity/timestamp/trace/event tuples
    for t, trace in enumerate(log):
        for e, event in enumerate(trace):
            if event['concept:name'] not in act_map.keys():
                act_map[event['concept:name']] = len(act_map)
                reverse_map[act_map[event['concept:name']]] = event['concept:name']

            zeros = np.zeros(no_activities)
            zeros[act_map[event['concept:name']]] = 1
            ar = ActivityRecord(act_map[event['concept:name']], event['time:timestamp'], t, event, zeros)

            event_map[(t, e)] = len(activity_records)
            activity_records.append(ar)

    # sort all events based on timestamp and store their original indices in the log
    indices, sorted_activity_records = zip(*sorted(enumerate(activity_records), key=itemgetter(1)))

    sorted_encoded_events = [act_rec.encoding for act_rec in sorted_activity_records]
    old_new_map = {old_ind: new_ind for new_ind, old_ind in enumerate(indices)}

    enc_trace_prefixes = []
    enc_process_prefixes = []
    enc_labels = []

    # create prefixes of both log and trace
    for t, trace in enumerate(log):
        enc_trace_prefix = []

        for e, event in enumerate(trace):
            # skip first activities in trace
            if e >= len(trace) - 1 or e == 0:
                continue

            act_no = act_map[event['concept:name']]
            zeros = np.zeros(no_activities)
            zeros[act_no] = 1
            enc_trace_prefix.append(zeros)

            # if minimum prefix lenght is reached, create log and trace prefix
            if len(enc_trace_prefix) > min_length:
                # add trace prefix - this is trivial
                enc_trace_prefixes.append(enc_trace_prefix[-(prefix_length+1):-1])

                # look up original index of event in the log
                original_index_of_event = event_map[(t, e)]
                index_of_event = old_new_map[original_index_of_event]

                # use index to create log prefix
                ind_fin = max(0, (index_of_event + 1 - prefix_model))
                enc_process_prefixes.append(sorted_encoded_events[ind_fin:index_of_event + 1])

                # store the label of the next activity prediction with the prefix
                zeros = np.zeros(len(act_map))
                zeros[act_map[trace[e + 1]['concept:name']]] = 1
                enc_labels.append(zeros)

    # pad sequences with low number as a mask
    X_train_t, X_test_t, X_train_p, X_test_p, y_train_ohe, y_test_ohe = \
        tts(enc_trace_prefixes, enc_process_prefixes, enc_labels, test_size=0.3, shuffle=shuffle, random_state=random_state)
    X_train_t = sequence.pad_sequences(X_train_t, maxlen=prefix_length, value=-100000000)
    X_train_p = sequence.pad_sequences(X_train_p, maxlen=prefix_model, value=-100000000)
    X_test_t = sequence.pad_sequences(X_test_t, maxlen=prefix_length, value=-100000000)
    X_test_p = sequence.pad_sequences(X_test_p, maxlen=prefix_model, value=-100000000)

    # convert all data to NumPy arrays
    X_train_t = np.array(X_train_t)
    X_test_t = np.array(X_test_t)
    X_train_p = np.array(X_train_p)
    X_test_p = np.array(X_test_p)
    y_train_ohe = np.array(y_train_ohe)
    y_test_ohe = np.array(y_test_ohe)

    return X_train_t, X_test_t, X_train_p, X_test_p, y_train_ohe, y_test_ohe, reverse_map

