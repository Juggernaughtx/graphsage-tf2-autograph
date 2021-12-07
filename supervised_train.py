import os
import time
import tensorflow as tf
import numpy as np
import sklearn
from datetime import datetime
from sklearn import metrics
from itertools import islice
import math
import random

from .supervised_models import SupervisedGraphsage
from .utils import load_data

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


SAMPLE_SIZES = [25, 10]               # implicit number of layers
BATCH_SIZE = 512
EPOCHS = 10

os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

GPU_MEM_FRACTION = 0.8
    

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

     

def load_dummy_data():
    all_nodes = np.random.permutation(20000)
    train_nodes = all_nodes[:10000]
    val_nodes = all_nodes[10000:15000]
    test_nodes = all_nodes[15000:20000]
    adj = all_nodes[np.random.randint(0, len(all_nodes), (len(all_nodes), 25))]
    feats = np.random.permutation(400000).reshape(len(all_nodes), 20)
    feats = np.array(feats, dtype=np.float64)
    labels = np.random.randint(41, size=len(all_nodes))
    num_classes = 41
    return feats, train_nodes, val_nodes, test_nodes, adj, labels, num_classes, timestamps


def calc_f1(y_true, y_pred):
    y_pred = np.argmax(y_pred, axis=1)
    return (
        metrics.f1_score(y_true, y_pred, average="micro"),
        metrics.f1_score(y_true, y_pred, average="macro"),
    )

        
def incremental_evaluate(nodes, id_map, model, class_map, size):
    t_test = time.time()
    val_preds = []
    labels = []
    iter_num = 0
    while iter_num < math.floor(len(nodes)/size):
        inputs = nodes[iter_num*size:(iter_num+1)*size]
        batch_labels = class_map[np.array(inputs)]
        val_logits, h, c = model(inputs, training=False)
        val_preds = np.vstack(val_logits)
        labels = np.vstack(batch_labels)
        iter_num += 1
    f1_scores = calc_f1(labels, val_preds)
    return f1_scores[0], f1_scores[1], (time.time() - t_test)


class minibatch_func(tf.keras.utils.Sequence):

    def __init__(self, x_set, y_set, batch_size, sample_sizes, train=True):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.sizes = sample_sizes
        self.train = train

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)


    def __getitem__(self, idx):
        if self.train:
            logging.info("\ntrain idx " + str(idx))
        else:
            logging.info("\nval idx " + str(idx))

        batch = np.array(self.x[idx*self.batch_size:(idx+1)*self.batch_size])
        batch_labels = all_labels[batch]

        return batch, batch_labels


def fit_train():
    print("Loading training data..")
    feats, train_nodes, val_nodes, test_nodes, class_map, adj = load_data()
    num_classes = len(set(class_map))
    print("Done loading training data..")

    feats = np.vstack([feats, np.zeros((feats.shape[1],))])    
    #feats, train_nodes, val_nodes, test_nodes, adj, labels, num_classes, timestamps = load_dummy_data()
    
    feats1 = tf.convert_to_tensor(feats, dtype=tf.float32)
    model = SupervisedGraphsage(
            feats1,
            adj,
            SAMPLE_SIZES,
            num_classes,
            BATCH_SIZE,
        )
    
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=acc_metric)
    
    train_gen = minibatch_func(train_nodes, class_map, BATCH_SIZE, adj, SAMPLE_SIZES, train=True)
    val_gen = minibatch_func(val_nodes, class_map, VAL_BATCH_SIZE, adj, SAMPLE_SIZES, train=False)

    
    model.fit(train_gen, epochs=10, shuffle=False, workers=1, validation_data=val_gen)
    print("Model trained!")


    test_f1_mic, test_f1_mac, duration = incremental_evaluate(test_nodes, model, class_map, BATCH_SIZE)
    print(
    "Full test stats:",
    "f1_micro=",
    "{:.5f}".format(test_f1_mic),
    "f1_macro=",
    "{:.5f}".format(test_f1_mac),
    "time=",
    "{:.5f}".format(duration),
    )

if __name__ == "__main__":
    train()