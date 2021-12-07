import tensorflow as tf
import math
import numpy as np

from .aggregators import (
    MeanAggregator,
)


class Sample(tf.keras.layers.Layer):
    """
    Randomly samples 2 hop neighborhood according to layer_infos
    """

    def __init__(
        self,
        adj,
        layer_infos,
        max_degree=25,
        **kwargs
    ):

        self.adj_info = adj
        self.max_degree = max_degree
        self.layer_infos = layer_infos
        super().__init__()


    def call(self, inputs):
        
        def UniformSample(ids, num_samples):
            adj_lists = tf.nn.embedding_lookup(params=self.adj_info, ids=ids)
            adj_lists = tf.transpose(a=tf.random.shuffle(tf.transpose(a=adj_lists)))

            adj_lists = tf.slice(adj_lists, [0, 0], [-1, num_samples])
            return adj_lists
            
            
        samples = [inputs] 
        support_size = 1
        support_sizes = [support_size]              
        for k in range(len(self.layer_infos)):   
            t = len(self.layer_infos) - k - 1
            support_size *= self.layer_infos[t]
            node = DynamicSample(samples[k], self.layer_infos[t])
            samples.append(tf.reshape(node, [support_size * len(samples[0]),]))
            support_sizes.append(support_size)

        return samples, support_sizes



class Aggregate(tf.keras.layers.Layer):
    """ 
        At each layer, aggregate hidden representations of neighbors to compute the hidden representations 
        at next layer and returns he hidden representation at the final layer for all nodes in batch
    """

    def __init__(
        self,
        features,
        dims,
        num_samples,
        dropout=0.0,
        concat=True,
        aggregator_type="mean",
        model_size="small",
        name=None,
        **kwargs
    ):
        super().__init__()
        
        if aggregator_type == "mean":
            self.aggregator_cls = MeanAggregator

        self.features = features
        self.dims = dims
        self.concat = concat
        self.num_samples = num_samples
    
        self.aggregators = []
        for layer in range(len(num_samples)):
            dim_mult = 2 if self.concat and (layer != 0) else 1
            # aggregator at current layer
            if layer == len(num_samples) - 1:
                aggregator = self.aggregator_cls(
                    dim_mult * dims[layer],
                    dims[layer + 1],
                    act=lambda x: x,
                    dropout=dropout,
                    name=name,
                    concat=concat,
                    model_size=model_size,
                )
            else:
                aggregator = self.aggregator_cls(
                    dim_mult * dims[layer],     
                    dims[layer + 1],
                    dropout=dropout,
                    name=name,
                    concat=concat,
                    model_size=model_size,
                )
            self.aggregators.append(aggregator)

    
    def call(self, inputs):
        samples, support_sizes = inputs

        # length: number of layers + 1
        hidden = [
            tf.nn.embedding_lookup(params=self.features, ids=node_samples)
                                            for node_samples in samples]

        for layer in range(len(self.num_samples)):
            # hidden representation at current layer for all support nodes that are various hops away
            next_hidden = []
            aggregator = self.aggregators[layer]
            # as layer increases, the number of support nodes needed decreases
            for hop in range(len(self.num_samples) - layer):
                dim_mult = 2 if self.concat and (layer != 0) else 1
                neigh_dims = [
                    len(samples[0]) * support_sizes[hop],
                    self.num_samples[len(self.num_samples) - hop - 1],
                    dim_mult * self.dims[layer],
                ]

                h = aggregator((hidden[hop], tf.reshape(hidden[hop + 1], neigh_dims)))
                next_hidden.append(h)
            hidden = next_hidden
            
        return hidden[0]