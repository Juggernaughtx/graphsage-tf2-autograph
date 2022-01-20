import tensorflow as tf

from .models import (Sample, Aggregate)
from .aggregators import (
    MeanAggregator,
)


class SupervisedGraphsage(tf.keras.Model):
    """Implementation of supervised GraphSAGE."""

    def __init__(
        self,
        features,
        adj_info,
        layer_infos,
        num_classes,
        concat=True,
        hidden_dim=128,
        **kwargs
    ):
        super().__init__()
            
        dims = [features.shape[1]]
        dims.extend([hidden_dim for i in range(len(layer_infos))])

        dim_mult = 2 if concat else 1
        self.sample_layer = Sample(adj_info, layer_infos)

        self.aggregate_layer = Aggregate(features, dims, layer_infos)

        self.node_pred = tf.keras.layers.Dense(num_classes)

        

    def call(self, inputs):

        x = self.sample_layer(inputs)

        x = self.aggregate_layer(x)

        x = tf.math.l2_normalize(x, 1)
        
        x = self.node_pred(x)
        
        x = tf.nn.softmax(x)

        return x