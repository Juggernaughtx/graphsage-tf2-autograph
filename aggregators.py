import tensorflow as tf

class MeanAggregator(tf.keras.layers.Layer):
    """
    Aggregates via mean followed by matmul and non-linearity.
    """
        
    def __init__(
        self,
        input_dim,
        output_dim,
        dropout=0.0,
        bias=False,
        act=tf.nn.relu,
        name=None,
        concat=False,
        **kwargs
    ):
        super().__init__()
        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        self.w1 = self.add_weight( name = "neigh_weights"
                                , shape = (input_dim, output_dim)
                                , dtype = tf.float32
                                , initializer = tf.keras.initializers.GlorotUniform
                                , trainable = True
                                )

        self.w2 = self.add_weight( name = "self_weights"
                                , shape = (input_dim, output_dim)
                                , dtype = tf.float32
                                , initializer = tf.keras.initializers.GlorotUniform
                                , trainable = True
                                )


    def call(self, inputs):
        self_vecs, neigh_vecs = inputs

        neigh_vecs = tf.nn.dropout(neigh_vecs, self.dropout)
        self_vecs = tf.nn.dropout(self_vecs, self.dropout)
        neigh_means = tf.math.reduce_mean(input_tensor=neigh_vecs, axis=1)

        # [nodes] x [out_dim]
        from_neighs = tf.linalg.matmul(neigh_means, self.w1)

        from_self = tf.linalg.matmul(self_vecs, self.w2)

        if not self.concat:
            output = tf.math.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)


        return self.act(output)