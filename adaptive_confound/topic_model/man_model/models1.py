import tensorflow as tf
class MlpFeatureExtractor(object):
    def __init__(self,
                network_architecture,
                dropout,
                learning_rate=0.001,
                batch_size=100,
                gpu_fraction=0.25):
        tf.reset_default_graph()
        self.network_architecture = network_architecture
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self._create_network()
        self._create_loss_optimizer()
        
        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.per_gpu_memory_fraction = gpu_fraction
        config.gpu_options.allocator_type = "BFC"
        self.sess = tf.Session(config=config)
        self.sess.run(init)
   
    def _initialize_weights(self,
                            n_hidden_1,
                            n_hidden_2,
                            n_input):
        all_weights = dict()
        