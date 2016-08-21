import tensorflow as tf

class Generator:
    def __init__(self, depths=[1024, 512, 256, 128], f_size=4):
        self.reuse = False
        self.f_size = f_size
        self.depths = depths + [3]

    def model(self, inputs):
        i_depth = self.depths[0:4]
        o_depth = self.depths[1:5]
        out = []
        with tf.variable_scope('g', reuse=self.reuse):
            # reshape from inputs
            with tf.variable_scope('reshape'):
                fc = tf.contrib.layers.fully_connected(inputs, i_depth[0] * self.f_size * self.f_size, normalizer_fn=tf.contrib.layers.batch_norm)
                outputs = tf.reshape(fc, [-1, self.f_size, self.f_size, i_depth[0]])
                out.append(outputs)
            # deconvolution (transpose of convolution) layers
            for i in range(4):
                with tf.variable_scope('conv%d' % (i + 1)):
                    activation_fn = tf.nn.relu if i < 3 else None
                    normalizer_fn = tf.contrib.layers.batch_norm if i < 3 else None
                    outputs = tf.contrib.layers.conv2d_transpose(outputs, o_depth[i], [5, 5], stride=2, activation_fn=activation_fn, normalizer_fn=normalizer_fn)
                    if i == 3:
                        outputs = tf.nn.tanh(outputs)
                    out.append(outputs)
        self.reuse = True
        return out

    def __call__(self, inputs):
        return self.model(inputs)

class Discriminator:
    def __init__(self, depths=[64, 128, 256, 512]):
        self.reuse = False
        self.depths = [3] + depths

    def model(self, inputs):
        def leaky_relu(x, leak=0.2):
            return tf.maximum(x, x * leak)
        i_depth = self.depths[0:4]
        o_depth = self.depths[1:5]
        out = []
        with tf.variable_scope('d', reuse=self.reuse):
            outputs = inputs
            # convolution layer
            for i in range(4):
                with tf.variable_scope('conv%d' % i):
                    outputs = tf.contrib.layers.conv2d(outputs, o_depth[i], [5, 5], stride=2, activation_fn=leaky_relu, normalizer_fn=tf.contrib.layers.batch_norm)
                    out.append(outputs)
            # reshepe and fully connect to 2 classes
            with tf.variable_scope('classify'):
                dim = 1
                for d in outputs.get_shape()[1:].as_list():
                    dim *= d
                w = tf.get_variable('weights', [dim, 2], tf.float32, tf.truncated_normal_initializer(stddev=0.02))
                b = tf.get_variable('biases', [2], tf.float32, tf.zeros_initializer)
                out.append(tf.nn.bias_add(tf.matmul(tf.reshape(outputs, [-1, dim]), w), b))
        self.reuse = True
        return out

    def __call__(self, inputs):
        return self.model(inputs)

class DCGAN:
    def __init__(self,
                 batch_size=128, f_size=4, z_dim=100,
                 gdepth1=1024, gdepth2=512, gdepth3=256, gdepth4=128,
                 ddepth1=64,   ddepth2=128, ddepth3=256, ddepth4=512):
        self.batch_size = batch_size
        self.f_size = f_size
        self.z_dim = z_dim
        self.g = Generator(depths=[gdepth1, gdepth2, gdepth3, gdepth4], f_size=self.f_size)
        self.d = Discriminator(depths=[ddepth1, ddepth2, ddepth3, ddepth4])
        self.z = tf.random_uniform([self.batch_size, self.z_dim], minval=-1.0, maxval=1.0)

    def loss(self, input_images, feature_matching=False):
        outputs_from_g = self.d(self.g(self.z)[-1])
        outputs_from_i = self.d(input_images)
        logits_from_g = outputs_from_g[-1]
        logits_from_i = outputs_from_i[-1]
        if feature_matching:
            features_from_g = tf.reduce_mean(outputs_from_g[-2], reduction_indices=(0))
            features_from_i = tf.reduce_mean(outputs_from_i[-2], reduction_indices=(0))
            tf.add_to_collection('g_losses', tf.mul(tf.nn.l2_loss(features_from_g - features_from_i), 1e-2))
        g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g')
        d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='d')
        tf.add_to_collection('g_losses', tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits_from_g, tf.ones([self.batch_size], dtype=tf.int64))))
        tf.add_to_collection('d_losses', tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits_from_i, tf.ones([self.batch_size], dtype=tf.int64))))
        tf.add_to_collection('d_losses', tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits_from_g, tf.zeros([self.batch_size], dtype=tf.int64))))
        return {
            'g': tf.add_n(tf.get_collection('g_losses'), name='total_g_loss'),
            'd': tf.add_n(tf.get_collection('d_losses'), name='total_d_loss')
        }

    def sample_images(self, row=8, col=8, inputs=None):
        if inputs is None:
            inputs = self.z
        images = tf.cast(tf.mul(tf.add(self.g(inputs)[-1], 1.0), 127.5), tf.uint8)
        images = [image for image in tf.split(0, self.batch_size, images)]
        rows = []
        for i in range(row):
            rows.append(tf.concat(2, images[col * i + 0:col * i + col]))
        image = tf.concat(1, rows)
        return tf.image.encode_jpeg(tf.squeeze(image, [0]))
