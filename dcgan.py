import tensorflow as tf

class DCGAN:
    def __init__(self,
                 batch_size=128, f_size=4, z_dim=100,
                 gdepth1=1024, gdepth2=512, gdepth3=256, gdepth4=128,
                 ddepth1=64,   ddepth2=128, ddepth3=256, ddepth4=512):
        self.batch_size = batch_size
        self.f_size = f_size
        self.z_dim = z_dim
        self.g = self.__generator(gdepth1, gdepth2, gdepth3, gdepth4)
        self.d = self.__discriminator(ddepth1, ddepth2, ddepth3, ddepth4)
        self.z = tf.random_uniform([self.batch_size, self.z_dim], minval=-1.0, maxval=1.0)

    def __generator(self, depth1, depth2, depth3, depth4):
        reuse = False
        def model(inputs):
            nonlocal reuse
            depths = [depth1, depth2, depth3, depth4, 3]
            i_depth = depths[0:4]
            o_depth = depths[1:5]
            with tf.variable_scope('g', reuse=reuse):
                # reshape from inputs
                with tf.variable_scope('reshape'):
                    w0 = tf.get_variable('weights', [self.z_dim, i_depth[0] * self.f_size * self.f_size], tf.float32, tf.truncated_normal_initializer(stddev=0.02))
                    b0 = tf.get_variable('biases', [i_depth[0]], tf.float32, tf.zeros_initializer)
                    dc0 = tf.nn.bias_add(tf.reshape(tf.matmul(inputs, w0), [-1, self.f_size, self.f_size, i_depth[0]]), b0)
                    bn0 = tf.contrib.layers.batch_norm(dc0)
                    out = tf.nn.relu(bn0)
                # deconvolution (transpose of convolution) layers
                for i in range(4):
                    with tf.variable_scope('conv%d' % (i + 1)):
                        w = tf.get_variable('weights', [5, 5, o_depth[i], i_depth[i]], tf.float32, tf.truncated_normal_initializer(stddev=0.02))
                        b = tf.get_variable('biases', [o_depth[i]], tf.float32, tf.zeros_initializer)
                        dc = tf.nn.conv2d_transpose(out, w, [self.batch_size, self.f_size * 2 ** (i + 1), self.f_size * 2 ** (i + 1), o_depth[i]], [1, 2, 2, 1])
                        out = tf.nn.bias_add(dc, b)
                        if i < 3:
                            out = tf.nn.relu(tf.contrib.layers.batch_norm(out))
            reuse = True
            return tf.nn.tanh(out)
        return model

    def __discriminator(self, depth1, depth2, depth3, depth4):
        reuse = False
        def model(inputs):
            nonlocal reuse
            depths = [3, depth1, depth2, depth3, depth4]
            i_depth = depths[0:4]
            o_depth = depths[1:5]
            out = []
            with tf.variable_scope('d', reuse=reuse):
                outputs = inputs
                # convolution layer
                for i in range(4):
                    with tf.variable_scope('conv%d' % i):
                        w = tf.get_variable('weights', [5, 5, i_depth[i], o_depth[i]], tf.float32, tf.truncated_normal_initializer(stddev=0.02))
                        b = tf.get_variable('biases', [o_depth[i]], tf.float32, tf.zeros_initializer)
                        c = tf.nn.bias_add(tf.nn.conv2d(outputs, w, [1, 2, 2, 1], padding='SAME'), b)
                        bn = tf.contrib.layers.batch_norm(c)
                        outputs = tf.maximum(0.2 * bn, bn)
                        out.append(outputs)
                # reshepe and fully connect to 2 classes
                with tf.variable_scope('classify'):
                    dim = 1
                    for d in outputs.get_shape()[1:].as_list():
                        dim *= d
                    w = tf.get_variable('weights', [dim, 2], tf.float32, tf.truncated_normal_initializer(stddev=0.02))
                    b = tf.get_variable('biases', [2], tf.float32, tf.zeros_initializer)
                    out.append(tf.nn.bias_add(tf.matmul(tf.reshape(outputs, [-1, dim]), w), b))
            reuse = True
            return out
        return model

    def train(self, input_images, learning_rate=0.0002, beta1=0.5, beta2=0.999,
              feature_matching=False):
        outputs_from_g = self.d(self.g(self.z))
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
        g_loss = tf.add_n(tf.get_collection('g_losses'), name='total_g_loss')
        d_loss = tf.add_n(tf.get_collection('d_losses'), name='total_d_loss')
        g_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2).minimize(g_loss, var_list=g_vars)
        d_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2).minimize(d_loss, var_list=d_vars)
        with tf.control_dependencies([g_optimizer, d_optimizer]):
            train_op = tf.no_op(name='train')
        return train_op, g_loss, d_loss

    def generate_images(self, row=8, col=8, inputs=None):
        if inputs is None:
            inputs = self.z
        images = tf.cast(tf.mul(tf.add(self.g(inputs), 1.0), 127.5), tf.uint8)
        images = [image for image in tf.split(0, self.batch_size, images)]
        rows = []
        for i in range(row):
            rows.append(tf.concat(2, images[col * i + 0:col * i + col]))
        image = tf.concat(1, rows)
        return tf.image.encode_jpeg(tf.squeeze(image, [0]))
