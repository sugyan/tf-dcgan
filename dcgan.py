import tensorflow as tf


class Generator:
    def __init__(self, depths=[1024, 512, 256, 128], f_size=4):
        self.reuse = False
        self.f_size = f_size
        self.depths = depths + [3]

    def model(self, inputs):
        inputs = tf.convert_to_tensor(inputs)
        with tf.variable_scope('g', reuse=self.reuse):
            # reshape from inputs
            with tf.variable_scope('reshape'):
                outputs = tf.layers.dense(inputs, self.depths[0] * self.f_size * self.f_size)
                outputs = tf.reshape(outputs, [-1, self.f_size, self.f_size, self.depths[0]])
                outputs = tf.layers.batch_normalization(tf.nn.relu(outputs))
                outputs = tf.identity(outputs, name='outputs')
            # deconvolution (transpose of convolution) x 4
            with tf.variable_scope('deconv1'):
                outputs = tf.layers.conv2d_transpose(outputs, self.depths[1], [5, 5], strides=(2, 2), padding='SAME')
                outputs = tf.layers.batch_normalization(tf.nn.relu(outputs))
                outputs = tf.identity(outputs, name='outputs')
            with tf.variable_scope('deconv2'):
                outputs = tf.layers.conv2d_transpose(outputs, self.depths[2], [5, 5], strides=(2, 2), padding='SAME')
                outputs = tf.layers.batch_normalization(tf.nn.relu(outputs))
                outputs = tf.identity(outputs, name='outputs')
            with tf.variable_scope('deconv3'):
                outputs = tf.layers.conv2d_transpose(outputs, self.depths[3], [5, 5], strides=(2, 2), padding='SAME')
                outputs = tf.layers.batch_normalization(tf.nn.relu(outputs))
                outputs = tf.identity(outputs, name='outputs')
            with tf.variable_scope('deconv4') as scope:
                outputs = tf.layers.conv2d_transpose(outputs, self.depths[4], [5, 5], strides=(2, 2), padding='SAME')
                outputs = tf.layers.batch_normalization(tf.nn.relu(outputs))
                outputs = tf.identity(outputs, name=scope.name)
            # output images
            with tf.variable_scope('tanh'):
                outputs = tf.tanh(outputs, name='outputs')
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g')
        return outputs

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
            # convolution x 4
            for i in range(4):
                with tf.variable_scope('conv%d' % i):
                    w = tf.get_variable(
                        'w',
                        [5, 5, i_depth[i], o_depth[i]],
                        tf.float32,
                        tf.truncated_normal_initializer(stddev=0.02))
                    b = tf.get_variable(
                        'b',
                        [o_depth[i]],
                        tf.float32,
                        tf.zeros_initializer())
                    c = tf.nn.conv2d(outputs, w, [1, 2, 2, 1], 'SAME')
                    mean, variance = tf.nn.moments(c, [0, 1, 2])
                    outputs = leaky_relu(tf.nn.batch_normalization(c, mean, variance, b, None, 1e-5))
                    out.append(outputs)
            # reshepe and fully connect to 2 classes
            with tf.variable_scope('classify'):
                dim = 1
                for d in outputs.get_shape()[1:].as_list():
                    dim *= d
                w = tf.get_variable('w', [dim, 2], tf.float32, tf.truncated_normal_initializer(stddev=0.02))
                b = tf.get_variable('b', [2], tf.float32, tf.zeros_initializer())
                out.append(tf.nn.bias_add(tf.matmul(tf.reshape(outputs, [-1, dim]), w), b))
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='d')
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
        # self.losses = {
        #     'g': None,
        #     'd': None
        # }

    def losses(self, inputs, learning_rate=0.0002, beta1=0.5):
        """build models, generate train op.

        Args:
            input_images: 4-D Tensor of shape `[batch, height, width, channels]`.

        Returns:
            operator for training models.
        """
        images = self.g(self.z)
        outputs_from_g = self.d(images)
        outputs_from_i = self.d(inputs)
        logits_from_g = outputs_from_g[-1]
        logits_from_i = outputs_from_i[-1]
        # losses
        tf.add_to_collection(
            'g_losses',
            tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.ones([self.batch_size], dtype=tf.int64),
                    logits=logits_from_g)))
        tf.add_to_collection(
            'd_losses',
            tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.ones([self.batch_size], dtype=tf.int64),
                    logits=logits_from_i)))
        tf.add_to_collection(
            'd_losses',
            tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.zeros([self.batch_size], dtype=tf.int64),
                    logits=logits_from_g)))
        # self.losses['g'] = tf.add_n(tf.get_collection('g_losses'), name='total_g_loss')
        # self.losses['d'] = tf.add_n(tf.get_collection('d_losses'), name='total_d_loss')
        # g_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
        # d_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
        # g_opt_op = g_opt.minimize(self.losses['g'], var_list=self.g.variables)
        # d_opt_op = d_opt.minimize(self.losses['d'], var_list=self.d.variables)
        # with tf.control_dependencies([g_opt_op, d_opt_op]):
        #     self.train = tf.no_op(name='train')
        # return self.train

    def sample_images(self, row=8, col=8, inputs=None):
        if inputs is None:
            inputs = self.z
        images = tf.cast(tf.multiply(tf.add(self.g(inputs)[-1], 1.0), 127.5), tf.uint8)
        images = [image for image in tf.split(images, self.batch_size, axis=0)]
        rows = []
        for i in range(row):
            rows.append(tf.concat(images[col * i + 0:col * i + col], 2))
        image = tf.concat(rows, 1)
        return tf.image.encode_jpeg(tf.squeeze(image, [0]))
