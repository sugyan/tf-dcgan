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
            inputs = tf.convert_to_tensor(inputs)
            with tf.variable_scope('fc_reshape'):
                w0 = tf.get_variable('w', [inputs.get_shape()[-1], i_depth[0] * self.f_size * self.f_size], tf.float32, tf.truncated_normal_initializer(stddev=0.02))
                b0 = tf.get_variable('b', [i_depth[0]], tf.float32, tf.zeros_initializer)
                fc = tf.matmul(inputs, w0)
                reshaped = tf.reshape(fc, [-1, self.f_size, self.f_size, i_depth[0]])
                mean, variance = tf.nn.moments(reshaped, [0, 1, 2])
                outputs = tf.nn.relu(tf.nn.batch_normalization(reshaped, mean, variance, b0, None, 1e-5))
                out.append(outputs)
            # deconvolution (transpose of convolution) x 4
            for i in range(4):
                with tf.variable_scope('conv%d' % (i + 1)):
                    w = tf.get_variable('w', [5, 5, o_depth[i], i_depth[i]], tf.float32, tf.truncated_normal_initializer(stddev=0.02))
                    b = tf.get_variable('b', [o_depth[i]], tf.float32, tf.zeros_initializer)
                    dc = tf.nn.conv2d_transpose(outputs, w, [int(outputs.get_shape()[0]), self.f_size * 2 ** (i + 1), self.f_size * 2 ** (i + 1), o_depth[i]], [1, 2, 2, 1])
                    if i < 3:
                        mean, variance = tf.nn.moments(dc, [0, 1, 2])
                        outputs = tf.nn.relu(tf.nn.batch_normalization(dc, mean, variance, b, None, 1e-5))
                    else:
                        outputs = tf.nn.tanh(tf.nn.bias_add(dc, b))
                    out.append(outputs)
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g')
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
            # convolution x 4
            for i in range(4):
                with tf.variable_scope('conv%d' % i):
                    w = tf.get_variable('w', [5, 5, i_depth[i], o_depth[i]], tf.float32, tf.truncated_normal_initializer(stddev=0.02))
                    b = tf.get_variable('b', [o_depth[i]], tf.float32, tf.zeros_initializer)
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
                b = tf.get_variable('b', [2], tf.float32, tf.zeros_initializer)
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
        self.losses = {
            'g': None,
            'd': None
        }

    def build(self, input_images,
              learning_rate=0.0002, beta1=0.5, feature_matching=0.0):
        """build model, generate losses, train op"""
        generated_images = self.g(self.z)[-1]
        outputs_from_g = self.d(generated_images)
        outputs_from_i = self.d(input_images)
        logits_from_g = outputs_from_g[-1]
        logits_from_i = outputs_from_i[-1]
        if feature_matching > 0.0:
            mean_image_from_g = tf.reduce_mean(generated_images, reduction_indices=(0))
            mean_image_from_i = tf.reduce_mean(input_images, reduction_indices=(0))
            tf.add_to_collection('g_losses', tf.mul(tf.nn.l2_loss(mean_image_from_g - mean_image_from_i), feature_matching))
            features_from_g = tf.reduce_mean(outputs_from_g[-2], reduction_indices=(0))
            features_from_i = tf.reduce_mean(outputs_from_i[-2], reduction_indices=(0))
            tf.add_to_collection('g_losses', tf.mul(tf.nn.l2_loss(features_from_g - features_from_i), feature_matching))
        tf.add_to_collection('g_losses', tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits_from_g, tf.ones([self.batch_size], dtype=tf.int64))))
        tf.add_to_collection('d_losses', tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits_from_i, tf.ones([self.batch_size], dtype=tf.int64))))
        tf.add_to_collection('d_losses', tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits_from_g, tf.zeros([self.batch_size], dtype=tf.int64))))

        self.losses['g'] = tf.add_n(tf.get_collection('g_losses'), name='total_g_loss')
        self.losses['d'] = tf.add_n(tf.get_collection('d_losses'), name='total_d_loss')

        g_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(self.losses['g'], var_list=self.g.variables)
        d_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(self.losses['d'], var_list=self.d.variables)
        with tf.control_dependencies([g_opt, d_opt]):
            self.train = tf.no_op(name='train')
        return self.train

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
