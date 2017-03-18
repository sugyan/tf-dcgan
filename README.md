# TensorFlow implementation of DCGAN


### What is DCGAN ? ###

Deep Convolutional Generative Adversarial Networks

- http://arxiv.org/abs/1511.06434


#### Other implementations of DCGAN ####

- https://github.com/Newmu/dcgan_code (Theano)
- https://github.com/soumith/dcgan.torch (Torch)
- https://github.com/mattya/chainer-DCGAN (Chainer)
- https://github.com/carpedm20/DCGAN-tensorflow (TensorFlow)


### Prerequisites ###

- Python >= 2.7 or 3.5
 - TensorFlow >= 1.0


### Usage ###

#### Train ####

```python
dcgan = DCGAN()
train_images = <images batch>
losses = dcgan.loss(train_images)
train_op = dcgan.train(losses)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(FLAGS.max_steps):
        _, g_loss_value, d_loss_value = sess.run([train_op, losses[dcgan.g], losses[dcgan.d]])
    # save trained variables
```

#### Generate ####

```python
dcgan = DCGAN()
images = dcgan.sample_images()

with tf.Session() as sess:
    # restore trained variables

    generated = sess.run(images)
    with open('<filename>', 'wb') as f:
        f.write(generated)
```


### Example ###

- https://github.com/sugyan/face-generator
