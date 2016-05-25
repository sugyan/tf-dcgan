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

- Python >= 3.5
 - TensorFlow >= 0.8.0


### Usage ###

#### Train ####

```python
dcgan = DCGAN()
input_images = <images batch>
train_op, g_loss, d_loss = dcgan.train(input_images)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    for step in xrange(FLAGS.max_steps):
        start_time = time.time()
        _, g_loss_value, d_loss_value = sess.run([train_op, g_loss, d_loss])
        duration = time.time() - start_time
        format_str = 'step %d, loss = (G: %.8f, D: %.8f) (%.3f sec/batch)'
        print(format_str % (step, g_loss_value, d_loss_value, duration))
```

#### Generate ####

```python
dcgan = DCGAN()
images = dcgan.generate_images()

with tf.Session() as sess:
    # restore trained data

    generated = sess.run(images)
    with open(filename, 'wb') as f:
        f.write(generated)
```
