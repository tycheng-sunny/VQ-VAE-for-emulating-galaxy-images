#===========================================================================================
# A test code using VQ-VAE on CANDELS images
# ** Check if hyper-parameters are the value/info you want.
#===========================================================================================
from __future__ import print_function

import time
import os
import pickle
#os.environ["CUDA_VISIBLE_DEVICES"]="0" # for GPU

#import matplotlib as mpl
#mpl.use('Agg') # to use matplotlib without visualisation envs
import matplotlib.pyplot as plt
import numpy as np
import sonnet as snt
import tensorflow as tf

from six.moves import xrange

#timer starts
Tstart = time.time()

### Data Preparation ###
## path
local_data_dir = 'data/'
train_data_fn = 'h_train.dict'
valid_data_fn = 'h_valid.dict'

## hyper-parameters for data
image_size = 84
channel_size = 1

## load data into Numpy
def unpickle(filename):
    with open(filename, 'rb') as fo:
        return pickle.load(fo, encoding='latin1')

def reshape_flattened_image_batch(flat_image_batch):
    return flat_image_batch.reshape(-1, image_size, image_size, channel_size)  # convert to NHWC

def combine_batches(batch_list):
    images = np.vstack([reshape_flattened_image_batch(batch_list['images'])])
    fn = np.vstack([np.array(batch_list['filename'])]).reshape(-1, 1)
    id = np.vstack([np.array(batch_list['id'])]).reshape(-1, 1)
    return {'images': images, 'filename': fn, 'id': id}

train_data_dict = combine_batches(unpickle(os.path.join(local_data_dir, train_data_fn)))
valid_data_dict = combine_batches(unpickle(os.path.join(local_data_dir, valid_data_fn)))

def cast_and_normalise_images(data_dict):
    """ The pixels in each image has been normalised to range [0., 1.] when generating .dict file """
    images = data_dict['images']
    data_dict['images'] = tf.cast(images, tf.float32)
    return data_dict

data_variance = np.var(train_data_dict['images']) # for the normalisation of the reconstruction loss

### Encoder & Decoder architecture ###
##residual
def residual_stack(h, num_hiddens, num_residual_layers, num_residual_hiddens):
    for i in range(num_residual_layers):
        h_i = tf.nn.relu(h)
        
        h_i = snt.Conv2D(
              output_channels=num_residual_hiddens,
              kernel_shape=(3, 3),
              stride=(1, 1),
              name="res3x3_%d" % i)(h_i)
        h_i = tf.nn.relu(h_i)

        h_i = snt.Conv2D(
              output_channels=num_hiddens,
              kernel_shape=(1, 1),
              stride=(1, 1),
              name="res1x1_%d" % i)(h_i)
        h += h_i
    return tf.nn.relu(h)

##encoder
class Encoder(snt.AbstractModule):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, name='encoder'):
        super(Encoder, self).__init__(name=name)
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

    def _build(self, x):
        h = snt.Conv2D(
            output_channels=self._num_hiddens / 2,
            kernel_shape=(4, 4),
            stride=(2, 2),
            name="enc_1")(x)
        h = tf.nn.relu(h)
                   
        h = snt.Conv2D(
            output_channels=self._num_hiddens,
            kernel_shape=(4, 4),
            stride=(2, 2),
            name="enc_2")(h)
        h = tf.nn.relu(h)
                   
        h = snt.Conv2D(
            output_channels=self._num_hiddens,
            kernel_shape=(3, 3),
            stride=(1, 1),
            name="enc_3")(h)
                   
        h = residual_stack(
            h,
            self._num_hiddens,
            self._num_residual_layers,
            self._num_residual_hiddens)
        return h

##decoder
class Decoder(snt.AbstractModule):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, name='decoder'):
        super(Decoder, self).__init__(name=name)
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

    def _build(self, x):
        h = snt.Conv2D(
            output_channels=self._num_hiddens,
            kernel_shape=(3, 3),
            stride=(1, 1),
            name="dec_1")(x)
        
        h = residual_stack(
            h,
            self._num_hiddens,
            self._num_residual_layers,
            self._num_residual_hiddens)
                   
        h = snt.Conv2DTranspose(
            output_channels=int(self._num_hiddens / 2),
            output_shape=None,
            kernel_shape=(4, 4),
            stride=(2, 2),
            name="dec_2")(h)
        h = tf.nn.relu(h)
                   
        x_recon = snt.Conv2DTranspose(
                  output_channels=1,
                  output_shape=None,
                  kernel_shape=(4, 4),
                  stride=(2, 2),
                  name="dec_3")(h)
                   
        return x_recon

### MAIN ###
tf.reset_default_graph()

# Set hyper-parameters.
batch_size = 4
image_size = 84

# 100k steps should take < 30 minutes on a modern (>= 2017) GPU.
num_training_updates = 10 #epoch

num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2
# These hyper-parameters define the size of the model (number of parameters and layers).
# The hyper-parameters in the paper were (For ImageNet):
# batch_size = 128
# image_size = 128
# num_hiddens = 128
# num_residual_hiddens = 32
# num_residual_layers = 2

# This value is not that important, usually 64 works.
# This will not change the capacity in the information-bottleneck.
embedding_dim = 64

# The higher this value, the higher the capacity in the information bottleneck.
num_embeddings = 512

# commitment_cost should be set appropriately. It's often useful to try a couple
# of values. It mostly depends on the scale of the reconstruction cost
# (log p(x|z)). So if the reconstruction cost is 100x higher, the
# commitment_cost should also be multiplied with the same amount.
commitment_cost = 0.25

# Use EMA updates for the codebook (instead of the Adam optimizer).
# This typically converges faster, and makes the model less dependent on choice
# of the optimizer. In the VQ-VAE paper EMA updates were not used (but was
# developed afterwards). See Appendix of the paper for more details.
vq_use_ema = False

# This is only used for EMA updates.
decay = 0.99

learning_rate = 3e-4

# Data Loading.
train_dataset_iterator = (
                          tf.data.Dataset.from_tensor_slices(train_data_dict)
                          .map(cast_and_normalise_images)
                          .shuffle(10000)
                          .repeat(-1)  # repeat indefinitely
                          .batch(batch_size)).make_one_shot_iterator()
valid_dataset_iterator = (
                          tf.data.Dataset.from_tensor_slices(valid_data_dict)
                          .map(cast_and_normalise_images)
                          .repeat(1)  # 1 epoch
                          .batch(batch_size)).make_initializable_iterator()
train_dataset_batch = train_dataset_iterator.get_next()
valid_dataset_batch = valid_dataset_iterator.get_next()

def get_images(sess, subset='train'):
    if subset == 'train':
        return sess.run(train_dataset_batch)['images']
    elif subset == 'valid':
        return sess.run(valid_dataset_batch)['images']

# Build modules.
encoder = Encoder(num_hiddens, num_residual_layers, num_residual_hiddens)
decoder = Decoder(num_hiddens, num_residual_layers, num_residual_hiddens)
pre_vq_conv1 = snt.Conv2D(
               output_channels=embedding_dim,
               kernel_shape=(1, 1),
               stride=(1, 1),
               name="to_vq")

if vq_use_ema:
    vq_vae = snt.nets.VectorQuantizerEMA(
             embedding_dim=embedding_dim,
             num_embeddings=num_embeddings,
             commitment_cost=commitment_cost,
             decay=decay)
else:
    vq_vae = snt.nets.VectorQuantizer(
             embedding_dim=embedding_dim,
             num_embeddings=num_embeddings,
             commitment_cost=commitment_cost)

# Process inputs with conv stack, finishing with 1x1 to get to correct size.
x = tf.placeholder(tf.float32, shape=(None, image_size, image_size, 1))
z = pre_vq_conv1(encoder(x)) #latent space

# vq_output_train["quantize"] are the quantized outputs of the encoder.
# That is also what is used during training with the straight-through estimator.
# To get the one-hot coded assignments use vq_output_train["encodings"] instead.
# These encodings will not pass gradients into to encoder,
# but can be used to train a PixelCNN on top afterwards.

# For training
vq_output_train = vq_vae(z, is_training=True)
x_recon = decoder(vq_output_train["quantize"])

recon_error = tf.reduce_mean((x_recon - x)**2) / data_variance  # Normalized MSE # reconstruction loss
loss = recon_error + vq_output_train["loss"] #total loss: reconstructed loss + commitment loss + codebook loss

# For evaluation, make sure is_training=False!
vq_output_eval = vq_vae(z, is_training=False)
x_recon_eval = decoder(vq_output_eval["quantize"])

# The following is a useful value to track during training.
# It indicates how many codes are 'active' on average.
perplexity = vq_output_train["perplexity"]

# Create optimizer and TF session.
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss)
sess = tf.train.SingularMonitoredSession()

# Train.
train_res_recon_error = []
train_res_perplexity = []
for i in xrange(num_training_updates):
    feed_dict = {x: get_images(sess)}
    results = sess.run(
              [train_op, recon_error, perplexity],
              feed_dict=feed_dict)
    train_res_recon_error.append(results[1])
    train_res_perplexity.append(results[2])

    if (i+1) % 100 == 0:
        print('%d iterations' % (i+1))
        print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
        print('perplexity: %.3f' % np.mean(train_res_perplexity[-100:]))
        print()

# Output reconstruction loss and average codebook usage
f = plt.figure(figsize=(16,8))
ax = f.add_subplot(1,2,1)
ax.plot(train_res_recon_error)
ax.set_yscale('log')
ax.set_title('NMSE.')

ax = f.add_subplot(1,2,2)
ax.plot(train_res_perplexity)
ax.set_title('Average codebook usage (perplexity).')
plt.savefig('loss.eps')

# Reconstructions
sess.run(valid_dataset_iterator.initializer)
train_originals = get_images(sess, subset='train')
train_reconstructions = sess.run(x_recon_eval, feed_dict={x: train_originals})
valid_originals = get_images(sess, subset='valid')
valid_reconstructions = sess.run(x_recon_eval, feed_dict={x: valid_originals})

def convert_batch_to_image_grid(image_batch):
    reshaped = (image_batch.reshape(2, 2, 84, 84)
                .transpose(0, 2, 1, 3)
                .reshape(2 * 84, 2 * 84))
    return reshaped + 0.5

# Plot the results
f = plt.figure(figsize=(16,8))
ax = f.add_subplot(2,2,1)
ax.imshow(convert_batch_to_image_grid(train_originals),
          interpolation='nearest')
ax.set_title('training data originals')
plt.axis('off')

ax = f.add_subplot(2,2,2)
ax.imshow(convert_batch_to_image_grid(train_reconstructions),
          interpolation='nearest')
ax.set_title('training data reconstructions')
plt.axis('off')

ax = f.add_subplot(2,2,3)
ax.imshow(convert_batch_to_image_grid(valid_originals),
          interpolation='nearest')
ax.set_title('validation data originals')
plt.axis('off')

ax = f.add_subplot(2,2,4)
ax.imshow(convert_batch_to_image_grid(valid_reconstructions),
          interpolation='nearest')
ax.set_title('validation data reconstructions')
plt.axis('off')
plt.savefig('reconstruction.eps')

#timer
print('\n', '## CODE RUNTIME:', time.time()-Tstart) #Timer end
