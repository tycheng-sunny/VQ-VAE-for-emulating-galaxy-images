#===========================================================================================
# VQ-VAE implementation
# ** Check if hyper-parameters are the value/info you want.
#===========================================================================================
from __future__ import print_function

import time
import os
import pickle
#os.environ["CUDA_VISIBLE_DEVICES"]="1" # for GPU

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
local_data_dir = 'data/' # path to input data
output_PATH = 'output/' # path for output results
data_fn = 'data_candels.dict' # input data filename
savemodel_vqvae_fn = 'savemodels/vqvae/vqvae_candels.ckpt' # vqvae model name

## hyper-parameters for data
ori_image_size = 84
channel_size = 1

## load data into Numpy
def unpickle(filename):
    with open(filename, 'rb') as fo:
        return pickle.load(fo)

def reshape_flattened_image_batch(flat_image_batch):
    return flat_image_batch.reshape(-1, ori_image_size, ori_image_size, channel_size)  # convert to NHWC

def combine_batches(batch_list):
    images = np.vstack([reshape_flattened_image_batch(batch_list['images'])])
    fn = np.vstack([np.array(batch_list['filename'])]).reshape(-1,1)
    id = np.vstack([np.array(batch_list['id'])]).reshape(-1,1)
    return {'images': images, 'filename': fn, 'id': id}

data_dict = combine_batches(unpickle(os.path.join(local_data_dir, data_fn)))
train_data_num = np.int(0.8* len(data_dict["id"])) # the number of training data # can change the ratio
print('Number of training data:', train_data_num)
print('Number of validation data:', (len(data_dict["id"]) - train_data_num))

data_variance = np.var(data_dict['images'][:train_data_num]) # for the normalisation of the reconstruction loss

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

# Train mode
Train = True # True: training a new model and save the model. False: reloading the pretrained model
# Set hyper-parameters.
image_size = 84
batch_size = 32

# 100k steps should take < 30 minutes on a modern (>= 2017) GPU.
num_training_updates = 100000 #epoch

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
def decode_and_get_image(x):
    x = x.decode('utf-8')
    data = np.array(data_dict["images"][np.where(data_dict["id"] == x)[0][0]])
    return data

def load(x):
    x = x.numpy()
    x = np.array(list(map(decode_and_get_image, x))) #images size = ori_image_size
    #print(x)
    #x = x[:,96:160,96:160]   #cropping images
    #x = np.array(list(map(get_zoom,x))) #zoom to half size
    return x

def get_zoom(x):
    x = zoom(x,0.5)
    return x

def loader(y):
    imgs = tf.py_function(load, [y], tf.float32)
    imgs = tf.cast(imgs, tf.float32)
    return imgs[0]

# training data
train_paths = tf.data.Dataset.from_tensor_slices(data_dict["id"][:train_data_num]) # load the id
train_dset = train_paths.map(loader)

train_dset = train_dset.repeat(-1).shuffle(10000).batch(batch_size)
train_iterator = train_dset.make_one_shot_iterator()
train_dataset_batch = train_iterator.get_next()

# validation data
valid_paths = tf.data.Dataset.from_tensor_slices(data_dict["id"][train_data_num:])
valid_dset = valid_paths.map(loader)

valid_dset = valid_dset.repeat(1).batch(batch_size)
valid_iterator = valid_dset.make_one_shot_iterator()
valid_dataset_batch = valid_iterator.get_next()

def get_images(sess, subset='train'):
    if subset == 'train':
        return sess.run(train_dataset_batch)
    elif subset == 'valid':
        return sess.run(valid_dataset_batch)

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
saver = tf.train.Saver()
sess = tf.train.SingularMonitoredSession()

# Special adapted code for saving model when using MonitoredSession() # Don't know why yet...
def get_session(sess):
    session = sess
    while type(session).__name__ != 'Session':
        #pylint: disable=W0212
        session = session._sess
    return session

# set the Train mode. If Train=True: do training and save the model, Train=False: reloading the pre-trained model
if Train:
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
            #saver.save(sess, 'vqvae_model/model', global_step=i+1)
            print('%d iterations' % (i+1))
            print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
            print('perplexity: %.3f' % np.mean(train_res_perplexity[-100:]))
            print()
    
    saver.save(get_session(sess), savemodel_vqvae_fn)
    # Output reconstruction loss and average codebook usage
    f = plt.figure(figsize=(16,8))
    ax = f.add_subplot(1,2,1)
    ax.plot(train_res_recon_error)
    ax.set_yscale('log')
    ax.set_title('NMSE.')

    ax = f.add_subplot(1,2,2)
    ax.plot(train_res_perplexity)
    ax.set_title('Average codebook usage (perplexity).')
    plt.savefig(output_PATH + 'loss.eps')

else:
    saver.restore(sess, savemodel_vqvae_fn)
'''
# Reconstructions
#sess.run(valid_dataset_iterator.initializer)
train_originals = get_images(sess, subset='train')
train_reconstructions = sess.run(x_recon_eval, feed_dict={x: train_originals})
valid_originals = get_images(sess, subset='valid')
valid_reconstructions = sess.run(x_recon_eval, feed_dict={x: valid_originals})

def convert_batch_to_image_grid(image_batch):
    reshaped = (image_batch.reshape(2, 2, image_size, image_size) # batch_size and image_size
                .transpose(0, 2, 1, 3)
                .reshape(2 * image_size, 2 * image_size))
    return reshaped + 0.5

# Plot the results
f = plt.figure(figsize=(16,8))
ax = f.add_subplot(2,2,1)
ax.imshow(convert_batch_to_image_grid(train_originals),
          interpolation='nearest', cmap='gray_r')
ax.set_title('training data originals')
plt.axis('off')

ax = f.add_subplot(2,2,2)
ax.imshow(convert_batch_to_image_grid(train_reconstructions),
          interpolation='nearest', cmap='gray_r')
ax.set_title('training data reconstructions')
plt.axis('off')

ax = f.add_subplot(2,2,3)
ax.imshow(convert_batch_to_image_grid(valid_originals),
          interpolation='nearest', cmap='gray_r')
ax.set_title('validation data originals')
plt.axis('off')

ax = f.add_subplot(2,2,4)
ax.imshow(convert_batch_to_image_grid(valid_reconstructions),
          interpolation='nearest', cmap='gray_r')
ax.set_title('validation data reconstructions')
plt.axis('off')
plt.savefig(output_PATH + 'reconstruction.eps')
'''
#timer
print('\n', '## CODE RUNTIME:', time.time()-Tstart) #Timer end
