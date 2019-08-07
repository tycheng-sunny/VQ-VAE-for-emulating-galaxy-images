#===========================================================================================
# PixelCNN trained on vector quantized feature map from VQ-VAE
# _condition: trained with labels
# ** Check if hyper-parameters are the value/info you want.
#===========================================================================================
from __future__ import print_function

import time
import os
import sys
import pickle
os.environ["CUDA_VISIBLE_DEVICES"]="3" # for GPU

import matplotlib as mpl
mpl.use('Agg') # to use matplotlib without visualisation envs
import matplotlib.pyplot as plt
import numpy as np
import sonnet as snt
import tensorflow as tf

from six.moves import xrange
from tqdm import tqdm

#timer starts
Tstart = time.time()

### Data Preparation ###
## path
local_data_dir = 'data/' # path to input data
output_PATH = 'output/' # path for output results
data_fn = 'data_sdss_wcata_tmp.dict' # input data filename
savemodel_vqvae_fn = 'savemodels/vqvae/vqvae_sdss.ckpt' # vqvae model name
savemodel_pixecnn_fn = 'savemodels/pixelcnn/' # pixelcnn model saving directory

## hyper-parameters for data
image_size = 64
channel_size = 1

## load data into Numpy
def unpickle(filename):
    with open(filename, 'rb') as fo:
        return pickle.load(fo)

def reshape_flattened_image_batch(flat_image_batch):
    return flat_image_batch.reshape(-1, image_size, image_size, 1)  # convert to NHWC

def combine_batches(batch_list):
    images = np.vstack([reshape_flattened_image_batch(batch_list['images'])])
    fn = np.vstack([np.array(batch_list['filename'])]).reshape(-1, 1)
    id = np.vstack([np.array(batch_list['id'])]).reshape(-1, 1)
    TT = np.vstack([np.array(batch_list['TT_pred'])]).reshape(-1)
    TTint = np.vstack([np.array(batch_list['TT_int'])]).reshape(-1)
    return {'images': images, 'filename': fn, 'id': id, 'TT': TT, 'TTint': TTint}

data_dict = combine_batches(unpickle(os.path.join(local_data_dir, data_fn)))

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

##pixel cnn
class PixelCNN(object):
    def __init__(self, lr, global_step, grad_clip,
                 size, embeds, K, D,
                 num_classes, num_layers, num_maps,
                 is_training=True):
        sys.path.append('pixelcnn')
        from layers import GatedCNN
        self.X = tf.placeholder(tf.int64, [None, size, size]) # input extracted feature map that each pixel is labeled by k
        
        if( num_classes is not None ):
            """ conditional PixelCNN which can conditionally do training when given labels """
            self.h = tf.placeholder(tf.int32, [None,])
            onehot_h = tf.one_hot(self.h, num_classes, axis=-1)
        else:
            onehot_h = None
        
        X_processed = tf.gather(tf.stop_gradient(embeds), self.X)
        
        v_stack_in, h_stack_in = X_processed, X_processed
        for i in range(num_layers):
            filter_size = 3 if i > 0 else 7
            mask = 'b' if i > 0 else 'a'
            residual = True if i > 0 else False
            i = str(i)
            with tf.variable_scope("v_stack"+i):
                v_stack = GatedCNN([filter_size, filter_size, num_maps], v_stack_in, mask=mask, conditional=onehot_h).output()
                v_stack_in = v_stack
            
            with tf.variable_scope("v_stack_1"+i):
                v_stack_1 = GatedCNN([1, 1, num_maps], v_stack_in, gated=False, mask=mask).output()
            
            with tf.variable_scope("h_stack"+i):
                h_stack = GatedCNN([1, filter_size, num_maps], h_stack_in, payload=v_stack_1, mask=mask, conditional=onehot_h).output()
            
            with tf.variable_scope("h_stack_1"+i):
                h_stack_1 = GatedCNN([1, 1, num_maps], h_stack, gated=False, mask=mask).output()
                if residual:
                    h_stack_1 += h_stack_in # Residual connection
                h_stack_in = h_stack_1
        
        with tf.variable_scope("fc_1"):
            fc1 = GatedCNN([1, 1, num_maps], h_stack_in, gated=False, mask='b').output()
        
        with tf.variable_scope("fc_2"):
            self.fc2 = GatedCNN([1, 1, K], fc1, gated=False, mask='b', activation=False).output()
            self.dist = tf.distributions.Categorical(logits=self.fc2)
            self.sampled = self.dist.sample()
            self.log_prob = self.dist.log_prob(self.sampled)
    
        loss_per_batch = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.fc2, labels=self.X), axis=[1,2])
        self.loss = tf.reduce_mean(loss_per_batch, axis=0)
                                                                                      
        save_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,tf.contrib.framework.get_name_scope())
        self.saver = tf.train.Saver(var_list=save_vars, max_to_keep = 3)
                                                                                          
        if( is_training ):
            with tf.variable_scope('backward'):
                optimizer = tf.train.AdamOptimizer(lr)
                                                                                                  
                gradients = optimizer.compute_gradients(self.loss, var_list=save_vars)
                if( grad_clip is None ):
                    clipped_gradients = gradients
                else :
                    clipped_gradients = [(tf.clip_by_value(_[0], -grad_clip, grad_clip), _[1]) for _ in gradients]
                    #clipped_gradients = [(tf.clip_by_average_norm(_[0], grad_clip), _[1]) for _ in gradients]
                self.train_op = optimizer.apply_gradients(clipped_gradients,global_step)
        #for var in save_vars:
        #    print(var,var.name)

    def sample_from_prior(self, sess, classes, batch_size):
        # Generates len(classes)* batch_size Z samples.
        size = self.X.get_shape()[1]
        feed_dict={self.X: np.zeros([len(classes)* batch_size, size, size], np.int32)}
        #feed_dict={self.X: np.zeros([batch_size, size, size], np.int32)}
        if( classes is not None ):
            feed_dict[self.h] = np.repeat(classes, batch_size).astype(np.int32)
        
        log_probs = np.zeros((len(classes)* batch_size, ))
        #log_probs = np.zeros((batch_size, ))
        for i in xrange(size):
            for j in xrange(size):
                sampled,log_prob = sess.run([self.sampled, self.log_prob],feed_dict=feed_dict)
                feed_dict[self.X][:,i,j]= sampled[:,i,j]
                log_probs += log_prob[:,i,j]
        return feed_dict[self.X], log_probs
    
    def save(self, sess, savepath, step=None):
        if(step is not None):
            self.saver.save(sess, savepath+'/model-pixelcnn_sdss_condition.ckpt', global_step=step)
        else :
            self.saver.save(sess, savepath+'/last-pixelcnn_sdss_condition.ckpt')

    def load(self, sess, model):
        self.saver.restore(sess, model)

### MAIN ###
# Set hyper-parameters.
image_size = 64

# hyper-parameters for VQ-VAE
num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2

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

# Build modules.
encoder = Encoder(num_hiddens, num_residual_layers, num_residual_hiddens)
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
z = pre_vq_conv1(encoder(x))

vq_output_train = vq_vae(z, is_training=False)
vq_output_embeds = vq_vae.embeddings # the codebook ek [D,K]

# Create TF session for vqvae
saver = tf.train.Saver()
sess = tf.train.SingularMonitoredSession()
saver.restore(sess, savemodel_vqvae_fn) # load vqvae pre-trained model

### Pixel CNN ###
# retrieve the input for PixelCNN sequentially
N = 50 # run N each time
runN = len(data_dict["id"])// N
for i in range(runN):
    if i == 0:
        embeds_indice_4_imgs = sess.run(vq_output_train["encoding_indices"], feed_dict={x: data_dict['images'][i*N: (i+1)*N]})
    else:
        tmp = sess.run(vq_output_train["encoding_indices"], feed_dict={x: data_dict['images'][i*N: (i+1)*N]})
        embeds_indice_4_imgs = np.concatenate((embeds_indice_4_imgs, tmp), axis=0)

tmp = sess.run(vq_output_train["encoding_indices"], feed_dict={x: data_dict['images'][runN* N:]})
embeds_indice_4_imgs = np.concatenate((embeds_indice_4_imgs, tmp), axis=0)
print('Shape of input: ', np.shape(embeds_indice_4_imgs))

# codebook
embeds = sess.run(vq_output_embeds, feed_dict={x: data_dict['images'][:2]})
embeds = np.transpose(embeds) # change tthe shape from [D,K] to [K,D]

# Reset the graph
tf.reset_default_graph()
# Set hyper-parameter
batch_size = 32 # training
train_num = 200 # need GPU for training
learning_rate = 3e-4
decay_steps = 50000 # half of training number
decay_val = 0.5
decay_staircase = False
grad_clip = 5.0
latent_size = np.shape(embeds_indice_4_imgs)[1] # the shape of latent map is (latent_size, latent_size)
num_layers = 18
num_feature_maps = latent_size* latent_size
K, D = np.shape(embeds)[0], np.shape(embeds)[1] # K=num_embeddings, D=embedding_dim

# classes number
Nmax = np.max(data_dict["TTint"])
Nmin = np.min(data_dict["TTint"])
num_classes = Nmax - Nmin +1

# classes
classes = np.arange(-3, 7, 1)

# Save frequency
save_period = 10000 # save model
summary_period = 100

# Data loading
def decode_and_get_images(x):
    x = x.decode('utf-8')
    data = np.array(embeds_indice_4_imgs[np.where(data_dict["id"] == x)[0][0]])
    return data

def decode_and_get_labels(x):
    x = x.decode('utf-8')
    labels = np.array(data_dict["TTint"][np.where(data_dict["id"] == x)[0][0]])
    return labels

def load(x):
    x = x.numpy()
    x = np.array(list(map(decode_and_get_images, x))) # feature map size (21, 21)
    return x

def load_labels(x):
    x = x.numpy()
    x = np.array(list(map(decode_and_get_labels, x))) # labels (-1)
    return x

def loader(y):
    imgs = tf.py_function(load, [y], tf.int64)
    labels = tf.py_function(load_labels, [y], tf.int64)
    imgs = tf.cast(imgs, tf.int64)
    #labels = tf.cast(labels, tf.int64)
    return [imgs[0], labels[0]]

train_paths = tf.data.Dataset.from_tensor_slices(data_dict["id"]) # load the id
train_dset = train_paths.map(loader)

train_dset = train_dset.repeat(-1).shuffle(10000).batch(batch_size)
train_iterator = train_dset.make_one_shot_iterator()
train_dataset_batch = train_iterator.get_next()

def get_features(sess):
    xs = sess.run(train_dataset_batch[0])
    ys = sess.run(train_dataset_batch[1])
    return xs, ys

# For training pixelcnn
x_cnn = tf.placeholder(tf.int32, [None, latent_size, latent_size])
gen = tf.gather(embeds, x_cnn)

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_val, staircase=decay_staircase)
tf.summary.scalar('lr', learning_rate)

# pixelcnn model setting
pixelcnn_net = PixelCNN(learning_rate, global_step, grad_clip, latent_size, embeds, K, D, num_classes, num_layers, num_feature_maps)

# Summary Operations
tf.summary.scalar('loss', pixelcnn_net.loss)
summary_op = tf.summary.merge_all()
    
# Initialize op
config = tf.ConfigProto()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

# PixelCNN Training RUN!!
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.graph.finalize()
sess.run(init_op)
    
summary_writer = tf.summary.FileWriter(savemodel_pixecnn_fn, sess.graph)
    
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord, sess=sess)
itarr, lossarr = [], [] # record the iteration and loss
try:
    for step in tqdm(xrange(train_num), dynamic_ncols=True):
        batch_xs, batch_ys = get_features(sess)
        it, loss, _ = sess.run([global_step, pixelcnn_net.loss, pixelcnn_net.train_op], feed_dict={pixelcnn_net.X:batch_xs, pixelcnn_net.h:batch_ys})
            
        if( it % save_period == 0 ):
            pixelcnn_net.save(sess, savemodel_pixecnn_fn, step=it)
            sampled_zs, log_probs = pixelcnn_net.sample_from_prior(sess, classes, batch_size)
            sampled_ims = sess.run(gen, feed_dict={x_cnn:sampled_zs})
            
        if( it % summary_period == 0 ):
            tqdm.write('[%5d] Loss: %1.3f'%(it, loss))
            itarr.append(it)
            lossarr.append(loss)
            summary = sess.run(summary_op, feed_dict={pixelcnn_net.X:batch_xs, pixelcnn_net.h:batch_ys})
            summary_writer.add_summary(summary, it)

except Exception as e:
    coord.request_stop(e)
finally :
    pixelcnn_net.save(sess, savemodel_pixecnn_fn)
        
    coord.request_stop()
    coord.join(threads)

    plt.plot(itarr, lossarr)
    plt.savefig(output_PATH + 'loss_pixelcnn_sdss_condition.eps')
#timer

print('\n', '## CODE RUNTIME:', time.time()-Tstart) #Timer end
