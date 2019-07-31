# Kavli_UCSC_2019
 The code is used for the project of Kavli summer program in Astrophysics in 2019. 

## What do we want to do?
 In this project, we are exploring the usage of VQ-VAE ([1](https://arxiv.org/abs/1711.00937)) ([2](https://arxiv.org/pdf/1906.00446.pdf)) to two ideas:
 - Unsupervised machine learning (clustering) to explore the morphology of galaxies
 - Emulating galaxy images (without noise/psf) to adapt to other surveys using generative models (e.g. PixelCNN)
 
 We modified the [VQ-VAE example](https://github.com/deepmind/sonnet/blob/master/sonnet/examples/vqvae_example.ipynb) ([Source code of VQ-VAE](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/nets/vqvae.py)) from Sonnet library.
 
## Preliminary results
### VQ-VAE code on CANDELS data
 ```
 $python vqvae.py
 ```
  I run 200000 epochs on CPU which costs me ~5-6hrs on my MacBook Pro. (GPU will speed up more)
  The data is from CANDELS (GOODS, h band).
  
  <p align="center">
  <img src="https://github.com/tycheng-sunny/Project_Kavli_UCSC_2019/blob/master/result_plots/reconstruction_200000.png" width=400>
  </p>
  
  ![](https://github.com/tycheng-sunny/Project_Kavli_UCSC_2019/blob/master/result_plots/loss_200000.png)
  
### Add noise/PSF layers
  ```
  $python vqvae_noise+psf.py
  ```  
  I added **noise and PSF layers** in the Decoder and run 100000 epochs for this example only on CPU which costs me ~3-4hrs on my MacBook Pro.
  The data is from CANDELS (GOODS, h band), and the PSF image is from [3DHST](https://3dhst.research.yale.edu/Data.php) (GOODS-S WFC3 PSFs, F160W).
  
  The process converges much faster than the previous process.
  
  ![](https://github.com/tycheng-sunny/Project_Kavli_UCSC_2019/blob/master/result_plots/reconstruction_noise+psf_100000.jpg)
  ![](https://github.com/tycheng-sunny/Project_Kavli_UCSC_2019/blob/master/result_plots/loss_noise+psf_100000.png)
  
  The convolution test is shown below that the last column shows the results after PSF convolution on the images of the third column:
  ![](https://github.com/tycheng-sunny/Project_Kavli_UCSC_2019/blob/master/result_plots/reconstruction_noise%2Bpsf_reconfirm.png)
  
### Connect with Gated PixelCNN
  In this section, I trained [Gated PixelCNN](https://arxiv.org/pdf/1606.05328.pdf) on vector quantized feature map. After implementing VQVAE, we will have two outputs from VQVAE which are the inputs for PixelCNN:
  
  - codebook (e): shape=(K,D), where K is the number of code and D is the dimension of the code.
  - encodings indice: shape=( , latent_size, latent_size, dtype=**int32**), where 'latent_size' means the size of latent map.
  
  I followed the code from [hiwonjoon](https://github.com/hiwonjoon/tf-vqvae) which tried to reproduce the results of the paper - VQ-VAE [2](https://arxiv.org/pdf/1906.00446.pdf). They applied the code of [Gated PixelCNN](https://github.com/anantzoid/Conditional-PixelCNN-decoder/tree/9a5c9a3df2c58100cf5e3600392e67db8ac7a59e) on the vector quantized feature map extracted from VQ-VAE.
  
  I trained PixelCNN 50000 epoches, and sample the prior to generate new feature map. I then use the decoder from VQ-VAE to reconstruct new images from generated feature maps. (waiting for results...)
  
  
### Unsupervised clustering
 
## To-do list
 - [x] Adapt the VQ-VAE code to astronomical images (e.g. CANDELS)
 - [x] Add noise/PSF layers before output and retrieve the reconstructed images from the layer before noise/PSF layers
 - [x] Connect with PixelCNN to generate random galaxy images
 - [ ] t-SNE to explore the meaning of quantized vector and compare with galaxy properties
 - [ ] Download the images fro 3DHST/CANDELS and cropping them (currently using the CANDELS data from Ryan)
 - [ ] SDSS data
 - [ ] Conditional to what labels? --> how to do conditional?
 - [ ] Where can we connect with to calculate the clustering part?
 - [ ] Examination......?
