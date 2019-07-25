# Kavli_UCSC_2019
 The code is used for the project of Kavli summer program in Astrophysics in 2019. 

## What do we want to do?
 In this project, we are exploring the usage of [VQ-VAE](https://arxiv.org/pdf/1906.00446.pdf) to two ideas:
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
  Th below result is that I added **only noise layer** and run 200000 epochs.
  ![](https://github.com/tycheng-sunny/Project_Kavli_UCSC_2019/blob/master/result_plots/reconstruction_noise_200000.jpg)
  ![](https://github.com/tycheng-sunny/Project_Kavli_UCSC_2019/blob/master/result_plots/loss_noise200000.png)
  
  Then I added **both noise and PSF layers** in the Decoder:
  I run 100000 epochs for this example only on CPU which costs me ~3-4hrs on my MacBook Pro.
  The data is from CANDELS (GOODS, h band), and the PSF image is from [3DHST](https://3dhst.research.yale.edu/Data.php) (GOODS-S WFC3 PSFs, F160W).
  
  However, the process converges faster than the previous two processes. Not sure about what causes this situation.
  
  ![](https://github.com/tycheng-sunny/Project_Kavli_UCSC_2019/blob/master/result_plots/reconstruction_noise+psf_100000.jpg)
  ![](https://github.com/tycheng-sunny/Project_Kavli_UCSC_2019/blob/master/result_plots/loss_noise+psf_100000.png)
  
 
## To-do list
 - [x] Adapt the VQ-VAE code to astronomical images (e.g. CANDELS)
 - [x] Add noise/PSF layers before output and retrieve the reconstructed images from the layer before noise/PSF layers
 - [ ] Download the images fro 3DHST/CANDELS and cropping them (currently using the CANDELS data from Ryan)
 - [ ] Modifiy the architecture to have better output of encoder to adapt to generative models
 - [ ] Connect with PixelCNN to generate random galaxy images
 - [ ] Conditional to what labels? --> how to do conditional?
 - [ ] Where can we connect with to calculate the clustering part?
 - [ ] Examination......?
