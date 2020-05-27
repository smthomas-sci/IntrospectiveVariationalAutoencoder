# Introspective Variational Autoencoder

This module contains the fundamental building blocks to implement
the Introspective Variational Autoencoder (IntroVAE), introduced by
[Huagn *et al* 2018](https://arxiv.org/abs/1807.06358)

![IntroVAE_overview](./assets/network_overview.png)

Since the main contribution of IntroVAEs is the training regime, the
methodology can be applied to any variational network. Therefore, included here are style-based 
architectures from [Karras *et al* 2018](https://arxiv.org/abs/1812.04948) using
Adaptive Instance Normalisation, as well as the skip and residual connections
used in their [latest work (2019)](https://arxiv.org/abs/1912.04958). Such additions make for predictably stable
adversarial training.

This implementation borrowed the hairy loss implementation from `bbeatrix` available on 
[GitHub](https://github.com/bbeatrix/introvae). The Adaptive Instance Normalisation
Layer was borrowed from `manicman1999` available [here](https://github.com/manicman1999/StyleGAN-Keras). Additionally,
the [Nvidia Labs GitHub](https://github.com/NVlabs/stylegan2/) served as some interesting and helpful reading.


# Demo 64x64 Anime

<p align="center">
    <figure>
        <img src="https://github.com/smthomas-sci/IntrospectiveVariationalAutoencoder/blob/master/assets/anime_loss.png" width="350">
        <figcaption>Fig.1 - Training Progress.</figcaption>
        <br>
    </figure>
    <figure>
        <img src="https://github.com/smthomas-sci/IntrospectiveVariationalAutoencoder/blob/master/assets/anime_interp.png" width="350">
        <figcaption>Fig.2 - Interpolations between real images</figcaption>
        <br>
    </figure>
    <figure>
        <img src="https://github.com/smthomas-sci/IntrospectiveVariationalAutoencoder/blob/master/assets/anime_sample.jpg" width="350">
        <figcaption>Fig.3 - Random Samples (mean = 0, std. = 1)</figcaption>
        <br>
    </figure>
</p>
 

  
