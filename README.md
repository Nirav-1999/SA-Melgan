# SA-Melgan

## A modified and better version of Melgan

### Abstract
Previous works like "MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis" 
have shown that it is possible to reliably generate coherent
waveforms of good quality by employing a Generative Adversarial Network(GAN). With an intent to
improve the stability of the GAN and enhance the quality of the generated audio, we propose a modified
version of MelGAN. We combine the ideas from Self-Attention GAN by Zhang et al. and MelGAN architecture.
Self-attention GAN have shown improved results in the generation of high-quality images. To make
use of the Self-Attention layer’s quality of establishing long-range dependencies in audio generation[2],
we embed a one-dimensional self-attention within the MelGAN’s generator. In this paper, we use the
LJ Speech Dataset to train MelGAN as well as our proposed architecture. To interpret the results, we
calculate and compare the Mean Opinion Scores and Mean Opinion Score- Listening Quality Objective
of both the architectures trained until the thousandth epoch.
