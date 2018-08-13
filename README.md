# JpegArtifactRemoval

## Current Solution
Simple network for image-to-image translation. Implemented variations:
1. U-net
2. SR-CNN: C. Dong, C. C. Loy, K. He, and X. Tang. "Learning a deep convolutional network for image super-resolution"
3. AR-CNN: C. Dong, Y. Deng, C. Change Loy, and X. Tang, “Compression artifacts reduction by a deep convolutional network”
4. DN-CNN: K. Zhang, W. Zuo, Y. Chen, D. Meng, and L. Zhang. "Beyond a Gaussian denoiser: Residual learning of deep CNN for image denoising"

## Future Ideas
1. Add data augmentations.
2. Try more models, especially ResNet-based models.
3. Try a window of `2*k+1` adjacent images in order to reconstruct the middle image.
4. Try generative-based models.

### List of relevant papers
Compression Artifacts Removal Using Convolutional Neural Networks
https://arxiv.org/pdf/1605.00366.pdf

CAS-CNN: A Deep Convolutional Neural Network for Image Compression Artifact Suppression
https://arxiv.org/pdf/1611.07233.pdf

Deep Generative Adversarial Compression Artifact Removal
http://www.micc.unifi.it/seidenari/wp-content/papercite-data/pdf/iccv_2017.pdf

Compression Artifacts Reduction by a Deep Convolutional Network
https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Dong_Compression_Artifacts_Reduction_ICCV_2015_paper.pdf

Deep Image Prior
https://dmitryulyanov.github.io/deep_image_prior

Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising
https://arxiv.org/pdf/1608.03981.pdf
