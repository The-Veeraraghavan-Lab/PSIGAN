# PSIGAN: Joint probabilistic segmentation and image distribution matching for unpaired cross-modality adaptation segmentation
## Jue Jiang, ... Harini Veeraraghavan

This is the code for our accepted manuscript in IEEE Trans Medical Imaging. Briefly, this is a GAN based approach for unpaired cross-modality learning between highly different imaging modalities like CT and MRI for unsupervised segmentation (where no target modality expert segmentations are available during training) on the target modality. The core idea of the approach is a joint distribution matching structure discriminator, which combines a pair of images - an image and its segmentation map to compute mismatches in the joint distribution of synthesized and expected target modality images. 

More details of the algorithm and the datasets used in this paper are available in https://arxiv.org/abs/2007.09465 

If you use this code, please cite our TMI paper/arxiv link.

## Prerequisities
Python, PyTorch, Numpy, Scipy, Matplotlib, and a recent NVIDIA GPU
## Train
python train_PSIGAN.py
## NOTE
This code is being updated: 
## Acknowledgement

