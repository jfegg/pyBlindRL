import torch

import tqdm
from PIL import Image
from commands import generate_initial_psf, RL_deconv_blind

import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from matplotlib import image
from scipy.signal import convolve2d as conv2
from skimage import color, data, restoration

rng = np.random.default_rng()

astro = color.rgb2gray(data.astronaut())

psf = np.ones((5, 5)) / 25
astro_conv = conv2(astro, psf, 'same')

# Add Noise to Image
astro_noisy = astro_conv.copy() + ((rng.poisson(lam=25, size=astro.shape) - 10) / 255.0)

# Restore Image using Richardson-Lucy algorithm
#deconvolved_RL = restoration.richardson_lucy(astro_noisy, psf, num_iter=30)
astro_noisy_3d = np.expand_dims(astro_noisy, -1)
psf_init = generate_initial_psf(astro_noisy_3d, psf_shape=(64,64,1))
result = np.copy(astro_noisy_3d)
result, result_psf, _ = RL_deconv_blind( torch.tensor(astro_noisy_3d), torch.tensor(result), torch.tensor(psf_init), iterations=50)
deconvoluted_RL = result[:,:,0]


fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 5))
plt.gray()

for a in (ax[0], ax[1], ax[2]):
    a.axis('off')

ax[0].imshow(astro)
ax[0].set_title('Original Data')

ax[1].imshow(astro_noisy)
ax[1].set_title('Noisy data')

ax[2].imshow(deconvoluted_RL, vmin=astro_noisy.min(), vmax=astro_noisy.max())
ax[2].set_title('Restoration using\nRichardson-Lucy')


fig.subplots_adjust(wspace=0.02, hspace=0.2, top=0.9, bottom=0.05, left=0, right=1)

plt.savefig("result.png")









