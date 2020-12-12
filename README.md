## JACK visualiser 

The script `jack_visualiser.py` should allow you to visualise a spectrogram
of the output of a JACK client using a trained StyleGAN network of your
choosing.


### Setup 
First, load the `stylegan2` repository submodule.

```.sh
git submodule init
```

Now you need to install the dependencies for StyleGAN 2. I've successfully
set StyleGAN 2 up on Ubuntu 18.

You need to install the following on your system:
 - Nvidia drivers (>= 410.48)
 - CUDA 10.0 toolkit
 - cuDNN 7.5
 - Python 3.6

See the [StyleGAN2 readme](https://github.com/NVlabs/stylegan2#requirements)
for more details.

To display the image, you'll also need a system-level package for Tkinter:

```
sudo apt install python3-tk
```

Once you have the system-level requirements, you'll need some pip packages:

```.sh
pip install -r requirements.txt
```

### Usage
The script is a Click CLI and it's `--help` output is below. 
```
Usage: visualize.py [OPTIONS] JACK_CLIENT_NAME NETWORK_PKL
Options:
  -p, --periodogram [simple|welch]
                                  Algorithm used to compute spectral density.
                                  [required]
  -s, --seeds TEXT                Comma-separated list or dash-separated range
                                  of network input seeds. These seeds are
                                  mapped onto frequency "bins" for audio. The
                                  first seed is mapped onto the lowest
                                  frequency bin and the last onto the highest.
                                  Generally, the more seeds you specify, the
                                  more detailed your visualisation. Unless
                                  you're working with exceptionally high-
                                  frequency audio, you're mainly going to see
                                  images based on first few seeds.
  --truncation-psi FLOAT          Psi value used for StyleGAN's truncation
                                  trick. Lower values result in images closer
                                  to an "average" of what the network has
                                  learned; these images tend to be better
                                  quality but less varied. Conversely, a
                                  higher value leads to more varied but
                                  potentially lower quality images.
  --samples-per-image INTEGER     How many samples of audio to use for each
                                  generated image. With the default value of
                                  2048 and sample rate of 48000 you should see
                                  around 23.4 FPS. You may need to make this
                                  value higher to reduce latency.This value
                                  should ideally be kept to multiples of 1024,
                                  or whatever the size audio frames coming
                                  from JACK happens to be.
  --sample-rate INTEGER           JACK sample rate.
  --help                          Show this message and exit.
```

### Examples
You'll need to supply two positional arguments: `JACK_CLIENT_NAME` and `NETWORK_PKL`.

In all examples below, the jack client name `SuperCollider` is used. 
This works if Supercollider's server is running and outputting audio to JACK. 
If you're outputting to JACK with something else you can check the client name
using one of the control applications listed
[here](https://jackaudio.org/applications/).

The `stylegan2-cat-config-a.pkl` network pkl is used in the first two examples.
This produces small images and is based on a simple architecture. It is fast but boring. 
If you want something more sophisticated, try one of the `config-e` or `config-f` networks. 
You can of course train your own.

In order for the imports within the `stylegan2` submodule to work,
you'll need to execute this command once before running the script.
```.sh
export PYTHONPATH=stylegan2
```
Along with the two positional arguments, 
you'll need to specify a periodogram algorithm using the `--periodogram` option.

#### Simple periodogram
This is Scipy's basic algorithm which doesn't do much more than apply the fast Fourier transform to the audio.
The resulting periodogram tends to be quite big, so the script divides it evenly between the seeds.

This will output images for every 4096 samples from Supercollider 
based on the first 128 seeds of the `stylegan2-cat-config-a.pkl` network.
```.bash
python visualize.py SuperCollider gdrive:networks/stylegan2-cat-config-a.pkl \
    --periodogram simple --samples-per-image 4096 --seeds 0-127
```

You'll get console output like this initially:
```
206848 samples left in buffer
Running simple periodogram over 4096 samples...
Raw periodogram size: 2049
Split and summed periodogram size: 128
Weights:
[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0.]
```
If the number of samples in the buffer keeps increasing, 
you need to increase `--samples-per-image`.
Once there are consistently a small number of samples in the buffer (ideally 0), 
you should get reasonably low-latency visualisation of your output.
Every time a new image is displayed, you should see something like the following:
```
0 samples left in buffer
Running simple periodogram over 4096 samples...
Raw periodogram size: 2049
Split and summed periodogram size: 128
Weights:
[3.08587722e+17 1.51684887e+16 5.47313126e+15 2.34306398e+15
 1.31886061e+15 9.98901811e+14 6.94232841e+14 5.56384289e+14
 4.48976489e+14 3.32476708e+14 2.65123514e+14 2.30043740e+14
 2.20722348e+14 1.66958594e+14 1.16301184e+14 1.19160621e+14
 1.12901880e+14 1.03223888e+14 9.61351211e+13 8.01935375e+13
 6.96241569e+13 6.32963803e+13 6.71914561e+13 6.29972429e+13
 4.63712250e+13 4.27720984e+13 4.24981796e+13 4.20905908e+13
 4.32242408e+13 3.71053733e+13 3.25540326e+13 3.02008509e+13
 3.03842946e+13 3.12167790e+13 2.65788427e+13 2.44064459e+13
 2.24887225e+13 2.07910341e+13 2.37495026e+13 2.24574093e+13
 1.94198089e+13 1.86993484e+13 1.84240379e+13 1.83315313e+13
 1.58749542e+13 1.54010771e+13 1.54316151e+13 1.32574690e+13
 1.45218821e+13 1.51715929e+13 1.33291654e+13 1.27263128e+13
 1.28034633e+13 1.33357602e+13 1.16161658e+13 1.00832810e+13
 1.08303869e+13 1.03263240e+13 1.07008711e+13 1.11954229e+13
 9.88675539e+12 9.24806093e+12 8.95850045e+12 9.82104138e+12
 1.00073123e+13 8.30769851e+12 7.78378773e+12 7.71519653e+12
 8.39114674e+12 9.06557467e+12 7.86883442e+12 7.25354891e+12
 6.88169894e+12 7.04596764e+12 8.03502979e+12 7.50562076e+12
 6.67421410e+12 6.03426031e+12 6.28427399e+12 7.42257995e+12
 6.83457958e+12 6.14175408e+12 6.03828287e+12 5.90019630e+12
 6.48061937e+12 6.25258726e+12 5.84355398e+12 5.59410057e+12
 5.30359279e+12 6.07940121e+12 6.11300887e+12 5.48554342e+12
 5.38140489e+12 5.37504808e+12 5.96223295e+12 5.58131742e+12
 4.72766399e+12 4.96177816e+12 5.09992697e+12 5.45553590e+12
 5.50313050e+12 4.93143550e+12 4.72041387e+12 4.58586623e+12
 5.37512075e+12 5.64926863e+12 4.42887849e+12 4.15853477e+12
 4.58538606e+12 5.11741124e+12 5.28090681e+12 4.64769235e+12
 4.39952872e+12 4.17034362e+12 4.61645466e+12 5.43344447e+12
 4.74289798e+12 4.15601029e+12 4.18955673e+12 4.54976567e+12
 5.13123089e+12 4.77105195e+12 4.41691842e+12 4.33055281e+12
 4.48278386e+12 5.04787991e+12 4.58219928e+12 4.25994836e+12]
```

#### Welch periodogram
The Welch periodogram uses overlapping frequency bins, which may look smoother.
This is currently more buggy than the simple periodogram 
and you may need to tweak the number of seeds to get it to work.

This example will output images based on 32 overlapping frequency bins for some custom seed numbers.
```.bash
python visualize.py SuperCollider gdrive:networks/stylegan2-cat-config-a.pkl \
    --periodogram welch --samples-per-image 4096 \
    --seeds '0,1,1,2,3,5,8,13,21,34,55,89,144,233,377,610,
             987,1597,2584,4181,6765,10946,17711,28657,46368,75025,
             121393,196418,17811,514229,832040,1346269'
```
#### Using your own network
Just specify a path to a local `.pkl` file rather than using the `gdrive` prefix.

This example won't work unless you have the network, but should serve to illustrate:
```.bash
python visualize.py SuperCollider ~/ai/networks/glitch-009084.pkl \
    --samples-per-image 5120 -p simple \
    --seeds '1137,1028,1139,1285,1776,2051,936,915,848,838,812,791,
             1,2,3,4,5,6,7,8,9,10,11,12,13,15,14,16,17,18,19,20,21,
             22,23,24,25,26,27,28,29,30,31,32,33,34'
```
