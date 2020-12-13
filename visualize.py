from collections import deque
import re
import struct
import sys
import time

import click
import jack
import numpy as np
from PIL import Image, ImageTk
from scipy import signal
import tkinter as tk

from stylegan2 import dnnlib
import stylegan2.dnnlib.tflib as tflib
from stylegan2 import pretrained_networks


def _parse_num_range(int_list_or_range):
    """
    Accept either a comma separated list of numbers 'a,b,c'
    or a range 'a-c' and return as a list of ints.
    """
    range_re = re.compile(r'^(\d+)-(\d+)$')
    match = range_re.match(int_list_or_range)
    if match:
        return list(range(int(match.group(1)), int(match.group(2))+1))

    return [int(num.strip()) for num in int_list_or_range.split(',') if num]


def _unpack_bytes(byte_string, int_count):
    """Unpacks a byte string to 32 bit little endian integers."""
    return struct.unpack(f'<{int_count}l', byte_string)


def aggressive_array_split(array, parts):
    """Potentially exclude some indeces to split an array of any shape."""
    end_index = len(array) - (len(array) % parts)
    return np.split(array[:end_index], parts)


def simple_periodogram(samples, sample_rate, *args, **kwargs):
    print(f'Running simple periodogram over {samples.size} samples...')
    _, spectral_density = signal.periodogram(samples, sample_rate, return_onesided=True)
    return spectral_density


def welch_periodogram(samples, sample_rate, bin_count):
    print(f'Running welch periodogram over {samples.size} samples...')
    segment_size = int(samples.size / bin_count) / 2
    _, spectral_density = signal.welch(
        samples,
        sample_rate,
        nperseg=segment_size,
        return_onesided=True
    )
    return spectral_density


PERIODOGRAM_FUNCTION_MAP = {
    'simple': simple_periodogram,
    'welch': welch_periodogram,
}


def generate_periodogram_from_audio(periodogram_function, audio_buffer, samples_per_image, sample_rate, bin_count):
    while True:
        if len(audio_buffer) < samples_per_image:
            time.sleep(0.001)
            continue

        audio =  np.array([audio_buffer.pop() for _ in range(samples_per_image)])
        audio_mono = np.sum(audio, axis=1)
        print(f'{len(audio_buffer)} samples left in buffer')

        periodogram = periodogram_function(audio_mono, sample_rate, bin_count)
        print(f'Raw periodogram size: {periodogram.size}')

        if periodogram.size == bin_count:
            yield periodogram
        if periodogram.size > bin_count:
            periodogram_split = aggressive_array_split(periodogram, bin_count)
            periodogram_summed = np.sum(periodogram_split, axis=1)
            print(f'Split and summed periodogram size: {periodogram_summed.size}')
            assert periodogram_summed.size == bin_count
            yield  periodogram_summed
        else:
            raise ValueError(
                "Too many seeds! "
                "Please specify {peridogram.size} or fewer seeds, or increase samples per image."
            )


def generate_images(network_pkl, seeds, truncation_psi, periodogram_generator, minibatch_size=4):
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    average_weights = Gs.get_var('dlatent_avg') # [component]
    print(f'Average weights shape: {average_weights.shape}')

    Gs_syn_kwargs = dnnlib.EasyDict()
    Gs_syn_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_syn_kwargs.randomize_noise = False
    Gs_syn_kwargs.minibatch_size = minibatch_size

    all_input_noise = np.stack([np.random.RandomState(seed).randn(*Gs.input_shape[1:]) for seed in seeds]) # [minibatch, component]

    for weights in periodogram_generator:
        print(f'Weights:\n{weights}')
        weighted_noise = weights.reshape(len(all_input_noise), 1) * all_input_noise
        weighted_sum = np.sum(weighted_noise, axis=0)
        normalised_noise = weighted_sum / np.linalg.norm(weighted_sum, ord=2, keepdims=True)
        normalised_noise = np.array([normalised_noise])

        layers = Gs.components.mapping.run(normalised_noise, None) # [minibatch, layer, component]
        layers = average_weights + (layers - average_weights) * truncation_psi # [minibatch, layer, component]
        images = Gs.components.synthesis.run(layers, **Gs_syn_kwargs) # [minibatch, height, width, channel]
        yield images[0]


@click.command()
@click.argument('jack_client_name')
@click.argument('network_pkl')
@click.option(
    '-p',
    '--periodogram',
    type=click.Choice(PERIODOGRAM_FUNCTION_MAP.keys()),
    required=True,
    help='Algorithm used to compute spectral density.'
)
@click.option(
    '-s',
    '--seeds',
    type=str,
    help=(
        'Comma-separated list or dash-separated range of network input seeds. '
        'These seeds are mapped onto frequency "bins" for audio. '
        'The first seed is mapped onto the lowest frequency bin and the last onto the highest. '
        'Generally, the more seeds you specify, the more detailed your visualisation. '
        'Unless you\'re working with exceptionally high-frequency audio, '
        'you\'re mainly going to see images based on first few seeds. '
    )
)
@click.option(
    '-f',
    '--seed-file',
    type=str,
    help=(
        "A YAML file containing named lists of seeds and the starting name."
    )
@click.option(
    '--truncation-psi',
    default=0.75,
    help=(
        'Psi value used for StyleGAN\'s truncation trick. '
        'Lower values result in images closer to an "average" of what the network has learned; '
        'these images tend to be better quality but less varied. '
        'Conversely, a higher value leads to more varied but potentially lower quality images.'
    )
)
@click.option(
   '--samples-per-image',
   default=2048,
   help=(
       'How many samples of audio to use for each generated image. '
       'With the default value of 2048 and sample rate of 48000 you '
       'should see around 23.4 FPS. '
       'You may need to make this value higher to reduce latency.'
       'This value should ideally be kept to multiples of 1024, '
       'or whatever the size audio frames coming from JACK happens to be.'
   )
)
@click.option('--sample-rate', default=48000, help='JACK sample rate.')
def visualise(jack_client_name,
              network_pkl,
              periodogram,
              seeds,
              seeds_file,
              truncation_psi,
              samples_per_image,
              sample_rate):

    if bool(seeds) == bool(seeds_file):
        raise ValueError(
            "Either --seeds, or --seeds-filte must be specified (not both)."
        )

    periodogram_function = PERIODOGRAM_FUNCTION_MAP[periodogram]
    seeds = _parse_num_range(seeds)

    client = jack.Client('StyleGan Visualiser')
    input_one = client.inports.register('in_1')
    input_two = client.inports.register('in_2')
    external_output_one = client.get_port_by_name(f'{jack_client_name}:out_1')
    external_output_two = client.get_port_by_name(f'{jack_client_name}:out_2')

    raw_audio = deque()

    @client.set_process_callback
    def process_audio(frame_count):
        buffer_one = _unpack_bytes(
            input_one.get_buffer()[:], frame_count
        )
        buffer_two = _unpack_bytes(
            input_two.get_buffer()[:], frame_count
        )

        for sample_one, sample_two in zip(buffer_one, buffer_two):
            raw_audio.appendleft((sample_one, sample_two))

    client.activate()
    client.connect(external_output_one, input_one)
    client.connect(external_output_one, input_two)

    root = tk.Tk()
    panel = tk.Label(root)
    panel.configure(bg='black')
    panel.pack(side='bottom', fill='both', expand='yes')

    while True:
        periodogram_generator = generate_periodogram_from_audio(
            periodogram_function,
            raw_audio,
            samples_per_image,
            sample_rate,
            len(seeds)
        )
        image_generator = generate_images(network_pkl,
                                          seeds,
                                          truncation_psi,
                                          periodogram_generator)
        reset = False

        for image_array in image_generator:
            image = Image.fromarray(image_array, 'RGB')
            gui_image = ImageTk.PhotoImage(image)

            panel.configure(image=gui_image)

            # To prevent GC getting rid of image?
            panel.image = gui_image

            root.update()

            if reset:
                break


if __name__ == '__main__':
    visualise()
