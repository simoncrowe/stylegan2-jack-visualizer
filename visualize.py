from collections import deque
import re
import struct
import sys
import time
from typing import Iterable, Iterator, List

import click
import jack
import numpy as np
from PIL import Image, ImageTk
from scipy import signal
import tkinter as tk
import yaml

# PYTHONPATH needs to be updated to allow stylegan2 imports
sys.path.append("stylegan2")
from stylegan2 import dnnlib
import stylegan2.dnnlib.tflib as tflib
from stylegan2 import pretrained_networks


def _cast_and_unpack(seeds: Iterable) -> Iterator[int]:
    """Casts items to integers, unpacking ranges like "10-15"."""
    range_re = re.compile(r'^(\d+)-(\d+)$')

    for seed in seeds:
        try:
            yield int(seed)
        except ValueError:
            match = range_re.match(seed)
            if match:
                range_from = int(match.group(1))
                range_to = int(match.group(2)) + 1
                yield from range(range_from, range_to)
            else:
                raise ValueError(
                    f"{seed} is not a valid integer or integer range."
                )


def _parse_seed_list(seeds: str) -> List[int]:
    """Parse a comma-separated list of integers and integer ranges."""
    raw = (num.strip() for num in seeds.split(",") if num)
    return list(_cast_and_unpack(raw))


def _unpack_bytes(byte_string, int_count):
    """Unpacks a byte string to 32 bit little endian integers."""
    return struct.unpack(f"<{int_count}l", byte_string)


def aggressive_array_split(array, parts):
    """Potentially exclude some indeces to split an array of any shape."""
    end_index = len(array) - (len(array) % parts)
    return np.split(array[:end_index], parts)


def simple_periodogram(samples, sample_rate, *args, **kwargs):
    print(f"Running simple periodogram over {samples.size} samples...")
    _, spectral_density = signal.periodogram(samples,
                                             sample_rate,
                                             return_onesided=True)
    return spectral_density


def welch_periodogram(samples, sample_rate, bin_count):
    print(f"Running welch periodogram over {samples.size} samples...")
    segment_size = int(samples.size / bin_count) / 2
    _, spectral_density = signal.welch(
        samples,
        sample_rate,
        nperseg=segment_size,
        return_onesided=True
    )
    return spectral_density


PERIODOGRAM_FUNCTION_MAP = {
    "simple": simple_periodogram,
    "welch": welch_periodogram,
}


def generate_periodogram_from_audio(periodogram_function,
                                    audio_buffer,
                                    samples_per_image,
                                    sample_rate,
                                    bin_count):
    while True:
        if len(audio_buffer) < samples_per_image:
            time.sleep(0.001)
            continue

        audio = np.array(
            [audio_buffer.pop() for _ in range(samples_per_image)]
        )
        audio_mono = np.sum(audio, axis=1)
        print(f"{len(audio_buffer)} samples left in buffer")

        periodogram = periodogram_function(audio_mono, sample_rate, bin_count)
        print(f"Raw periodogram size: {periodogram.size}")

        if periodogram.size == bin_count:
            yield periodogram
        if periodogram.size > bin_count:
            periodogram_split = aggressive_array_split(periodogram, bin_count)
            periodogram_summed = np.sum(periodogram_split, axis=1)
            print(
                "Split and summed periodogram size: "
                f"{periodogram_summed.size}"
            )
            assert periodogram_summed.size == bin_count
            yield periodogram_summed
        else:
            raise ValueError(
                "Too many seeds! "
                "Please specify {peridogram.size} or fewer seeds, "
                "or increase samples per image."
            )


def generate_images(network_pkl,
                    seeds,
                    truncation_psi,
                    periodogram_generator,
                    minibatch_size=4):

    print(f"Loading networks from {network_pkl}...")
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    average_weights = Gs.get_var("dlatent_avg")  # [component]
    print(f"Average weights shape: {average_weights.shape}")

    Gs_syn_kwargs = dnnlib.EasyDict()
    Gs_syn_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8,
                                          nchw_to_nhwc=True)
    Gs_syn_kwargs.randomize_noise = False
    Gs_syn_kwargs.minibatch_size = minibatch_size

    # [minibatch, component]
    all_input_noise = np.stack(
        [np.random.RandomState(seed).randn(*Gs.input_shape[1:])
         for seed in seeds]
    )

    for weights in periodogram_generator:
        if len(weights) > 100:
            print(f"First 100 of {len(weights)} weights:\n{weights[:100]}\n")
        else:
            print(f"Weights:\n{weights}\n")

        reshaped_weights = weights.reshape(len(all_input_noise), 1)
        weighted_noise = reshaped_weights * all_input_noise
        weighted_sum = np.sum(weighted_noise, axis=0)
        noise_norm = np.linalg.norm(weighted_sum, ord=2, keepdims=True)
        normalised_noise = weighted_sum / noise_norm
        normalised_noise_matrix = np.array([normalised_noise])

        # [minibatch, layer, component]
        layers = Gs.components.mapping.run(normalised_noise_matrix, None)
        # [minibatch, layer, component]
        layers = average_weights + (layers - average_weights) * truncation_psi
        # [minibatch, height, width, channel]
        images = Gs.components.synthesis.run(layers, **Gs_syn_kwargs)
        yield images[0]


@click.command()
@click.argument("jack_client_name")
@click.argument("network_pkl")
@click.option(
    "-p",
    "--periodogram",
    type=click.Choice(PERIODOGRAM_FUNCTION_MAP.keys()),
    required=True,
    help="Algorithm used to compute spectral density."
)
@click.option(
    "-s",
    "--seeds-list",
    type=str,
    help=(
        "Comma-separated list of input seeds or seed ranges.\n"
        "e.g. '1, 3, 5, 2-44, 100, 355, 512-533'.\n"
        "These seeds are mapped onto frequency 'bins' for audio. "
        "The first seed is mapped onto the lowest frequency bin "
        "and the last onto the highest. "
        "Generally, the more seeds you specify, "
        "the more detailed your visualisation. "
        "Unless you're working with exceptionally high-frequency audio, "
        "you're mainly going to see images based on first few seeds. "
    )
)
@click.option(
    "-f",
    "--seeds-file",
    type=str,
    help=(
        "A YAML file containing named lists of seeds and the starting name. "
        "See the README for details of the format."
    )
)
@click.option(
    "--truncation-psi",
    default=0.75,
    help=(
        "Psi value used for StyleGAN's truncation trick. "
        "Lower values result in images closer to an 'average' "
        "of what the network has learned; "
        "these images tend to be better quality but less varied. "
        "Conversely, a higher value leads to more varied "
        "but potentially lower quality images."
    )
)
@click.option(
   "--samples-per-image",
   default=2048,
   help=(
       "How many samples of audio to use for each generated image. "
       "With the default value of 2048 and sample rate of 48000 you "
       "should see around 23.4 FPS."
       "This value should ideally be kept to multiples the size audio frames "
       "coming from JACK (usually 1024)."
       "You may need to increase this value to reduce latency."
   )
)
@click.option("--sample-rate", default=48000, help="JACK sample rate.")
def visualise(jack_client_name,
              network_pkl,
              periodogram,
              seeds_list,
              seeds_file,
              truncation_psi,
              samples_per_image,
              sample_rate):
    if bool(seeds_list) == bool(seeds_file):
        raise ValueError(
            "Either --seeds-list, or --seeds-file must be specified "
            "(not both)."
        )

    if seeds_list:
        # Hackily ensuring the same data structure
        seeds = {"seeds list": _parse_seed_list(seeds_list)}
        config = {}
    else:
        with open(seeds_file, "r") as fileobj:
            config = yaml.load(fileobj)
        seeds = {
            str(name): list(_cast_and_unpack(seeds_list))
            for name, seeds_list in config["seeds"].items()
        }

    periodogram_function = PERIODOGRAM_FUNCTION_MAP[periodogram]

    client = jack.Client("StyleGan Visualiser")
    input_one = client.inports.register("in_1")
    input_two = client.inports.register("in_2")
    external_output_one = client.get_port_by_name(f"{jack_client_name}:out_1")
    external_output_two = client.get_port_by_name(f"{jack_client_name}:out_2")

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
    client.connect(external_output_two, input_two)

    root = tk.Tk()
    root.title("StyleGAN 2 JACK Visualizer")
    panel = tk.Label(root)
    panel.configure(bg="black")
    panel.pack(side="bottom", fill="both", expand="yes")

    seeds_name = str(config.get("starting_name", next(iter(seeds.keys()))))
    reset = False
    chars = []

    def keypress(event):
        """Listened asychonously for keypresses to change seeds."""

        if event.keysym in ("KP_Enter", "Return"):
            nonlocal chars
            name = "".join(chars)

            if name in seeds.keys():
                nonlocal seeds_name
                seeds_name = name
                nonlocal reset
                reset = True
                print("\n" * 100,
                      f"Changed seeds to {name}!\n")
            else:
                print("\n" * 100,
                      f"ERROR: seeds with name '{name}' not found!\n")
            chars = []

        elif event.keysym == "BackSpace":
            chars = chars[:-1]

        elif event.char:
            chars.append(event.char)

    root.bind("<KeyPress>", keypress)

    while True:
        selected_seeds = seeds[seeds_name]
        print(f"Selected seeds '{seeds_name}': {selected_seeds}")
        periodogram_generator = generate_periodogram_from_audio(
            periodogram_function,
            raw_audio,
            samples_per_image,
            sample_rate,
            len(selected_seeds)
        )
        image_generator = generate_images(network_pkl,
                                          selected_seeds,
                                          truncation_psi,
                                          periodogram_generator)
        reset = False

        for image_array in image_generator:
            image = Image.fromarray(image_array, "RGB")
            gui_image = ImageTk.PhotoImage(image)

            panel.configure(image=gui_image)

            # To prevent GC getting rid of image?
            panel.image = gui_image

            root.update()

            if reset:
                break


if __name__ == "__main__":
    visualise()
