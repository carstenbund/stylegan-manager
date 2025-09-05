# stylegan-manager

Flask service that streams interpolated images from a pre‑trained
[StyleGAN3](https://github.com/NVlabs/stylegan3) network.

## Setup

Install required dependencies (PyTorch, StyleGAN3 utils and Flask):

```bash
pip install flask torch torchvision pillow numpy opencv-python
```

The server downloads a default StyleGAN3 network on first run. To use a
different network, set the `NETWORK_PKL` environment variable to point to a
`.pkl` file or URL.

## Usage

Start the server:

```bash
python stylegan_server.py
```

Generate images by performing a latent walk:

```bash
# define a random walk with 120 steps in range [-1.5, 1.5] and load it
curl -X POST -H "Content-Type: application/json" \
     -d '{"steps":120,"keyframes":4,"extent":1.5}' \
     http://localhost:5000/start_random_walk

# fetch the next image in the walk (PNG binary response)
curl -o frame.png http://localhost:5000/next_image
```

Latent vectors are generated via the ``sample_latents`` utility, which draws
bounded samples from a standard normal distribution and optionally mixes in a
Sobol low-discrepancy sequence.  The ``/start_random_walk`` endpoint accepts
``steps`` (or ``n_vectors``) to control how many vectors are generated and an
``extent`` value that limits sampling to ``[-extent, extent]``.

Visit `http://localhost:5000/gallery` to browse saved walks and images. The
gallery interface also allows creating curated walks by interpolating between
selected keyframes.

### Gallery Filters

The gallery view supports optional query parameters to control which images are
shown. Filters are parsed by the server and mirrored on the client so that
additional types can be added with minimal changes. Currently the following
filter is available:

* `liked=1` – only display images that have been marked as liked.

Future filters can be introduced by extending the server's filter parser and
adding corresponding UI elements in the gallery template.

## Library usage

The project now exposes a lightweight Python package, ``stylegan_manager``,
that provides modular building blocks for exploring StyleGAN models. The
package includes:

* ``Walk`` – abstract base class for latent walks.
* ``RandomWalk`` – quick exploratory walk that generates random frames.
* ``CustomWalk`` – interpolation-based walk with optional video rendering.
* ``VideoManager`` – tracks curated walks and their render status.

These components can be imported directly from ``stylegan_manager`` for use in
other applications.

## Output Directory Structure

Generated images are stored in a layout where each walk has its own folder:

```
<outdir>/<walk_id>/<filename>
```

`walk_id` is the identifier of the walk in the database. Filenames only need to
be unique within a given walk, allowing the same names to appear in different
walk folders. The relative path is stored in the database and used by the
gallery when serving images.

