# stylegan-server

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
# define a random walk and load it
curl -X POST http://localhost:5000/start_random_walk

# fetch the next image in the walk (PNG binary response)
curl -o frame.png http://localhost:5000/next_image
```

Visit `http://localhost:5000/gallery` to browse saved walks and images. The
gallery interface also allows creating curated walks by interpolating between
selected keyframes.

## Output Directory Structure

Generated images are now stored in a nested layout under the configured
`outdir`:

```
<outdir>/<instance_id>/<rand8>/<filename>
```

`instance_id` identifies the running server instance (it can be overridden via
the `INSTANCE_ID` environment variable) and `rand8` is a random eight-digit
folder created for each rendered image. The relative path is stored in the
database and used by the gallery when serving images.

