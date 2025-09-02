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
# start a new interpolation sequence
curl -X POST http://localhost:5000/start

# fetch the next image (PNG binary response)
curl -o frame.png http://localhost:5000/next

# retrieve the latent vector used for the last image
curl http://localhost:5000/vector
```

Repeated calls to `/next` continue the interpolation. Restart the sequence by
calling `/start` again.

