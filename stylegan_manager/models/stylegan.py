import PIL.Image
import numpy as np
import torch
import legacy
import dnnlib

class StyleGANGenerator:
    """
    A wrapper for a pre-trained StyleGAN3 generator network.

    This class loads a pre-trained model from a .pkl file and provides a simple
    interface to generate images from latent vectors.
    """
    def __init__(self, network_pkl: str, precision: str = 'fp32', device: torch.device = None):
        """Initializes the generator.

        Args:
            network_pkl (str): Path to the StyleGAN network pickle file.
            precision (str): The inference precision to use, 'fp16' or 'fp32'.
            device (torch.device): The PyTorch device to run on (e.g., torch.device('cuda')).
                                   Defaults to 'cuda' if available, otherwise 'cpu'.
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.precision = precision
        self.G = self._load_model(network_pkl)
        self.z_dim = self.G.z_dim
        self.c_dim = self.G.c_dim

    def _load_model(self, network_pkl: str):
        """Loads and prepares the generator model."""
        print(f"Loading networks from '{network_pkl}'...")
        with dnnlib.util.open_url(network_pkl) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(self.device)

        G.eval()

        if self.precision == 'fp16':
            print("Converting model to FP16...")
            G.half()
            # Set custom flags for full FP16 compatibility
            for module in G.modules():
                if hasattr(module, 'use_fp16'):
                    module.use_fp16 = True

        print(f"Model loaded and running on {self.device} with {self.precision} precision.")
        return G

    @torch.no_grad()
    def generate_image(self, z: np.ndarray, label: int = None, truncation_psi: float = 1.0, noise_mode: str = 'const') -> PIL.Image.Image:
        """Generates an image from a given latent vector z.

        Args:
            z (np.ndarray): The input latent vector, should have shape [z_dim] or [1, z_dim].
            label (int): Optional class label for conditional models.
            truncation_psi (float): Truncation psi value.
            noise_mode (str): Noise mode ('const', 'random', 'none').

        Returns:
            PIL.Image.Image: The generated image.
        """
        # 1. Input Validation and Preparation
        if not isinstance(z, np.ndarray):
            raise TypeError(f"Input z must be a NumPy array, but got {type(z)}")
        if z.shape not in [(self.z_dim,), (1, self.z_dim)]:
            raise ValueError(f"Input z must have shape ({self.z_dim},) or (1, {self.z_dim}), but got {z.shape}")

        z = z.reshape(1, self.z_dim)  # Ensure batch dimension

        dtype = torch.float16 if self.precision == 'fp16' else torch.float32
        z_tensor = torch.from_numpy(z).to(self.device, dtype=dtype)

        # Prepare class label tensor
        label_tensor = torch.zeros([1, self.c_dim], device=self.device, dtype=dtype)
        if self.c_dim != 0:
            if label is None:
                raise ValueError("Conditional model requires a class label.")
            label_tensor[:, label] = 1

        # 2. Forward Pass
        img_tensor = self.G(z_tensor, label_tensor, truncation_psi=truncation_psi, noise_mode=noise_mode)

        # 3. Post-Processing
        img_tensor = (img_tensor.float().permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        return PIL.Image.fromarray(img_tensor[0].cpu().numpy(), 'RGB')
