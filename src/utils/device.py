import torch


def get_device():
    """
    Return CUDA device if available, else CPU.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def print_device_info():
    device = get_device()
    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version (PyTorch): {torch.version.cuda}")