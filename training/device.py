import torch

# helper functions for managing PyTorch devices (CUDA vs CPU)

def resolve_device(device_choice: str = "auto", require_cuda: bool = False) -> torch.device:
    cuda_available = torch.cuda.is_available()

    if require_cuda and not cuda_available:
        raise RuntimeError(
            "CUDA was required but PyTorch cannot access it. Install a CUDA-enabled "
            "PyTorch build and confirm with: python -c \"import torch; print(torch.cuda.is_available())\""
        )

    if device_choice == "cuda":
        if not cuda_available:
            raise RuntimeError(
                "CUDA was requested but PyTorch cannot access it. Install a "
                "CUDA-enabled PyTorch build or use automatic CPU fallback."
            )
        return torch.device("cuda")

    if device_choice == "cpu":
        return torch.device("cpu")

    return torch.device("cuda" if cuda_available else "cpu")


def print_device_summary(device: torch.device) -> None:
    print(f"selected_device={device}")
    print(f"torch_cuda_available={torch.cuda.is_available()}")
    print(f"torch_cuda_version={torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"cuda_device_name={torch.cuda.get_device_name(0)}")
