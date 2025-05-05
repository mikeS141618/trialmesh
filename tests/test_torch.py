# tests/test_torch.py

import pytest

def test_torch_environment():
    import torch

    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch built with CUDA: {torch.version.cuda}")
    try:
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
    except Exception:
        print("cuDNN version: unavailable")

    cuda = torch.cuda.is_available()
    print(f"CUDA available: {cuda}")

    if cuda:
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs: {num_gpus}")
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}")
            print(f"    Total memory: {props.total_memory // (1024 ** 3)} GB")
            print(f"    Compute capability: {props.major}.{props.minor}")
            pci_bus_id = getattr(props, "pci_bus_id", "unavailable")
            print(f"    PCI Bus ID: {pci_bus_id}")

            # Run a simple tensor operation on this device
            device = torch.device(f"cuda:{i}")
            x = torch.randn(1000, 1000, device=device)
            y = x @ x
            result_sum = y.sum().item()
            print(f"    Tensor op (1000x1000 matmul) result sum: {result_sum:.4f}")
            assert not torch.isnan(y).any()
    else:
        print("No GPUs detected. Running CPU-only checks.")
        # CPU tensor operation
        x = torch.randn(1000, 1000)
        y = x @ x
        result_sum = y.sum().item()
        print(f"CPU tensor op (1000x1000 matmul) result sum: {result_sum:.4f}")
        assert not torch.isnan(y).any()

def test_tensor_operations():
    import torch
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])
    c = a + b
    assert torch.allclose(c, torch.tensor([5.0, 7.0, 9.0]))
    d = a * b
    assert torch.allclose(d, torch.tensor([4.0, 10.0, 18.0]))

def test_simple_nn_forward():
    import torch
    import torch.nn as nn
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2)
    )
    x = torch.randn(1, 4)
    y = model(x)
    assert y.shape == (1, 2)
    assert not torch.isnan(y).any()