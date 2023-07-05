import torch
import pytest

@pytest.fixture(autouse=True)
def pytorch_random_seed():
    torch.manual_seed(0)
