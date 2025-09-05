import numpy as np
import pytest


@pytest.fixture
def sample_rate():
    return 22050


@pytest.fixture
def tone_440(sample_rate):
    # Generate a 440 Hz tone for 0.5 seconds
    # amplitude = 0.2
    duration = 0.5
    amplitude = 0.2
    t = np.arange(int(sample_rate * duration)) / sample_rate
    return amplitude * np.sin(2 * np.pi * 440 * t)


@pytest.fixture
def white_noise(sample_rate):
    # Generate white noise for 0.5 seconds
    # Use a fixed random seed for reproducibility
    rng = np.random.default_rng(123)
    duration = 0.5
    return 0.05 * rng.standard_normal(int(sample_rate * duration))
