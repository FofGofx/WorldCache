import random
import numpy as np
import torch


def set_seed(seed: int = 3047) -> None:
    """
    Set random seed for reproducibility (without forcing deterministic algorithms).
    
    This function sets seeds for Python random, NumPy, and PyTorch to improve
    reproducibility. However, it does not enable torch.use_deterministic_algorithms(True)
    to avoid CuBLAS compatibility issues and potential changes in computation paths.
    
    Args:
        seed: Random seed value. Default is 3047.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Note: We don't enable torch.use_deterministic_algorithms(True) 
    # to avoid CuBLAS compatibility issues and potential changes in computation paths
    # that could affect metric return types


def print_banner(text: str) -> None:
    try:
        from pyfiglet import Figlet
        print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
        fig = Figlet(font='slant')
        ascii_art = fig.renderText("WORLDSCORE")
        print(ascii_art)
        fig = Figlet(font='small')
        ascii_art = fig.renderText(text)
        print(ascii_art)
        print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
    except ImportError:
        print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
        print("WORLDSCORE")
        print(text)
        print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")