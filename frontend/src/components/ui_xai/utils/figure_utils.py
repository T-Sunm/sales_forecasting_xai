"""
Figure utilities for saving and managing matplotlib figures
"""

import os
import tempfile
from pathlib import Path


def save_figure_to_temp(fig, prefix="plot"):
    """
    Save matplotlib figure to temporary file.
    
    Args:
        fig: Matplotlib figure object
        prefix: Filename prefix
    
    Returns:
        Path to saved temporary file
    """
    temp_dir = tempfile.gettempdir()
    temp_path = Path(temp_dir) / f"{prefix}_{os.getpid()}.png"
    fig.savefig(temp_path, bbox_inches='tight', dpi=100)
    return str(temp_path)


def cleanup_temp_file(file_path):
    """
    Safely remove a temporary file.
    
    Args:
        file_path: Path to file to remove
    
    Returns:
        bool: True if removed successfully, False otherwise
    """
    try:
        os.remove(file_path)
        return True
    except Exception:
        return False
