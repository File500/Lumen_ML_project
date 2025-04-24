import torch
import gc

def clear_gpu_cache(device=None):
    """
    Clear GPU cache and perform garbage collection.
    
    Args:
        device (torch.device, optional): Specific device to synchronize. Defaults to None.
    """
    # Perform garbage collection
    gc.collect()
    
    # Clear CUDA cache if CUDA is available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
        # Optional: Synchronize the selected device if specified
        if device is not None:
            try:
                with torch.cuda.device(device):
                    torch.cuda.synchronize()
            except Exception as e:
                print(f"Error synchronizing device: {e}")

def get_device(gpu_device=None):
    """
    Get the appropriate computing device.
    
    Args:
        gpu_device (int, optional): Specific GPU device to use. Defaults to None.
    
    Returns:
        torch.device: Selected computing device
    """
    # Check if CUDA is available
    if torch.cuda.is_available():
        # If a specific GPU device is provided and is valid
        if gpu_device is not None and gpu_device < torch.cuda.device_count():
            device = torch.device(f"cuda:{gpu_device}")
            print(f"Using GPU {gpu_device}: {torch.cuda.get_device_name(device)}")
            return device
        
        # Default to first available GPU
        device = torch.device("cuda:0")
        print(f"Using GPU 0: {torch.cuda.get_device_name(device)}")
        return device
    
    # Fallback to CPU
    print("Using CPU")
    return torch.device("cpu")