# objectives/cheap.py
import torch
from utils.logger import get_logger

logger = get_logger("cheap_obj", logfile="logs/cheap.log")

def count_parameters(model: torch.nn.Module) -> int:
    """
    Counts the total number of trainable parameters.
    """
    if model is None:
        return 0
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def estimate_flops(model: torch.nn.Module, input_size=(1, 3, 32, 32)) -> float:
    """
    Estimates MACs/FLOPs using the `thop` library.
    """
    if model is None:
        return 0.0

    try:
        from thop import profile
    except ImportError:
        logger.warning("The 'thop' library is missing. FLOPs reported as 0.0. "
                       "Install via: pip install thop")
        return 0.0

    device = next(model.parameters()).device
    dummy_input = torch.randn(*input_size).to(device)
    
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            macs, _ = profile(model, inputs=(dummy_input,), verbose=False)
        except Exception as e:
            logger.error("FLOP estimation failed: %s", str(e))
            return 0.0
            
    return float(macs)