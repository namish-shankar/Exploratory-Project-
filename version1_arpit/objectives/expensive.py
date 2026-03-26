# objectives/expensive.py
from train.trainer import train_model
from train.distill import train_with_distillation
from train.evaluate import evaluate_accuracy as calculate_val_error
from utils.logger import get_logger

logger = get_logger("expensive_obj", logfile="logs/expensive.log")

def evaluate_accuracy(model, train_loader, val_loader, device="cpu", epochs=1, teacher_model=None):
    """
    The main expensive objective pipeline.
    Routes to Knowledge Distillation if a teacher is provided.
    """
    logger.info("Starting expensive evaluation pipeline.")
    
    # 1. Mutate the state (Train)
    if teacher_model is not None:
        train_with_distillation(
            student_model=model, 
            teacher_model=teacher_model, 
            train_loader=train_loader, 
            device=device, 
            epochs=epochs
        )
    else:
        train_model(model, train_loader, device=device, epochs=epochs)
    
    # 2. Read the state (Evaluate)
    val_error = calculate_val_error(model, val_loader, device=device)
    
    return val_error