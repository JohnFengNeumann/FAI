from transformers import Pipeline
import torch

class FAIPipeline(Pipeline):
    """
    Huggingface Pipeline wrapper for Fast Audio Inference.
    """
    def __init__(self, model, model_type):
        self.model = self.load_model(model) if isinstance(model, str) else model
    
    def load_model(self, model_path):
        self.model = MODEL['model_type']
        self.model = torch.load(model_path)
        
    