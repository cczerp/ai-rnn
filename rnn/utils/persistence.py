"""
Model Persistence Category
Handles saving and loading of trained models.
"""

import numpy as np
import json
import os
from typing import Dict


class ModelPersistence:
    """
    Manages model saving and loading operations.
    
    Categories of persistence:
    - Model saving: Save trained parameters to disk
    - Model loading: Load parameters from disk
    - Checkpoint management: Save periodic checkpoints
    - Export/Import: Support different formats
    """
    
    def __init__(self, model_dir: str = './models'):
        """
        Initialize model persistence manager.
        
        Args:
            model_dir: Directory to save/load models
        """
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
    def save_model(self, model, model_name: str, 
                   metadata: Dict = None) -> str:
        """
        Save model parameters and metadata.
        
        Args:
            model: Model to save
            model_name: Name for the saved model
            metadata: Additional metadata to save
            
        Returns:
            Path to saved model
        """
        model_path = os.path.join(self.model_dir, f"{model_name}.npz")
        metadata_path = os.path.join(self.model_dir, f"{model_name}_meta.json")
        
        # Get model parameters
        params = model.get_parameters()
        
        # Save parameters
        np.savez(model_path, **params)
        
        # Save metadata
        if metadata is None:
            metadata = {}
        
        metadata['model_name'] = model_name
        metadata['model_type'] = 'RNN'
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return model_path
    
    def load_model(self, model, model_name: str) -> Dict:
        """
        Load model parameters and metadata.
        
        Args:
            model: Model to load parameters into
            model_name: Name of the saved model
            
        Returns:
            Loaded metadata
        """
        model_path = os.path.join(self.model_dir, f"{model_name}.npz")
        metadata_path = os.path.join(self.model_dir, f"{model_name}_meta.json")
        
        # Check if files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load parameters
        loaded_data = np.load(model_path)
        params = {key: loaded_data[key] for key in loaded_data.files}
        
        # Set model parameters
        model.set_parameters(params)
        
        # Load metadata if exists
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        return metadata
    
    def save_checkpoint(self, model, epoch: int, 
                       loss: float, checkpoint_name: str = 'checkpoint') -> str:
        """
        Save a training checkpoint.
        
        Args:
            model: Model to checkpoint
            epoch: Current epoch number
            loss: Current loss value
            checkpoint_name: Base name for checkpoint
            
        Returns:
            Path to saved checkpoint
        """
        metadata = {
            'epoch': epoch,
            'loss': float(loss),
            'checkpoint': True
        }
        
        checkpoint_full_name = f"{checkpoint_name}_epoch_{epoch}"
        return self.save_model(model, checkpoint_full_name, metadata)
    
    def list_models(self) -> list:
        """
        List all saved models in the model directory.
        
        Returns:
            List of model names
        """
        models = []
        for file in os.listdir(self.model_dir):
            if file.endswith('.npz'):
                model_name = file[:-4]  # Remove .npz extension
                models.append(model_name)
        return sorted(models)
    
    def delete_model(self, model_name: str):
        """
        Delete a saved model.
        
        Args:
            model_name: Name of model to delete
        """
        model_path = os.path.join(self.model_dir, f"{model_name}.npz")
        metadata_path = os.path.join(self.model_dir, f"{model_name}_meta.json")
        
        if os.path.exists(model_path):
            os.remove(model_path)
        
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
    
    def export_to_dict(self, model) -> Dict:
        """
        Export model to dictionary format.
        
        Args:
            model: Model to export
            
        Returns:
            Dictionary containing model parameters
        """
        params = model.get_parameters()
        
        # Convert numpy arrays to lists for JSON serialization
        exportable = {}
        for key, value in params.items():
            if isinstance(value, np.ndarray):
                exportable[key] = value.tolist()
            else:
                exportable[key] = value
        
        return exportable
    
    def import_from_dict(self, model, params_dict: Dict):
        """
        Import model from dictionary format.
        
        Args:
            model: Model to import into
            params_dict: Dictionary containing model parameters
        """
        params = {}
        for key, value in params_dict.items():
            if isinstance(value, list):
                params[key] = np.array(value)
            else:
                params[key] = value
        
        model.set_parameters(params)
