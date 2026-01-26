import json
import os
from pathlib import Path
from typing import Dict, List, Union, Any


class ResultSaver:
    
    def __init__(self, output_dir: str = "output/results"):
        self.output_dir = output_dir
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    def save_result(self, 
                   model_name: str,
                   dataset: str,
                   metrics: Dict[str, Any]) -> str:
        filename = f"{model_name}_{dataset}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"✓ Saved: {filepath}")
        return filepath
    
    def save_training_history(self,
                             model_name: str,
                             dataset: str,
                             history: List[Dict[str, Any]]) -> str:
        filename = f"{model_name}_{dataset}_history.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"✓ Saved: {filepath}")
        return filepath
    
    def save_epoch_metrics(self,
                         model_name: str,
                         dataset: str,
                         epoch: int,
                         loss: float,
                         psnr: float,
                         ssim: float,
                         lr: float = None,
                         **kwargs) -> str:
        filename = f"{model_name}_{dataset}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        # Load existing history or create new
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                history = json.load(f)
                if not isinstance(history, list):
                    history = [history]
        else:
            history = []
        
        # Create epoch record
        epoch_record = {
            'epoch': epoch,
            'loss': round(loss, 6),
            'psnr': round(psnr, 4),
            'ssim': round(ssim, 6),
        }
        
        if lr is not None:
            epoch_record['lr'] = round(lr, 8)
        
        # Add any additional metrics
        for key, value in kwargs.items():
            if isinstance(value, float):
                epoch_record[key] = round(value, 6)
            else:
                epoch_record[key] = value
        
        # Append to history
        history.append(epoch_record)
        
        # Save updated history
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"✓ Saved epoch {epoch}: {filepath}")
        return filepath
    
    def save_final_metrics(self,
                          model_name: str,
                          dataset: str,
                          loss: float,
                          psnr: float,
                          ssim: float,
                          epoch: int = None,
                          lr: float = None,
                          **kwargs) -> str:
        metrics = {
            'loss': round(loss, 6),
            'psnr': round(psnr, 4),
            'ssim': round(ssim, 6),
        }
        
        if epoch is not None:
            metrics['epoch'] = epoch
        
        if lr is not None:
            metrics['lr'] = round(lr, 8)
        
        # Add any additional metrics
        for key, value in kwargs.items():
            if isinstance(value, float):
                metrics[key] = round(value, 6)
            else:
                metrics[key] = value
        
        return self.save_result(model_name, dataset, metrics)


# Example usage
if __name__ == "__main__":
    saver = ResultSaver()
    
    # Example 1: Save epoch-by-epoch metrics
    print("Example 1: Saving epoch-by-epoch metrics")
    for epoch in range(1, 101):
        loss = 0.5 * (0.99 ** epoch)  # Simulated decreasing loss
        psnr = 20 + 8 * (1 - 0.99 ** epoch)  # Simulated increasing PSNR
        ssim = 0.6 + 0.2 * (1 - 0.99 ** epoch)  # Simulated increasing SSIM
        
        saver.save_epoch_metrics(
            model_name="PReNet",
            dataset="1400",
            epoch=epoch,
            loss=loss,
            psnr=psnr,
            ssim=ssim,
            lr=0.001 * (0.9 ** (epoch // 30))
        )
    
    # Example 2: Save training history
    print("\nExample 2: Saving training history")
    history = [
        {"epoch": 1, "loss": 0.5234, "psnr": 20.12, "ssim": 0.6234, "lr": 0.001},
        {"epoch": 2, "loss": 0.4234, "psnr": 21.45, "ssim": 0.6534, "lr": 0.001},
        {"epoch": 100, "loss": 0.0234, "psnr": 28.45, "ssim": 0.8234, "lr": 0.0001},
    ]
    
    saver.save_training_history(
        model_name="PReNet_CBAM",
        dataset="1400",
        history=history
    )
    
    print("\nExamples saved successfully!")
