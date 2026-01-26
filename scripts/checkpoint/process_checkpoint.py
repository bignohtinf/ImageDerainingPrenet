import os
import torch
import glob
import re

def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'net_epoch*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*epoch(.*).pth.*", file_)
            if result:
                epochs_exist.append(int(result[0]))
        if epochs_exist:
            initial_epoch = max(epochs_exist)
        else:
            initial_epoch = 0
    else:
        initial_epoch = 0

    return initial_epoch

def load_checkpoint(model, optimizer, scheduler, save_dir):
    checkpoint_path = os.path.join(save_dir, 'checkpoint.pth')
    latest_path = os.path.join(save_dir, 'net_latest.pth')
    
    # Try to load full checkpoint first
    if os.path.exists(checkpoint_path):
        print(f"Loading full checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        initial_epoch = checkpoint['epoch']
        best_psnr = checkpoint.get('best_psnr', 0.0)
        print(f"Resumed from epoch {initial_epoch}, best PSNR: {best_psnr:.2f} dB")
        return initial_epoch, best_psnr
    
    # Fallback to latest model only
    elif os.path.exists(latest_path):
        print(f"Loading latest model from {latest_path}")
        model.load_state_dict(torch.load(latest_path))
        initial_epoch = findLastCheckpoint(save_dir)
        return initial_epoch, 0.0
    
    return 0, 0.0