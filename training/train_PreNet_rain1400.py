from scripts.data.dataset import prepare_data_Rain12600
from configs.config import data_path, train_data_path, batch_size, recurrent_iter, use_augmentation, use_curriculum, use_gpu, save_path, lr, epochs, cfg, save_freq
from models.PReNet import PReNet
from scripts.checkpoint import load_training_history, save_training_history, cleanup_old_models
from scripts.checkpoint.process_checkpoint import load_checkpoint, findLastCheckpoint
from scripts.data.augmentation import AugmentationPipeline
from utils.print_network import print_network
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from torch.autograd import Variable
from scripts.data.dataset import Dataset
import torch.optim as optim
from evaluation.metric_calculator import MetricCalculator
from torch.optim.lr_scheduler import MultiStepLR

def main():
    print("="*60)
    print("PReNet Training - Corrected Version")
    print("="*60)
    
    # Prepare data
    print("\\nStep 1: Data Preparation")
    if 'rain1400' in data_path.lower():
        prepare_data_Rain12600(data_path, train_data_path, patch_size=100, stride=100)
    
    # Setup dataset
    print("\\nStep 2: Setup Dataset")
    dataset_train = Dataset(data_path=train_data_path, use_augmentation=use_augmentation, aug_level=0)
    
    # Initialize augmentation pipeline if enabled
    if use_augmentation:
        dataset_train.aug_pipeline = AugmentationPipeline(dataset=dataset_train)
        print("Augmentation pipeline initialized")
    
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=batch_size, shuffle=True)
    print(f"Training samples: {len(dataset_train)}")
    
    # Build model
    print("\\nStep 3: Build Model")
    model = PReNet(recurrent_iter=recurrent_iter, use_GPU=use_gpu)
    print_network(model)
    
    # Loss function - Multi-stage supervision with L1+MSE
    criterion_l1 = nn.L1Loss()
    criterion_mse = nn.MSELoss()
    print("\\nUsing Multi-stage supervision: L1 + MSE loss")
    
    # Move to GPU
    if use_gpu:
        model = model.cuda()
        criterion_l1 = criterion_l1.cuda()
        criterion_mse = criterion_mse.cuda()
        print("Model moved to GPU")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    milestone = [30, 50, 80]
    scheduler = MultiStepLR(optimizer, milestones=milestone, gamma=0.2)
    
    # Metrics
    metric_calc = MetricCalculator()
    
    # Load checkpoint (with full state)
    initial_epoch, best_psnr = load_checkpoint(model, optimizer, scheduler, save_path)
    
    # Load training history
    training_history = load_training_history(save_path)
    
    # Filter out epochs that will be re-trained (if resuming)
    if initial_epoch > 0:
        training_history = [h for h in training_history if h['epoch'] < initial_epoch]
        print(f"Kept {len(training_history)} epochs from previous training")
    
    # Training loop
    print("\\nStep 4: Training")
    print("="*60)
    
    # best_psnr already loaded from checkpoint (or 0.0 if starting fresh)
    
    for epoch in range(initial_epoch, epochs):
        print(f"\\nEpoch {epoch+1}/{epochs}")
        print("-"*40)
        
        scheduler.step()  # MultiStepLR doesn't need epoch argument
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Learning rate: {current_lr:.2e}")
        
        # Curriculum learning (ACTUALLY WORKING NOW)
        if use_curriculum and use_augmentation:
            if epoch < cfg["E1"]:
                dataset_train.aug_level = 0
                stage = "Stage 1: Basic Aug"
            elif epoch < cfg["E2"]:
                dataset_train.aug_level = 1
                stage = "Stage 2: + Rain Streak Aug"
            else:
                dataset_train.aug_level = 2
                stage = "Stage 3: + MixUp Aug"
            print(f"Curriculum: {stage} (Aug Level: {dataset_train.aug_level})")
        elif use_augmentation:
            print(f"Augmentation: Level {dataset_train.aug_level} (Fixed)")
        else:
            print("No augmentation")
        
        # Training
        model.train()
        epoch_loss = 0.0
        epoch_psnr = 0.0
        epoch_ssim = 0.0
        
        pbar = tqdm(enumerate(loader_train, 0), total=len(loader_train), desc="Training")
        for i, (input_train, target_train) in pbar:
            model.zero_grad()
            optimizer.zero_grad()

            input_train, target_train = Variable(input_train), Variable(target_train)

            if use_gpu:
                input_train, target_train = input_train.cuda(), target_train.cuda()

            # Forward pass
            out_train, x_list = model(input_train)
            
            # Multi-stage supervision: L1 + MSE loss for all outputs
            loss = 0
            for x in x_list:
                # Combine L1 and MSE: 0.5 * L1 + 0.5 * MSE
                loss += 0.5 * criterion_l1(x, target_train) + 0.5 * criterion_mse(x, target_train)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Calculate metrics for monitoring
            with torch.no_grad():
                out_train_clamped = torch.clamp(out_train, 0., 1.)
                psnr_train = metric_calc.calculate_psnr(out_train_clamped, target_train)
                ssim_train = metric_calc.calculate_ssim(out_train_clamped, target_train)
            
            epoch_loss += loss.item()
            epoch_psnr += psnr_train
            epoch_ssim += ssim_train
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'PSNR': f'{psnr_train:.2f}',
                'SSIM': f'{ssim_train:.4f}'
            })

        # Epoch summary
        avg_loss = epoch_loss / len(loader_train)
        avg_psnr = epoch_psnr / len(loader_train)
        avg_ssim = epoch_ssim / len(loader_train)
        print(f"\\nEpoch {epoch+1} Summary:")
        print(f"   Loss: {avg_loss:.4f}")
        print(f"   PSNR: {avg_psnr:.2f} dB")
        print(f"   SSIM: {avg_ssim:.4f}")
        
        # Save training history
        training_history.append({
            'epoch': epoch + 1,
            'loss': round(avg_loss, 6),
            'psnr': round(avg_psnr, 4),
            'ssim': round(avg_ssim, 6),
            'lr': round(current_lr, 8)
        })
        save_training_history(save_path, training_history)

        # Save checkpoint (full state for resume)
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_psnr': best_psnr,
        }
        torch.save(checkpoint, os.path.join(save_path, 'checkpoint.pth'))
        
        # Save latest model
        torch.save(model.state_dict(), os.path.join(save_path, 'net_latest.pth'))
        
        # Save epoch model if needed
        if (epoch + 1) % save_freq == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f'net_epoch{epoch+1}.pth'))
        
        # Track and save best model
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save(model.state_dict(), os.path.join(save_path, 'net_best.pth'))
            print(f"New best PSNR: {best_psnr:.2f} dB")
        
        # Cleanup old epoch models (keep only latest and best)
        cleanup_old_models(save_path, keep_latest=True, keep_best=True)

    print("\\n" + "="*60)
    print("TRAINING COMPLETE")
    print(f"Best PSNR: {best_psnr:.2f} dB")
    print("="*60)

if __name__ == "__main__":
    main()