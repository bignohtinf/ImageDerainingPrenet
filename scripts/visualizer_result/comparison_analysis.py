import json
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10


class ComparisonAnalyzer:
    def __init__(self, results_dir: str = "output/results", output_dir: str = "output/comparisons"):
        self.results_dir = results_dir
        self.output_dir = output_dir
        self.data_1400 = {}  # {filename: list of epoch data}
        self.data_100h = {}  # {filename: list of epoch data}
        
        # Create output directories if they don't exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.output_dir, "rain1400")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.output_dir, "rain100h")).mkdir(parents=True, exist_ok=True)
    
    def load_results(self) -> None:
        json_files = glob.glob(os.path.join(self.results_dir, "*.json"))
        
        if not json_files:
            print(f"No JSON files found in {self.results_dir}")
            return
        
        print(f"Found {len(json_files)} JSON files")
        
        for json_file in json_files:
            filename = os.path.basename(json_file)
            
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Ensure data is a list of epochs
                if isinstance(data, dict):
                    # Single epoch, convert to list
                    data = [data]
                elif not isinstance(data, list):
                    print(f"Skipped {filename}: Invalid format (not dict or list)")
                    continue
                
                # Categorize by dataset
                if "1400" in filename:
                    self.data_1400[filename] = data
                    print(f"Loaded Rain1400: {filename} ({len(data)} epochs)")
                elif "100h" in filename.lower() or "100H" in filename:
                    self.data_100h[filename] = data
                    print(f"Loaded Rain100H: {filename} ({len(data)} epochs)")
                else:
                    print(f"Skipped (unknown dataset): {filename}")
            
            except json.JSONDecodeError as e:
                print(f"Error reading {filename}: {e}")
            except Exception as e:
                print(f"Unexpected error with {filename}: {e}")
    
    def extract_model_name(self, filename: str) -> str:
        # Remove dataset identifier and extension
        name = filename.replace("_1400", "").replace("_100h", "").replace("_100H", "")
        name = name.replace(".json", "")
        return name
    
    def prepare_epoch_dataframe(self, data_dict: Dict, dataset_name: str) -> pd.DataFrame:
        records = []
        
        for filename, epochs_data in data_dict.items():
            model_name = self.extract_model_name(filename)
            
            # Process each epoch
            for epoch_data in epochs_data:
                if not isinstance(epoch_data, dict):
                    continue
                
                record = {
                    'Model': model_name,
                    'Filename': filename,
                    'Epoch': epoch_data.get('epoch', np.nan),
                    'Loss': epoch_data.get('loss', np.nan),
                    'PSNR': epoch_data.get('psnr', np.nan),
                    'SSIM': epoch_data.get('ssim', np.nan),
                    'LR': epoch_data.get('lr', np.nan),
                }
                records.append(record)
        
        df = pd.DataFrame(records)
        df['Dataset'] = dataset_name
        return df
    
    def plot_loss_comparison(self, dataset_name: str, data_dict: Dict, output_subdir: str) -> None:
        df = self.prepare_epoch_dataframe(data_dict, dataset_name)
        
        if df.empty:
            print(f"No data for {dataset_name}")
            return
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Plot line for each model
        for model in df['Model'].unique():
            model_data = df[df['Model'] == model].sort_values('Epoch')
            ax.plot(model_data['Epoch'], model_data['Loss'], 
                   marker='o', label=model, linewidth=2, markersize=4)
        
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax.set_title(f'Loss Comparison - {dataset_name}', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, output_subdir, 'loss_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def plot_psnr_comparison(self, dataset_name: str, data_dict: Dict, output_subdir: str) -> None:
        df = self.prepare_epoch_dataframe(data_dict, dataset_name)
        
        if df.empty:
            print(f"No data for {dataset_name}")
            return
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Plot line for each model
        for model in df['Model'].unique():
            model_data = df[df['Model'] == model].sort_values('Epoch')
            ax.plot(model_data['Epoch'], model_data['PSNR'], 
                   marker='o', label=model, linewidth=2, markersize=4)
        
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('PSNR (dB)', fontsize=12, fontweight='bold')
        ax.set_title(f'PSNR Comparison - {dataset_name}', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, output_subdir, 'psnr_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def plot_ssim_comparison(self, dataset_name: str, data_dict: Dict, output_subdir: str) -> None:
        df = self.prepare_epoch_dataframe(data_dict, dataset_name)
        
        if df.empty:
            print(f"No data for {dataset_name}")
            return
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Plot line for each model
        for model in df['Model'].unique():
            model_data = df[df['Model'] == model].sort_values('Epoch')
            ax.plot(model_data['Epoch'], model_data['SSIM'], 
                   marker='o', label=model, linewidth=2, markersize=4)
        
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('SSIM', fontsize=12, fontweight='bold')
        ax.set_title(f'SSIM Comparison - {dataset_name}', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, output_subdir, 'ssim_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def plot_combined_metrics(self, dataset_name: str, data_dict: Dict, output_subdir: str) -> None:
        df = self.prepare_epoch_dataframe(data_dict, dataset_name)
        
        if df.empty:
            print(f"No data for {dataset_name}")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Loss by epoch
        for model in df['Model'].unique():
            model_data = df[df['Model'] == model].sort_values('Epoch')
            axes[0, 0].plot(model_data['Epoch'], model_data['Loss'], 
                           marker='o', label=model, linewidth=2, markersize=4)
        axes[0, 0].set_xlabel('Epoch', fontsize=11)
        axes[0, 0].set_ylabel('Loss', fontsize=11)
        axes[0, 0].set_title('Loss by Epoch', fontsize=12, fontweight='bold')
        axes[0, 0].legend(fontsize=9)
        axes[0, 0].grid(True, alpha=0.3)
        
        # PSNR by epoch
        for model in df['Model'].unique():
            model_data = df[df['Model'] == model].sort_values('Epoch')
            axes[0, 1].plot(model_data['Epoch'], model_data['PSNR'], 
                           marker='o', label=model, linewidth=2, markersize=4)
        axes[0, 1].set_xlabel('Epoch', fontsize=11)
        axes[0, 1].set_ylabel('PSNR (dB)', fontsize=11)
        axes[0, 1].set_title('PSNR by Epoch', fontsize=12, fontweight='bold')
        axes[0, 1].legend(fontsize=9)
        axes[0, 1].grid(True, alpha=0.3)
        
        # SSIM by epoch
        for model in df['Model'].unique():
            model_data = df[df['Model'] == model].sort_values('Epoch')
            axes[1, 0].plot(model_data['Epoch'], model_data['SSIM'], 
                           marker='o', label=model, linewidth=2, markersize=4)
        axes[1, 0].set_xlabel('Epoch', fontsize=11)
        axes[1, 0].set_ylabel('SSIM', fontsize=11)
        axes[1, 0].set_title('SSIM by Epoch', fontsize=12, fontweight='bold')
        axes[1, 0].legend(fontsize=9)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 1])
        
        # Final metrics comparison (last epoch)
        final_df = df.loc[df.groupby('Model')['Epoch'].idxmax()]
        final_df_sorted = final_df.sort_values('PSNR', ascending=False)
        
        x_pos = np.arange(len(final_df_sorted))
        width = 0.25
        
        # Normalize metrics for comparison
        psnr_norm = final_df_sorted['PSNR'] / final_df_sorted['PSNR'].max()
        ssim_norm = final_df_sorted['SSIM']
        loss_norm = 1 - (final_df_sorted['Loss'] / final_df_sorted['Loss'].max())
        
        axes[1, 1].bar(x_pos - width, psnr_norm, width, label='PSNR (norm)', color='steelblue')
        axes[1, 1].bar(x_pos, ssim_norm, width, label='SSIM', color='coral')
        axes[1, 1].bar(x_pos + width, loss_norm, width, label='Loss (1-norm)', color='mediumseagreen')
        
        axes[1, 1].set_xlabel('Model', fontsize=11)
        axes[1, 1].set_ylabel('Normalized Value', fontsize=11)
        axes[1, 1].set_title('Final Metrics Comparison', fontsize=12, fontweight='bold')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(final_df_sorted['Model'], rotation=45, ha='right', fontsize=9)
        axes[1, 1].legend(fontsize=9)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, output_subdir, 'combined_metrics.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def generate_summary_table(self, dataset_name: str, data_dict: Dict, output_subdir: str) -> None:
        df = self.prepare_epoch_dataframe(data_dict, dataset_name)
        
        if df.empty:
            print(f"No data to summarize for {dataset_name}")
            return
        
        # Get final epoch for each model
        final_df = df.loc[df.groupby('Model')['Epoch'].idxmax()]
        
        # Sort by PSNR descending
        df_summary = final_df[['Model', 'Epoch', 'Loss', 'PSNR', 'SSIM']].sort_values(
            'PSNR', ascending=False
        )
        
        # Save to CSV
        output_path = os.path.join(self.output_dir, output_subdir, 'metrics_summary.csv')
        df_summary.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")
        
        # Print summary
        print(f"\n{dataset_name} - METRICS SUMMARY")
        print("="*80)
        print(df_summary.to_string(index=False))
        print("="*80)
    
    def run_analysis(self) -> None:
        print("\n" + "="*80)
        print("COMPARISON ANALYSIS - RAIN REMOVAL MODELS")
        print("="*80 + "\n")
        
        print("Step 1: Loading results...")
        self.load_results()
        
        if not self.data_1400 and not self.data_100h:
            print("\nNo data loaded. Please ensure JSON files exist in output/results/")
            return
        
        print(f"\nLoaded {len(self.data_1400)} Rain1400 models")
        print(f"Loaded {len(self.data_100h)} Rain100H models")
        
        # Process Rain1400
        if self.data_1400:
            print("\n" + "="*80)
            print("RAIN1400 ANALYSIS")
            print("="*80)
            print("\nStep 2: Generating Rain1400 visualizations...")
            self.plot_loss_comparison("Rain1400", self.data_1400, "rain1400")
            self.plot_psnr_comparison("Rain1400", self.data_1400, "rain1400")
            self.plot_ssim_comparison("Rain1400", self.data_1400, "rain1400")
            self.plot_combined_metrics("Rain1400", self.data_1400, "rain1400")
            
            print("\nStep 3: Generating Rain1400 summary table...")
            self.generate_summary_table("Rain1400", self.data_1400, "rain1400")
        
        # Process Rain100H
        if self.data_100h:
            print("\n" + "="*80)
            print("RAIN100H ANALYSIS")
            print("="*80)
            print("\nStep 2: Generating Rain100H visualizations...")
            self.plot_loss_comparison("Rain100H", self.data_100h, "rain100h")
            self.plot_psnr_comparison("Rain100H", self.data_100h, "rain100h")
            self.plot_ssim_comparison("Rain100H", self.data_100h, "rain100h")
            self.plot_combined_metrics("Rain100H", self.data_100h, "rain100h")
            
            print("\nStep 3: Generating Rain100H summary table...")
            self.generate_summary_table("Rain100H", self.data_100h, "rain100h")
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print(f"Results saved to: {self.output_dir}")
        print("  - output/comparisons/rain1400/")
        print("  - output/comparisons/rain100h/")
        print("="*80 + "\n")


def main():
    analyzer = ComparisonAnalyzer(
        results_dir="output/results",
        output_dir="output/comparisons"
    )
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
