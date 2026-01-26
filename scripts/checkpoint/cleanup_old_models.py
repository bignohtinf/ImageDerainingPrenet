import os
import glob

def cleanup_old_models(save_dir, keep_latest=True, keep_best=True):
    """Delete old epoch models, keeping only latest and best"""
    file_list = glob.glob(os.path.join(save_dir, 'net_epoch*.pth'))
    
    if not file_list:
        return
    
    # Keep latest and best
    files_to_keep = set()
    if keep_latest:
        latest_path = os.path.join(save_dir, 'net_latest.pth')
        if os.path.exists(latest_path):
            files_to_keep.add(latest_path)
    
    if keep_best:
        best_path = os.path.join(save_dir, 'net_best.pth')
        if os.path.exists(best_path):
            files_to_keep.add(best_path)
    
    # Delete old epoch models
    deleted_count = 0
    for file_path in file_list:
        if file_path not in files_to_keep:
            try:
                os.remove(file_path)
                deleted_count += 1
            except Exception as e:
                print(f"Warning: Could not delete {file_path}: {e}")
    
    if deleted_count > 0:
        print(f"Cleaned up {deleted_count} old model file(s)")
