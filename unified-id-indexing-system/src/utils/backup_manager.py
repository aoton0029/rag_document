from typing import List
import os
import shutil
import logging

logger = logging.getLogger(__name__)

class BackupManager:
    """Manages backup operations for the application."""
    
    def __init__(self, backup_directory: str):
        self.backup_directory = backup_directory
        os.makedirs(self.backup_directory, exist_ok=True)
    
    def backup_file(self, file_path: str) -> str:
        """Backs up a single file to the backup directory."""
        if not os.path.isfile(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
        
        backup_path = os.path.join(self.backup_directory, os.path.basename(file_path))
        shutil.copy2(file_path, backup_path)
        logger.info(f"Backed up {file_path} to {backup_path}")
        return backup_path
    
    def backup_files(self, file_paths: List[str]) -> List[str]:
        """Backs up multiple files to the backup directory."""
        backup_paths = []
        for file_path in file_paths:
            try:
                backup_path = self.backup_file(file_path)
                backup_paths.append(backup_path)
            except Exception as e:
                logger.error(f"Failed to back up {file_path}: {e}")
                continue
        return backup_paths
    
    def restore_file(self, backup_file: str, restore_directory: str) -> str:
        """Restores a file from the backup directory to the specified directory."""
        if not os.path.isfile(backup_file):
            logger.error(f"Backup file not found: {backup_file}")
            raise FileNotFoundError(f"Backup file not found: {backup_file}")
        
        restore_path = os.path.join(restore_directory, os.path.basename(backup_file))
        shutil.copy2(backup_file, restore_path)
        logger.info(f"Restored {backup_file} to {restore_path}")
        return restore_path
    
    def restore_files(self, backup_files: List[str], restore_directory: str) -> List[str]:
        """Restores multiple files from the backup directory to the specified directory."""
        restored_paths = []
        for backup_file in backup_files:
            try:
                restored_path = self.restore_file(backup_file, restore_directory)
                restored_paths.append(restored_path)
            except Exception as e:
                logger.error(f"Failed to restore {backup_file}: {e}")
                continue
        return restored_paths