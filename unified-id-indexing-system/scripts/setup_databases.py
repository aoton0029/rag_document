import os
from src.db.database_manager import DatabaseManager

def setup_databases():
    db_manager = DatabaseManager()
    
    try:
        db_manager.initialize_connections()
        print("All databases have been set up successfully.")
    except Exception as e:
        print(f"An error occurred while setting up databases: {e}")

if __name__ == "__main__":
    setup_databases()