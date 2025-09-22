import logging

class AuditLogger:
    def __init__(self, log_file: str):
        logging.basicConfig(filename=log_file, level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger()

    def log_event(self, event: str, details: str):
        self.logger.info(f"Event: {event}, Details: {details}")

    def log_error(self, error_message: str):
        self.logger.error(f"Error: {error_message}")

    def log_warning(self, warning_message: str):
        self.logger.warning(f"Warning: {warning_message}")

    def log_info(self, info_message: str):
        self.logger.info(f"Info: {info_message}")