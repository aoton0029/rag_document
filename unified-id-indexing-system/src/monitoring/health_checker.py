from typing import List, Dict
import logging
import requests

logger = logging.getLogger(__name__)

class HealthChecker:
    """Application health checker to monitor the status of various components."""

    def __init__(self, services: Dict[str, str]):
        """
        Initialize the HealthChecker with a dictionary of services.

        Args:
            services: A dictionary where keys are service names and values are their health check URLs.
        """
        self.services = services

    def check_health(self) -> Dict[str, bool]:
        """
        Check the health status of all registered services.

        Returns:
            A dictionary with service names as keys and their health status (True/False) as values.
        """
        health_status = {}
        for service_name, url in self.services.items():
            try:
                response = requests.get(url)
                health_status[service_name] = response.status_code == 200
                logger.info(f"{service_name} health check: {health_status[service_name]}")
            except requests.RequestException as e:
                health_status[service_name] = False
                logger.error(f"Health check for {service_name} failed: {e}")
        
        return health_status

    def report_health(self) -> None:
        """
        Report the health status of all services.
        """
        health_status = self.check_health()
        for service_name, status in health_status.items():
            if not status:
                logger.warning(f"{service_name} is down!")
            else:
                logger.info(f"{service_name} is healthy.")