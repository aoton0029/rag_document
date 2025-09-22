from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class MetricsCollector:
    """Class to collect and manage application metrics."""
    
    def __init__(self):
        self.metrics: Dict[str, Any] = {}
    
    def record_metric(self, metric_name: str, value: Any):
        """Record a metric with its name and value."""
        if metric_name in self.metrics:
            self.metrics[metric_name].append(value)
        else:
            self.metrics[metric_name] = [value]
        logger.info(f"Metric recorded: {metric_name} = {value}")
    
    def get_metric(self, metric_name: str):
        """Retrieve the recorded metrics for a given name."""
        return self.metrics.get(metric_name, [])
    
    def reset_metrics(self):
        """Reset all recorded metrics."""
        self.metrics.clear()
        logger.info("All metrics have been reset.")
    
    def report_metrics(self):
        """Report all collected metrics."""
        for metric_name, values in self.metrics.items():
            logger.info(f"{metric_name}: {values}")