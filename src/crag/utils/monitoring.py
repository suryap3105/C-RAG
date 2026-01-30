"""
Production Monitoring and Observability System
Metrics, Tracing, Health Checks
"""
import time
import logging
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import json
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


@dataclass
class Metric:
    """Single metric data point."""
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """
    Collect and aggregate metrics for monitoring.
    """
    def __init__(self, buffer_size: int = 10000):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=buffer_size))
        self.lock = threading.Lock()
        
    def record(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a metric."""
        metric = Metric(name=name, value=value, tags=tags or {})
        
        with self.lock:
            self.metrics[name].append(metric)
            
    def increment(self, name: str, value: float = 1.0, tags: Dict[str, str] = None):
        """Increment a counter."""
        self.record(name, value, tags)
        
    def gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """Set a gauge value."""
        self.record(name, value, tags)
        
    def timing(self, name: str, duration_ms: float, tags: Dict[str, str] = None):
        """Record a timing."""
        self.record(f"{name}.duration_ms", duration_ms, tags)
        
    def get_stats(self, name: str, window_seconds: int = 60) -> Dict[str, float]:
        """Get statistics for a metric."""
        with self.lock:
            if name not in self.metrics:
                return {}
                
            cutoff = time.time() - window_seconds
            recent = [m for m in self.metrics[name] if m.timestamp > cutoff]
            
            if not recent:
                return {}
                
            values = [m.value for m in recent]
            
            return {
                'count': len(values),
                'sum': sum(values),
                'mean': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'p50': self._percentile(values, 50),
                'p95': self._percentile(values, 95),
                'p99': self._percentile(values, 99)
            }
            
    @staticmethod
    def _percentile(values: List[float], percentile: float) -> float:
        """Calculate percentile."""
        sorted_vals = sorted(values)
        idx = int(len(sorted_vals) * percentile / 100)
        return sorted_vals[min(idx, len(sorted_vals) - 1)]
        
    def get_all_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get all metrics statistics."""
        return {name: self.get_stats(name) for name in self.metrics.keys()}
        
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        for name, stats in self.get_all_metrics().items():
            clean_name = name.replace('.', '_')
            
            for stat_name, value in stats.items():
                metric_name = f"{clean_name}_{stat_name}"
                lines.append(f"{metric_name} {value}")
                
        return '\n'.join(lines)


class PerformanceTracker:
    """
    Track performance metrics with context manager.
    """
    def __init__(self, collector: MetricsCollector, operation: str, tags: Dict[str, str] = None):
        self.collector = collector
        self.operation = operation
        self.tags = tags or {}
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (time.time() - self.start_time) * 1000  # ms
        
        self.collector.timing(self.operation, duration, self.tags)
        
        if exc_type is not None:
            self.collector.increment(f"{self.operation}.errors", tags=self.tags)
        else:
            self.collector.increment(f"{self.operation}.success", tags=self.tags)
            
        return False  # Don't suppress exceptions


class HealthCheck:
    """
    Health check system for service monitoring.
    """
    def __init__(self):
        self.checks: Dict[str, callable] = {}
        self.last_results: Dict[str, bool] = {}
        
    def register(self, name: str, check_fn: callable):
        """Register a health check."""
        self.checks[name] = check_fn
        
    def run_check(self, name: str) -> bool:
        """Run a specific health check."""
        if name not in self.checks:
            return False
            
        try:
            result = self.checks[name]()
            self.last_results[name] = result
            return result
        except Exception as e:
            logger.error(f"Health check '{name}' failed: {e}")
            self.last_results[name] = False
            return False
            
    def run_all(self) -> Dict[str, bool]:
        """Run all health checks."""
        return {name: self.run_check(name) for name in self.checks.keys()}
        
    def is_healthy(self) -> bool:
        """Check if all health checks pass."""
        results = self.run_all()
        return all(results.values())
        
    def get_status(self) -> Dict[str, Any]:
        """Get detailed health status."""
        results = self.run_all()
        
        return {
            'healthy': all(results.values()),
            'timestamp': datetime.now().isoformat(),
            'checks': {
                name: {'passing': result}
                for name, result in results.items()
            }
        }


class RequestTracer:
    """
    Distributed tracing for requests.
    """
    def __init__(self):
        self.traces: Dict[str, Dict] = {}
        self.lock = threading.Lock()
        
    def start_trace(self, trace_id: str, operation: str):
        """Start a new trace."""
        with self.lock:
            self.traces[trace_id] = {
                'trace_id': trace_id,
                'operation': operation,
                'start_time': time.time(),
                'spans': []
            }
            
    def add_span(self, trace_id: str, name: str, duration_ms: float, metadata: Dict = None):
        """Add a span to a trace."""
        with self.lock:
            if trace_id in self.traces:
                self.traces[trace_id]['spans'].append({
                    'name': name,
                    'duration_ms': duration_ms,
                    'metadata': metadata or {}
                })
                
    def end_trace(self, trace_id: str) -> Dict:
        """End a trace and return results."""
        with self.lock:
            if trace_id not in self.traces:
                return {}
                
            trace = self.traces[trace_id]
            trace['end_time'] = time.time()
            trace['total_duration_ms'] = (trace['end_time'] - trace['start_time']) * 1000
            
            return trace
            
    def export_trace(self, trace_id: str) -> str:
        """Export trace in JSON format."""
        trace = self.end_trace(trace_id)
        return json.dumps(trace, indent=2)


class SystemMonitor:
    """
    Monitor system resources.
    """
    def __init__(self):
        self.metrics = MetricsCollector()
        self._stop_event = threading.Event()
        self._monitor_thread = None
        
    def start(self, interval: float = 10.0):
        """Start monitoring."""
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
        
    def stop(self):
        """Stop monitoring."""
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join()
            
    def _monitor_loop(self, interval: float):
        """Monitoring loop."""
        while not self._stop_event.is_set():
            try:
                self._collect_system_metrics()
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                
            self._stop_event.wait(interval)
            
    def _collect_system_metrics(self):
        """Collect system metrics."""
        try:
            import psutil
            
            # CPU
            self.metrics.gauge('system.cpu.percent', psutil.cpu_percent())
            
            # Memory
            mem = psutil.virtual_memory()
            self.metrics.gauge('system.memory.percent', mem.percent)
            self.metrics.gauge('system.memory.available_mb', mem.available / 1024 / 1024)
            
            # Disk
            disk = psutil.disk_usage('/')
            self.metrics.gauge('system.disk.percent', disk.percent)
            
        except ImportError:
            logger.warning("psutil not available, skipping system metrics")
            
    def get_current_stats(self) -> Dict:
        """Get current system statistics."""
        return self.metrics.get_all_metrics()


# Global instances
_metrics_collector = MetricsCollector()
_health_check = HealthCheck()
_tracer = RequestTracer()

def get_metrics_collector() -> MetricsCollector:
    return _metrics_collector

def get_health_check() -> HealthCheck:
    return _health_check

def get_tracer() -> RequestTracer:
    return _tracer
