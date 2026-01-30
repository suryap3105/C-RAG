"""
Advanced Profiling and Debugging Utilities
"""
import time
import functools
import cProfile
import pstats
import io
import tracemalloc
from contextlib import contextmanager
from typing import Callable, Any
import logging

logger = logging.getLogger(__name__)


@contextmanager
def profile_memory():
    """Context manager for memory profiling."""
    tracemalloc.start()
    
    yield
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    logger.info(f"Memory usage - Current: {current / 1024 / 1024:.2f} MB, Peak: {peak / 1024 / 1024:.2f} MB")


@contextmanager
def profile_time(operation: str = "operation"):
    """Context manager for time profiling."""
    start = time.time()
    
    yield
    
    elapsed = time.time() - start
    logger.info(f"{operation} took {elapsed:.4f}s")


def profile_function(func: Callable) -> Callable:
    """Decorator for function profiling."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            profiler.disable()
            
            # Print stats
            s = io.StringIO()
            stats = pstats.Stats(profiler, stream=s)
            stats.sort_stats('cumulative')
            stats.print_stats(20)  # Top 20 functions
            
            logger.info(f"\nProfile for {func.__name__}:\n{s.getvalue()}")
            
    return wrapper


def memory_profile(func: Callable) -> Callable:
    """Decorator for memory profiling."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            logger.info(
                f"{func.__name__} memory - "
                f"Current: {current / 1024 / 1024:.2f} MB, "
                f"Peak: {peak / 1024 / 1024:.2f} MB"
            )
            
    return wrapper


class PerformanceLogger:
    """
    Log performance metrics for debugging.
    """
    def __init__(self):
        self.metrics = []
        
    def log(self, operation: str, duration: float, metadata: dict = None):
        """Log a performance metric."""
        self.metrics.append({
            'operation': operation,
            'duration_ms': duration * 1000,
            'timestamp': time.time(),
            'metadata': metadata or {}
        })
        
    def summary(self):
        """Print summary statistics."""
        if not self.metrics:
            logger.info("No metrics logged")
            return
            
        by_operation = {}
        for m in self.metrics:
            op = m['operation']
            if op not in by_operation:
                by_operation[op] = []
            by_operation[op].append(m['duration_ms'])
            
        logger.info("\nPerformance Summary:")
        for op, durations in by_operation.items():
            avg = sum(durations) / len(durations)
            min_d = min(durations)
            max_d = max(durations)
            logger.info(f"  {op}: avg={avg:.2f}ms, min={min_d:.2f}ms, max={max_d:.2f}ms, count={len(durations)}")
