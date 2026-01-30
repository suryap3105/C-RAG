"""
Production Error Handling and Recovery System
Circuit Breakers, Retry Logic, Fallbacks
"""
import logging
import time
import functools
from typing import Callable, Any, Optional, Type, Tuple
from dataclasses import dataclass
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout: float = 60.0  # seconds
    
    
class CircuitBreaker:
    """
    Circuit breaker pattern implementation.
    Prevents cascading failures by stopping calls to failing services.
    """
    def __init__(self, config: CircuitBreakerConfig = None):
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.lock = threading.Lock()
        
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self.lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    logger.info("Circuit breaker entering HALF_OPEN state")
                else:
                    raise CircuitBreakerError("Circuit breaker is OPEN")
                    
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
            
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.config.timeout
        
    def _on_success(self):
        """Handle successful call."""
        with self.lock:
            self.failure_count = 0
            
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.success_count = 0
                    logger.info("Circuit breaker CLOSED (recovered)")
                    
    def _on_failure(self):
        """Handle failed call."""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
                logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
                
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                self.success_count = 0
                logger.warning("Circuit breaker reopened during recovery attempt")


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""
    pass


def with_retry(max_attempts: int = 3, 
               backoff: float = 1.0,
               exceptions: Tuple[Type[Exception], ...] = (Exception,)):
    """
    Decorator for automatic retry with exponential backoff.
    
    Args:
        max_attempts: Maximum retry attempts
        backoff: Base backoff time in seconds
        exceptions: Exception types to retry on
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        wait_time = backoff * (2 ** attempt)
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_attempts}). "
                            f"Retrying in {wait_time:.2f}s... Error: {e}"
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts. Error: {e}"
                        )
                        
            raise last_exception
            
        return wrapper
    return decorator


def with_timeout(seconds: float):
    """
    Decorator to enforce timeout on function execution.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"{func.__name__} exceeded timeout of {seconds}s")
                
            # Set signal handler (Unix only)
            try:
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(seconds))
                
                try:
                    result = func(*args, **kwargs)
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
                    
                return result
            except AttributeError:
                # Windows doesn't support SIGALRM, fallback to threading
                result = [None]
                exception = [None]
                
                def target():
                    try:
                        result[0] = func(*args, **kwargs)
                    except Exception as e:
                        exception[0] = e
                        
                thread = threading.Thread(target=target)
                thread.daemon = True
                thread.start()
                thread.join(timeout=seconds)
                
                if thread.is_alive():
                    raise TimeoutError(f"{func.__name__} exceeded timeout of {seconds}s")
                    
                if exception[0]:
                    raise exception[0]
                    
                return result[0]
                
        return wrapper
    return decorator


class FallbackHandler:
    """
    Handles fallback strategies when primary operations fail.
    """
    def __init__(self, primary: Callable, fallback: Callable):
        self.primary = primary
        self.fallback = fallback
        self.fallback_count = 0
        
    def execute(self, *args, **kwargs):
        """Execute with fallback."""
        try:
            return self.primary(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Primary operation failed: {e}. Using fallback.")
            self.fallback_count += 1
            return self.fallback(*args, **kwargs)
            
    def get_stats(self):
        """Get fallback statistics."""
        return {'fallback_count': self.fallback_count}


class GracefulDegradation:
    """
    Implements graceful degradation patterns.
    """
    @staticmethod
    def cached_response(cache_key: str, ttl: int = 3600):
        """Return cached response if operation fails."""
        cache = {}
        
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Try primary
                try:
                    result = func(*args, **kwargs)
                    cache[cache_key] = {
                        'data': result,
                        'timestamp': time.time()
                    }
                    return result
                except Exception as e:
                    # Check cache
                    if cache_key in cache:
                        cached = cache[cache_key]
                        age = time.time() - cached['timestamp']
                        
                        if age < ttl:
                            logger.warning(f"Using cached response (age: {age:.0f}s)")
                            return cached['data']
                            
                    raise
                    
            return wrapper
        return decorator
        
    @staticmethod
    def default_value(default: Any):
        """Return default value if operation fails."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Operation failed, returning default: {e}")
                    return default
                    
            return wrapper
        return decorator


class ErrorBudget:
    """
    Track error budget for SLO compliance.
    """
    def __init__(self, budget: float = 0.01, window: int = 3600):
        """
        Args:
            budget: Error budget (e.g., 0.01 = 1% error rate)
            window: Time window in seconds
        """
        self.budget = budget
        self.window = window
        self.events = []  # (timestamp, success: bool)
        self.lock = threading.Lock()
        
    def record(self, success: bool):
        """Record an event."""
        with self.lock:
            now = time.time()
            self.events.append((now, success))
            
            # Clean old events
            cutoff = now - self.window
            self.events = [(t, s) for t, s in self.events if t > cutoff]
            
    def get_error_rate(self) -> float:
        """Get current error rate."""
        with self.lock:
            if not self.events:
                return 0.0
                
            failures = sum(1 for _, success in self.events if not success)
            return failures / len(self.events)
            
    def is_exhausted(self) -> bool:
        """Check if error budget is exhausted."""
        return self.get_error_rate() > self.budget
        
    def get_remaining_budget(self) -> float:
        """Get remaining error budget percentage."""
        error_rate = self.get_error_rate()
        if error_rate >= self.budget:
            return 0.0
        return (self.budget - error_rate) / self.budget * 100
