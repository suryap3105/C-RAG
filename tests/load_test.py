"""
Load Testing Suite with Locust
"""
from locust import HttpUser, task, between, events
import random
import json
import logging

logger = logging.getLogger(__name__)


class CRAGUser(HttpUser):
    """Simulated user for load testing."""
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    # Sample queries for realistic load testing
    queries = [
        "Who directed Inception?",
        "What movies did Christopher Nolan make?",
        "Find all actors in The Dark Knight",
        "Movies produced by Warner Bros",
        "Films released in 2010",
        "Director of Interstellar",
        "Actors in Tenet",
        "Batman movies by Nolan",
        "Science fiction films",
        "Action thriller movies"
    ]
    
    def on_start(self):
        """Called when user starts."""
        logger.info("User started")
        
    @task(10)  # Weight: 10x more likely than other tasks
    def query_retrieval(self):
        """Test query endpoint."""
        query = random.choice(self.queries)
        
        payload = {
            "query": query,
            "k": random.choice([5, 10, 20]),
            "use_reranking": random.choice([True, False])
        }
        
        with self.client.post(
            "/query",
            json=payload,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                
                # Validate response
                if "results" not in data:
                    response.failure("Missing 'results' in response")
                elif len(data["results"]) == 0:
                    response.failure("Empty results")
                else:
                    # Check latency
                    latency_ms = data.get("latency_ms", 0)
                    if latency_ms > 1000:  # 1 second threshold
                        logger.warning(f"High latency: {latency_ms}ms")
                        
                    response.success()
            else:
                response.failure(f"Got status code {response.status_code}")
                
    @task(2)
    def health_check(self):
        """Test health endpoint."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if not data.get("healthy"):
                    response.failure("Service unhealthy")
                else:
                    response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")
                
    @task(1)
    def metrics(self):
        """Test metrics endpoint."""
        self.client.get("/metrics")
        
    @task(1)
    def info(self):
        """Test info endpoint."""
        self.client.get("/info")


class StressUser(CRAGUser):
    """User for stress testing with higher load."""
    wait_time = between(0.1, 0.5)  # Faster requests
    
    @task(20)
    def rapid_queries(self):
        """Fire rapid queries."""
        for _ in range(5):
            query = random.choice(self.queries)
            self.client.post("/query", json={"query": query, "k": 5})


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when load test starts."""
    logger.info("Load test starting...")
    
    
@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when load test stops."""
    stats = environment.stats
    
    logger.info("Load test finished. Statistics:")
    logger.info(f"Total requests: {stats.total.num_requests}")
    logger.info(f"Total failures: {stats.total.num_failures}")
    logger.info(f"Average response time: {stats.total.avg_response_time:.2f}ms")
    logger.info(f"RPS: {stats.total.total_rps:.2f}")
    
    # Generate report
    report = {
        "total_requests": stats.total.num_requests,
        "total_failures": stats.total.num_failures,
        "avg_response_time_ms": stats.total.avg_response_time,
        "median_response_time_ms": stats.total.median_response_time,
        "p95_response_time_ms": stats.total.get_response_time_percentile(0.95),
        "p99_response_time_ms": stats.total.get_response_time_percentile(0.99),
        "rps": stats.total.total_rps,
        "failure_rate": stats.total.fail_ratio
    }
    
    with open("load_test_report.json", "w") as f:
        json.dump(report, f, indent=2)
        
    logger.info("Report saved to load_test_report.json")
