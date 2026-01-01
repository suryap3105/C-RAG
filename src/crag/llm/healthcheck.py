
import requests
import sys
import argparse
from typing import Dict, Tuple

class LLMHealthCheck:
    """Healthcheck utility for LLM backends."""
    
    def __init__(self, backend: str = "ollama", base_url: str = "http://localhost:11434"):
        self.backend = backend
        self.base_url = base_url
    
    def check_ollama(self) -> Tuple[bool, Dict]:
        """Check Ollama service health."""
        result = {
            "status": "unknown",
            "url": self.base_url,
            "endpoint": f"{self.base_url}/api/tags",
            "models": [],
            "error": None
        }
        
        try:
            # Check if service is reachable
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                result["status"] = "OK"
                result["models"] = [m.get("name") for m in models]
                return True, result
            else:
                result["status"] = "FAIL"
                result["error"] = f"HTTP {response.status_code}"
                return False, result
                
        except requests.ConnectionError:
            result["status"] = "FAIL"
            result["error"] = "Connection refused - Ollama not running"
            return False, result
        except requests.Timeout:
            result["error"] = "Timeout - Ollama not responding"
            return False, result
        except Exception as e:
            result["status"] = "FAIL"
            result["error"] = str(e)
            return False, result
    
    def print_diagnosis(self, success: bool, result: Dict):
        """Print diagnostic information."""
        print("\n" + "="*60)
        print(f"LLM Backend Healthcheck: {self.backend.upper()}")
        print("="*60)
        
        if success:
            print(f"[OK] Status: {result['status']}")
            print(f"[*] Endpoint: {result['endpoint']}")
            print(f"[#] Models available: {len(result['models'])}")
            for model in result['models']:
                print(f"   - {model}")
        else:
            print(f"[FAIL] Status: {result['status']}")
            print(f"[*] Attempted: {result['endpoint']}")
            print(f"[!] Error: {result['error']}")
            
            # Suggestions
            print("\n[?] Troubleshooting:")
            if "Connection refused" in str(result['error']):
                print("   1. Start Ollama: ollama serve")
                print("   2. Check port availability: netstat -an | findstr 11434")
                print(f"   3. Verify URL: curl {result['endpoint']}")
            elif "Timeout" in str(result['error']):
                print("   1. Ollama might be starting - wait 10s and retry")
                print("   2. Check Ollama logs for errors")
            else:
                print(f"   1. Test manually: curl {result['endpoint']}")
                print("   2. Verify Ollama installation: ollama list")
                print("   3. Check firewall settings")
        
        print("="*60 + "\n")
        
    def run(self) -> bool:
        """Run healthcheck and return success status."""
        if self.backend == "ollama":
            success, result = self.check_ollama()
            self.print_diagnosis(success, result)
            return success
        elif self.backend == "mock":
            print("[OK] MockLLM backend always available (no healthcheck needed)")
            return True
        else:
            print(f"❌ Unknown backend: {self.backend}")
            return False

def main():
    parser = argparse.ArgumentParser(description="LLM Backend Healthcheck")
    parser.add_argument("--backend", type=str, default="ollama", 
                       choices=["ollama", "mock"],
                       help="LLM backend to check")
    parser.add_argument("--url", type=str, default="http://localhost:11434",
                       help="Base URL for backend")
    args = parser.parse_args()
    
    checker = LLMHealthCheck(backend=args.backend, base_url=args.url)
    success = checker.run()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
