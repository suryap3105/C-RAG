
import re
from typing import Dict, Optional, Tuple

class LLMResponseParser:
    """
    Flexible parser for LLM responses.
    Handles both structured (HYPOTHESIS/ACTION format) and unstructured responses.
    """
    
    def __init__(self):
        # Keywords that indicate an answer was found
        self.answer_indicators = [
            "answer is", "answer:", "the answer", "it is", "it's",
            "result is", "therefore", "thus", "so the answer"
        ]
        
        # Keywords that indicate need for expansion
        self.expand_indicators = [
            "need to", "should", "must", "require", "missing",
            "don't know", "unclear", "not sure", "expand", "explore"
        ]
    
    def parse(self, response: str, query: str, context: list) -> Dict[str, str]:
        """
        Parse LLM response and extract:
        - hypothesis: Current reasoning
        - action: Either 'answer' or 'expand'
        - answer: The final answer if found
        - expansion_target: What to expand (relation/entity)
        """
        result = {
            "hypothesis": "",
            "action": "expand",
            "answer": None,
            "expansion_target": "generic"
        }
        
        response_lower = response.lower()
        
        # 1. Try structured format first (backward compatibility)
        structured = self._parse_structured(response)
        if structured["found"]:
            return structured["data"]
        
        # 2. Check for direct answer indicators
        answer = self._extract_answer(response, response_lower)
        if answer:
            result["action"] = "answer"
            result["answer"] = answer
            result["hypothesis"] = f"Found answer: {answer}"
            return result
        
        # 3. Extract hypothesis from response
        result["hypothesis"] = self._extract_hypothesis(response)
        
        # 4. Determine if we should expand based on keywords
        if any(indicator in response_lower for indicator in self.expand_indicators):
            result["action"] = "expand"
            result["expansion_target"] = self._extract_expansion_target(response, context)
        else:
            # Default: keep expanding if no clear answer
            result["action"] = "expand"
        
        return result
    
    def _parse_structured(self, response: str) -> Dict:
        """Parse strict HYPOTHESIS/ACTION format."""
        lines = response.split('\n')
        data = {
            "hypothesis": "",
            "action": "expand",
            "answer": None,
            "expansion_target": "generic"
        }
        found_structured = False
        
        for line in lines:
            line_stripped = line.strip()
            
            if line_stripped.startswith("HYPOTHESIS:"):
                data["hypothesis"] = line_stripped.split("HYPOTHESIS:", 1)[1].strip()
                found_structured = True
            
            if line_stripped.startswith("ACTION:"):
                action_line = line_stripped.split("ACTION:", 1)[1].strip()
                if "ANSWER_FOUND" in action_line:
                    data["action"] = "answer"
                    # Extract answer after ANSWER_FOUND:
                    if ":" in action_line:
                        data["answer"] = action_line.split(":", 1)[1].strip()
                found_structured = True
            
            # Also check for direct ANSWER_FOUND line
            if line_stripped.startswith("ANSWER_FOUND:"):
                data["action"] = "answer"
                data["answer"] = line_stripped.split("ANSWER_FOUND:", 1)[1].strip()
                found_structured = True
        
        return {"found": found_structured, "data": data}
    
    def _extract_answer(self, response: str, response_lower: str) -> Optional[str]:
        """Try to extract a direct answer from the response."""
        # Pattern 1: "The answer is X"
        patterns = [
            r"(?:the\s+)?answer\s+is\s+(.+?)(?:\.|$)",
            r"(?:it\s+is|it's)\s+(.+?)(?:\.|$)",
            r"therefore[,]?\s+(.+?)(?:\.|$)",
            r"thus[,]?\s+(.+?)(?:\.|$)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response_lower, re.IGNORECASE)
            if match:
                # Get the original case version
                start, end = match.span(1)
                return response[start:end].strip()
        
        # Pattern 2: Look for quoted entities
        quoted = re.findall(r'"([^"]+)"', response)
        if quoted:
            return quoted[0]
        
        return None
    
    def _extract_hypothesis(self, response: str) -> str:
        """Extract reasoning/hypothesis from response."""
        # Take first few sentences as hypothesis
        sentences = response.split('.')
        if sentences:
            # Return first 1-2 sentences
            hypothesis = '. '.join(sentences[:2]).strip()
            return hypothesis if hypothesis else response[:200]
        return response[:200]
    
    def _extract_expansion_target(self, response: str, context: list) -> str:
        """Try to determine what relation/entity to expand."""
        response_lower = response.lower()
        
        # Look for relation keywords
        relations = {
            "director": ["director", "directed by", "filmmaker"],
            "actor": ["actor", "starred", "cast", "starring"],
            "spouse": ["spouse", "married", "wife", "husband"],
            "genre": ["genre", "type of", "category"],
            "year": ["year", "date", "when", "time"]
        }
        
        for relation, keywords in relations.items():
            if any(kw in response_lower for kw in keywords):
                return relation
        
        return "generic"

# Global parser instance
llm_parser = LLMResponseParser()
