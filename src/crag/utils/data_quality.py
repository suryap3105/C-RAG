"""
Data Quality and Validation Pipeline
"""
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DataQualityIssue:
    """Data quality issue report."""
    severity: str  # ERROR, WARNING, INFO
    category: str
    message: str
    location: Optional[str] = None
    data: Optional[Dict] = None


class DataValidator:
    """
    Validate data quality for graphs and documents.
    """
    def __init__(self):
        self.issues: List[DataQualityIssue] = []
        
    def validate_nodes_file(self, path: str) -> List[DataQualityIssue]:
        """Validate nodes file format and content."""
        self.issues = []
        
        if not Path(path).exists():
            self.issues.append(DataQualityIssue(
                severity="ERROR",
                category="file",
                message=f"Nodes file not found: {path}"
            ))
            return self.issues
            
        with open(path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                    
                try:
                    node = json.loads(line)
                    
                    # Required fields
                    if 'id' not in node:
                        self.issues.append(DataQualityIssue(
                            severity="ERROR",
                            category="schema",
                            message="Missing 'id' field",
                            location=f"line {line_num}"
                        ))
                        
                    if 'text' not in node and 'name' not in node:
                        self.issues.append(DataQualityIssue(
                            severity="WARNING",
                            category="schema",
                            message="Missing both 'text' and 'name' fields",
                            location=f"line {line_num}"
                        ))
                        
                    # Check ID type
                    if 'id' in node and not isinstance(node['id'], (int, str)):
                        self.issues.append(DataQualityIssue(
                            severity="ERROR",
                            category="type",
                            message=f"Invalid ID type: {type(node['id'])}",
                            location=f"line {line_num}"
                        ))
                        
                    # Check text quality
                    if 'text' in node:
                        text = node['text']
                        if len(text) < 10:
                            self.issues.append(DataQualityIssue(
                                severity="WARNING",
                                category="quality",
                                message="Text too short (< 10 chars)",
                                location=f"line {line_num}"
                            ))
                        elif len(text) > 10000:
                            self.issues.append(DataQualityIssue(
                                severity="INFO",
                                category="quality",
                                message="Very long text (> 10000 chars)",
                                location=f"line {line_num}"
                            ))
                            
                except json.JSONDecodeError as e:
                    self.issues.append(DataQualityIssue(
                        severity="ERROR",
                        category="format",
                        message=f"JSON parse error: {e}",
                        location=f"line {line_num}"
                    ))
                    
        return self.issues
        
    def validate_edges_file(self, path: str, valid_node_ids: set = None) -> List[DataQualityIssue]:
        """Validate edges file format and content."""
        self.issues = []
        
        if not Path(path).exists():
            self.issues.append(DataQualityIssue(
                severity="ERROR",
                category="file",
                message=f"Edges file not found: {path}"
            ))
            return self.issues
            
        edge_set = set()
        
        with open(path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                    
                try:
                    edge = json.loads(line)
                    
                    # Required fields
                    for field in ['src', 'dst']:
                        if field not in edge:
                            self.issues.append(DataQualityIssue(
                                severity="ERROR",
                                category="schema",
                                message=f"Missing '{field}' field",
                                location=f"line {line_num}"
                            ))
                            
                    # Check for self-loops
                    if 'src' in edge and 'dst' in edge and edge['src'] == edge['dst']:
                        self.issues.append(DataQualityIssue(
                            severity="WARNING",
                            category="quality",
                            message="Self-loop detected",
                            location=f"line {line_num}",
                            data=edge
                        ))
                        
                    # Check for duplicate edges
                    edge_tuple = (edge.get('src'), edge.get('dst'))
                    if edge_tuple in edge_set:
                        self.issues.append(DataQualityIssue(
                            severity="WARNING",
                            category="quality",
                            message="Duplicate edge",
                            location=f"line {line_num}",
                            data=edge
                        ))
                    edge_set.add(edge_tuple)
                    
                    # Check node existence
                    if valid_node_ids:
                        for node_id in [edge.get('src'), edge.get('dst')]:
                            if node_id not in valid_node_ids:
                                self.issues.append(DataQualityIssue(
                                    severity="ERROR",
                                    category="integrity",
                                    message=f"Reference to non-existent node: {node_id}",
                                    location=f"line {line_num}"
                                ))
                                
                except json.JSONDecodeError as e:
                    self.issues.append(DataQualityIssue(
                        severity="ERROR",
                        category="format",
                        message=f"JSON parse error: {e}",
                        location=f"line {line_num}"
                    ))
                    
        return self.issues
        
    def generate_report(self) -> str:
        """Generate human-readable report."""
        report_lines = ["Data Quality Report", "=" * 50, ""]
        
        if not self.issues:
            report_lines.append("✓ No issues found")
            return "\n".join(report_lines)
            
        # Group by severity
        by_severity = {"ERROR": [], "WARNING": [], "INFO": []}
        for issue in self.issues:
            by_severity[issue.severity].append(issue)
            
        for severity in ["ERROR", "WARNING", "INFO"]:
            issues = by_severity[severity]
            if issues:
                report_lines.append(f"\n{severity}S ({len(issues)}):")
                for issue in issues:
                    loc_str = f" at {issue.location}" if issue.location else ""
                    report_lines.append(f"  - [{issue.category}] {issue.message}{loc_str}")
                    
        return "\n".join(report_lines)


class DataCleaner:
    """
    Clean and normalize data.
    """
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Remove control characters
        text = "".join(char for char in text if ord(char) >= 32 or char in '\n\t')
        
        return text.strip()
        
    @staticmethod
    def dedup_edges(edges_path: str, output_path: str):
        """Remove duplicate edges."""
        seen = set()
        unique_count = 0
        duplicate_count = 0
        
        with open(edges_path, 'r') as fin, open(output_path, 'w') as fout:
            for line in fin:
                if not line.strip():
                    continue
                    
                edge = json.loads(line)
                edge_key = (edge.get('src'), edge.get('dst'), edge.get('relation'))
                
                if edge_key not in seen:
                    fout.write(line)
                    seen.add(edge_key)
                    unique_count += 1
                else:
                    duplicate_count += 1
                    
        logger.info(f"Removed {duplicate_count} duplicates, kept {unique_count} unique edges")
