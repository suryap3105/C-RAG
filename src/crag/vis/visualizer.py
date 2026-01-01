import json
import logging
from typing import List, Dict

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [C-RAG] - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("crag_runtime.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("C-RAG")

class TraceVisualizer:
    """
    Generates an HTML report of the agent's reasoning trace.
    """
    def generate_html(self, trace_history: List[Dict], output_file: str = "trace.html"):
        html_content = """
        <html>
        <head>
            <style>
                body { font-family: sans-serif; padding: 20px; background: #f4f4f4; }
                .step { background: white; padding: 15px; margin-bottom: 10px; border-radius: 8px; border-left: 5px solid #007bff; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .thought { color: #555; font-style: italic; margin-bottom: 5px; }
                .action { font-weight: bold; color: #007bff; }
                .candidates { margin-top: 10px; font-size: 0.9em; }
                .node { display: inline-block; background: #e9ecef; padding: 2px 6px; margin: 2px; border-radius: 4px; }
                h1 { color: #333; }
            </style>
        </head>
        <body>
            <h1>C-RAG Reasoning Trace</h1>
        """
        
        for step_data in trace_history:
            step_num = step_data.get('step', 0)
            action = step_data.get('action', 'Unknown')
            # Assuming 'candidates' is an int in history, but we can enrich it in future
            info = f"Processed {step_data.get('candidates')} nodes."
            
            html_content += f"""
            <div class="step">
                <div class="action">Step {step_num}: {action}</div>
                <div class="thought">{info}</div>
            </div>
            """
            
        html_content += "</body></html>"
        
        with open(output_file, "w") as f:
            f.write(html_content)
        logger.info(f"Trace visualization saved to {output_file}")
