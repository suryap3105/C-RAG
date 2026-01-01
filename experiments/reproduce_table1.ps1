
# Reproduce Table 1 Results
Write-Host "Starting C-RAG NeruIPS Reproduction Suite..."
$env:PYTHONPATH='src'

# 1. Vector Baseline
Write-Host "`n[1/4] Running Vector Baseline..."
python src/crag/run_exp.py --config configs/baseline_vector.yaml

# 2. Static Graph Baseline
Write-Host "`n[2/4] Running Static Graph Baseline..."
python src/crag/run_exp.py --config configs/baseline_graph.yaml

# 3. C-RAG (No Rerank)
Write-Host "`n[3/4] Running C-RAG (No Rerank)..."
python src/crag/run_exp.py --config configs/crag_no_rerank.yaml

# 4. C-RAG (Full)
Write-Host "`n[4/4] Running C-RAG (Full)..."
python src/crag/run_exp.py --config configs/crag_full.yaml

Write-Host "`nAll experiments completed. Results saved to 'runs/' directory."
