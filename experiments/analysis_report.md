# Experiment Analysis Report

## Aggregate Performance
| System                |   Accuracy |   Samples |   Avg_Latency |   Avg_Nodes_Expanded |
|:----------------------|-----------:|----------:|--------------:|---------------------:|
| baseline_graph_static |        1   |         4 |     4.83043   |                    0 |
| baseline_vector       |        0.5 |         4 |     0.0594407 |                    0 |
| crag_full_v2          |        1   |         2 |     1.74482   |                    5 |
| crag_no_rerank        |        1   |         4 |     0.0845342 |                    5 |

## Failure Modes
Total Failures: 2
| system          |   failures |
|:----------------|-----------:|
| baseline_vector |          2 |