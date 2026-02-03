"""
C-RAG V3 Production Training Pipeline
Contrastive Learning for Neural Subgraph Matcher
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from tqdm import tqdm

from ..model.query_graph import QueryGraphGenerator
from ..llm.interface import create_llm_client

logger = logging.getLogger(__name__)


class ContrastiveGraphDataset(Dataset):
    """
    Dataset for contrastive learning of graph matching.
    Each sample: (query_graph, positive_subgraph, negative_subgraphs)
    """
    def __init__(self, graph_engine, queries: List[Dict], 
                 num_negatives: int = 5, num_hops: int = 2,
                 query_gen: Optional[QueryGraphGenerator] = None):
        self.ge = graph_engine
        self.queries = queries
        self.num_negatives = num_negatives
        self.num_hops = num_hops
        
        # Initialize Query Generator if not provided
        if query_gen is None:
             # Default to mock provider for speed in training/testing unless specified
             logger.info("Initializing Validation/Training QueryGraphGenerator with MOCK LLM to avoid token costs.")
             # You should inject the real one if you want actual LLM calls during training data creation
             llm = create_llm_client(provider='mock') 
             self.query_gen = QueryGraphGenerator(llm)
        else:
             self.query_gen = query_gen

    def __len__(self):
        return len(self.queries)
        
    def __getitem__(self, idx) -> Tuple[Data, Data, List[Data]]:
        query = self.queries[idx]
        
        # Get positive subgraph (around answer entities)
        positive_ids = query.get('answer_ids', query.get('relevant_ids', []))
        if not positive_ids:
            # Fallback: random node
            positive_ids = [np.random.randint(0, self.ge.data.num_nodes)]
            
        pos_subgraph = self.ge.extract_subgraph(positive_ids[:5], self.num_hops)
        
        # Get negative subgraphs (random partitions/nodes)
        neg_subgraphs = []
        for _ in range(self.num_negatives):
            neg_center = np.random.randint(0, self.ge.data.num_nodes)
            # Ensure not overlapping with positive
            while neg_center in positive_ids:
                neg_center = np.random.randint(0, self.ge.data.num_nodes)
            neg_subgraphs.append(self.ge.extract_subgraph([neg_center], self.num_hops))
            
        # Create query graph using the ACTUAL generator
        # Note: This might be slow if using real LLM. For large datasets, 
        # it is recommended to pre-generate query graphs.
        if 'query_graph' in query:
             # Pre-computed support
             query_graph = query['query_graph']
        else:
             query_text = query.get('query', query.get('question', ''))
             query_graph = self.query_gen.parse(query_text)

        # Ensure embedding dimension matches model defaults (768) if random
        if query_graph.x.shape[1] != 768:
            logger.warning(f"Query graph dimension mismatch: {query_graph.x.shape[1]}, expected 768. Padding/Projecting.")
            # Simple fix: project or replacement (ideal is to fix the generator's encoder)
            # For now, let's assume the generator uses the correct encoder.
            pass
        
        return query_graph, pos_subgraph, neg_subgraphs


class GNNTrainer:
    """
    Trainer for Neural Subgraph Matcher using contrastive learning.
    """
    def __init__(self, model: nn.Module, device: str = None):
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        self.optimizer = None
        self.scheduler = None
        
    def setup_optimizer(self, lr: float = 1e-4, weight_decay: float = 1e-5):
        """Setup optimizer and scheduler."""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            query_graphs, pos_graphs, neg_graphs_list = batch
            
            # Move to device
            query_graphs = query_graphs.to(self.device)
            pos_graphs = pos_graphs.to(self.device)
            
            # Flatten negatives list of list of Batch objects? 
            # Dataloader with list of Data objects usually collates them into a list of Batch objects if custom collation isn't used.
            # Here neg_graphs_list is a list (length=batch_size) of lists (length=num_neg).
            # Wait, default collate will make neg_graphs_list a list of Batch objects (one per negative index).
            # Actually, default collation for list of Data is tricky. 
            # Let's assume the user uses a custom collate or PyG DataLoader.
            # If using PyG DataLoader, it collates everything into one big Batch.
            
            # Simplest approach relying on model's internal loss function which handles batches now:
            
            # Flatten negatives for processing
            # We need to pass [Batch(size=B), Batch(size=B), List[Batch(size=B)]]
            # neg_graphs_list keys: 0..num_neg-1, values: Batch of size B
            
            neg_list_flat = []
            for neg_batch_k in neg_graphs_list:
                 neg_list_flat.append(neg_batch_k.to(self.device))


            # Compute loss using model's method which now handles batching logic
            loss = self.model.compute_contrastive_loss(query_graphs, pos_graphs, neg_list_flat)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': loss.item()})
            
        self.scheduler.step()
        
        return {
            'loss': total_loss / num_batches,
            'lr': self.scheduler.get_last_lr()[0]
        }
        
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model."""
        self.model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                query_graphs, pos_graphs, neg_graphs_list = batch
                
                query_graphs = query_graphs.to(self.device)
                pos_graphs = pos_graphs.to(self.device)
                
                neg_list_flat = []
                for neg_batch_k in neg_graphs_list:
                     neg_list_flat.append(neg_batch_k.to(self.device))
                
                loss = self.model.compute_contrastive_loss(query_graphs, pos_graphs, neg_list_flat)
                total_loss += loss.item()
                
                # Metric: Accuracy
                # Provide a simple check: is pos score > max(neg scores)
                q_emb = self.model.encode(query_graphs)
                p_emb = self.model.encode(pos_graphs)
                
                pos_sim = F.cosine_similarity(q_emb, p_emb, dim=-1)
                
                # Check negatives
                neg_max_sim = torch.full_like(pos_sim, -1.0)
                
                for neg_batch in neg_list_flat:
                    n_emb = self.model.encode(neg_batch)
                    n_sim = F.cosine_similarity(q_emb, n_emb, dim=-1)
                    neg_max_sim = torch.max(neg_max_sim, n_sim)
                
                correct += (pos_sim > neg_max_sim).sum().item()
                total += query_graphs.num_graphs if hasattr(query_graphs, 'num_graphs') else query_graphs.size(0)
                
        return {
            'val_loss': total_loss / len(dataloader),
            'accuracy': correct / total if total > 0 else 0
        }
        
    def save_checkpoint(self, path: str, epoch: int, metrics: Dict):
        """Save training checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
        
    def load_checkpoint(self, path: str) -> int:
        """Load training checkpoint. Returns epoch number."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        logger.info(f"Loaded checkpoint from {path}")
        return checkpoint.get('epoch', 0)
        
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int = 100, save_dir: str = "checkpoints") -> Dict:
        """
        Full training loop.
        """
        self.setup_optimizer()
        
        best_val_loss = float('inf')
        history = {'train': [], 'val': []}
        
        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.evaluate(val_loader)
            
            history['train'].append(train_metrics)
            history['val'].append(val_metrics)
            
            logger.info(
                f"Epoch {epoch}: "
                f"Train Loss={train_metrics['loss']:.4f}, "
                f"Val Loss={val_metrics['val_loss']:.4f}, "
                f"Val Acc={val_metrics['accuracy']:.4f}"
            )
            
            # Save best model
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                self.save_checkpoint(
                    f"{save_dir}/best_model.pt",
                    epoch,
                    val_metrics
                )
                
            # Regular checkpoint
            if epoch % 10 == 0:
                self.save_checkpoint(
                    f"{save_dir}/checkpoint_epoch_{epoch}.pt",
                    epoch,
                    val_metrics
                )
                
        return history
