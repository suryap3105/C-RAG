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

logger = logging.getLogger(__name__)


class ContrastiveGraphDataset(Dataset):
    """
    Dataset for contrastive learning of graph matching.
    Each sample: (query_graph, positive_subgraph, negative_subgraphs)
    """
    def __init__(self, graph_engine, queries: List[Dict], 
                 num_negatives: int = 5, num_hops: int = 2):
        self.ge = graph_engine
        self.queries = queries
        self.num_negatives = num_negatives
        self.num_hops = num_hops
        
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
            
        # Create query graph (placeholder - would use QueryGraphGenerator in real setting)
        query_graph = self._create_query_graph(query)
        
        return query_graph, pos_subgraph, neg_subgraphs
        
    def _create_query_graph(self, query: Dict) -> Data:
        """Create simple query graph from query dict."""
        # Placeholder: single node with random embedding
        # In production, would use QueryGraphGenerator
        x = torch.randn(2, 768)
        edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        return Data(x=x, edge_index=edge_index, num_nodes=2)


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
        
    def compute_infonce_loss(self, anchor: torch.Tensor, positive: torch.Tensor,
                            negatives: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
        """
        Compute InfoNCE contrastive loss.
        """
        # Similarity with positive
        pos_sim = F.cosine_similarity(anchor, positive, dim=-1) / temperature
        
        # Similarity with negatives
        neg_sims = F.cosine_similarity(
            anchor.unsqueeze(1).expand_as(negatives),
            negatives,
            dim=-1
        ) / temperature
        
        # InfoNCE
        logits = torch.cat([pos_sim.unsqueeze(-1), neg_sims], dim=-1)
        labels = torch.zeros(anchor.size(0), dtype=torch.long, device=self.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss
        
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
            
            # Encode
            anchor_embs = self.model.encode(query_graphs)
            pos_embs = self.model.encode(pos_graphs)
            
            # Process negatives
            neg_embs_list = []
            for neg_graphs in neg_graphs_list:
                neg_graphs = neg_graphs.to(self.device)
                neg_embs_list.append(self.model.encode(neg_graphs))
                
            neg_embs = torch.stack(neg_embs_list, dim=1)
            
            # Compute loss
            loss = self.compute_infonce_loss(anchor_embs, pos_embs, neg_embs)
            
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
                
                anchor_embs = self.model.encode(query_graphs)
                pos_embs = self.model.encode(pos_graphs)
                
                neg_embs_list = []
                for neg_graphs in neg_graphs_list:
                    neg_graphs = neg_graphs.to(self.device)
                    neg_embs_list.append(self.model.encode(neg_graphs))
                neg_embs = torch.stack(neg_embs_list, dim=1)
                
                loss = self.compute_infonce_loss(anchor_embs, pos_embs, neg_embs)
                total_loss += loss.item()
                
                # Accuracy: is positive most similar?
                pos_sim = F.cosine_similarity(anchor_embs, pos_embs, dim=-1)
                neg_sims = F.cosine_similarity(
                    anchor_embs.unsqueeze(1).expand_as(neg_embs),
                    neg_embs, dim=-1
                )
                
                correct += (pos_sim.unsqueeze(-1) > neg_sims).all(dim=-1).sum().item()
                total += anchor_embs.size(0)
                
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
