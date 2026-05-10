import torch
from torch.utils.data import Dataset
from pathlib import Path

class GraphDataset(Dataset):
    """
    Dataset class to load pre-generated Graph objects (.pt) from disk.
    Each file contains a torch_geometric.data.Data object.
    """
    def __init__(self, graphs_dir, patient_ids):
        """
        Args:
            graphs_dir (str or Path): Directory where the .pt files are stored.
            patient_ids (list): List of patient IDs to include in the dataset.
        """
        self.graphs_dir = Path(graphs_dir)
        self.patient_ids = patient_ids

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, index: int):
        p_id = self.patient_ids[index]
        graph_path = self.graphs_dir / f"{p_id}.pt"
        
        # Load the torch_geometric Data object
        try:
            graph = torch.load(graph_path, weights_only=False)
        except Exception as e:
            print(f"Error loading graph for patient {p_id} at {graph_path}: {e}")
            raise e
            
        # Ensure label is long for CrossEntropyLoss
        if hasattr(graph, 'y') and graph.y is not None:
            graph.y = graph.y.long()
            
            # If y is a single-element tensor (e.g., tensor([1])), 
            # some GNN models prefer it to be a 0-d tensor or a specific shape.
            # Usually CrossEntropyLoss expects (N,) or (N, C).
            # Here graph.y is typically [1]
            graph.y = graph.y.view(-1)
            
        return graph
