import torch
from torch.utils.data import Dataset
from pathlib import Path

class GraphDataset(Dataset):
    """
    Dataset class to load pre-generated Graph objects (.pt) from disk.
    Optionally loads features from a separate directory to save disk space.
    """
    def __init__(self, graphs_dir, patient_ids, features_dir=None):
        """
        Args:
            graphs_dir (str or Path): Directory where the .pt files (edges/labels) are stored.
            patient_ids (list): List of patient IDs to include in the dataset.
            features_dir (str or Path, optional): Directory where the .pt files (features) are stored.
        """
        self.graphs_dir = Path(graphs_dir)
        self.patient_ids = patient_ids
        self.features_dir = Path(features_dir) if features_dir else None
        
        # Cache for features to avoid redundant disk I/O
        self.feature_cache = {}

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, index: int):
        p_id = self.patient_ids[index]
        graph_path = self.graphs_dir / f"{p_id}.pt"
        
        # Load the torch_geometric Data object (contains edges, y, etc.)
        try:
            graph = torch.load(graph_path, weights_only=False)
        except Exception as e:
            print(f"Error loading graph for patient {p_id} at {graph_path}: {e}")
            raise e
            
        # LINK FEATURES (Option B)
        # If the graph doesn't have 'x' and we have a features_dir, load them.
        if (not hasattr(graph, 'x') or graph.x is None) and self.features_dir:
            if p_id not in self.feature_cache:
                feat_path = self.features_dir / f"{p_id}.pt"
                self.feature_cache[p_id] = torch.load(feat_path, weights_only=False)
            
            graph.x = self.feature_cache[p_id]

        # Ensure label is long for CrossEntropyLoss
        if hasattr(graph, 'y') and graph.y is not None:
            graph.y = graph.y.long()
            graph.y = graph.y.view(-1)
            
        return graph
