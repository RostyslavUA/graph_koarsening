import dgl
import os
import sys
import torch
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(THIS_DIR, "../../"))
from Datasets.NodeClassification.NodeClassificationDataset import NodeClassificationDataset


class Citeseer(NodeClassificationDataset):
    def __init__(self, rnd_seed):
        self.rnd_seed = rnd_seed
        super().__init__(name='Citeseer')

        self.epochs = 600

    def load_graph(self) -> dgl.DGLGraph:
        graph = dgl.data.CiteseerGraphDataset(verbose=False)[0]
        graph.ndata["feat"] = torch.nn.functional.normalize(graph.ndata["feat"], p=1.0)
        return graph
    
    def _create_splits(
        self,
        train_per_class: int = 20,
        val_size: int = 500,
        test_size: int = 1000,
    ):
        labels = self.labels.cpu().numpy()
        num_nodes = labels.shape[0]
        classes = np.unique(labels)
        num_classes = len(classes)

        required = train_per_class * num_classes + val_size + test_size
        if required > num_nodes:
            raise ValueError(
                f"Not enough nodes for requested Planetoid-style split "
                f"({required} needed, have {num_nodes})."
            )

        rng = np.random.default_rng(self.rnd_seed)

        train_idx = []
        for c in classes:
            c_idx = np.where(labels == c)[0]
            rng.shuffle(c_idx)
            take = c_idx[:train_per_class]
            if take.shape[0] < train_per_class:
                raise ValueError(
                    f"Class {c} has only {take.shape[0]} nodes; "
                    f"need {train_per_class} for training."
                )
            train_idx.extend(take.tolist())

        train_idx = np.array(train_idx, dtype=np.int64)

        all_idx = np.arange(num_nodes, dtype=np.int64)
        remaining = np.setdiff1d(all_idx, train_idx, assume_unique=False)
        rng.shuffle(remaining)

        val_idx = remaining[:val_size]
        test_idx = remaining[val_size:val_size + test_size]

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[torch.from_numpy(train_idx)] = True
        val_mask[torch.from_numpy(val_idx)] = True
        test_mask[torch.from_numpy(test_idx)] = True

        self.graph.ndata["train_mask"] = train_mask
        self.graph.ndata["val_mask"] = val_mask
        self.graph.ndata["test_mask"] = test_mask

        split = {
            "Train Indices": torch.from_numpy(train_idx).long(),
            "Validation Indices": torch.from_numpy(val_idx).long(),
            "Test Indices": torch.from_numpy(test_idx).long(),
        }
        self.splits = [split]
