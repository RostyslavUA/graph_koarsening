import os
import sys
import torch
import dgl

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(THIS_DIR, "../../"))
from Datasets.NodeClassification.NodeClassificationDataset import NodeClassificationDataset


class Cornell(NodeClassificationDataset):
    def __init__(self, split):
        assert split <= 9, "Cornell provides 10 default splits indexed 0...9"
        self.split = split
        super().__init__(name="Cornell")
        self.epochs = 200

    def load_graph(self) -> dgl.DGLGraph:
        """
        Loads the Cornell graph and ensures:
          - ndata['feat'] is L1 row-normalized
          - train/val/test masks are 1-D bool tensors of shape [N]
        """
        g = None
        from torch_geometric.datasets import WebKB
        root = os.path.join(THIS_DIR, "../../data")
        pyg = WebKB(root=root, name="Cornell")[0]
        self.pyg = pyg
        src = pyg.edge_index[0].cpu().numpy()
        dst = pyg.edge_index[1].cpu().numpy()
        g = dgl.graph((src, dst), num_nodes=pyg.num_nodes)

        g.ndata["feat"] = pyg.x
        g.ndata["label"] = pyg.y

        return g
    
    def _create_splits(self):
        self.num_splits = 10
        self.splits = [
            {
            "Train Indices": self.graph.nodes()[self.pyg["train_mask"][:, i]].long(),
            "Validation Indices": self.graph.nodes()[self.pyg["val_mask"][:, i]].long(),
            "Test Indices": self.graph.nodes()[self.pyg["test_mask"][:, i]].long()
            } for i in range(self.num_splits)
        ]
        self.graph.ndata['train_mask'] = self.pyg["train_mask"][:, self.split]
        self.graph.ndata['val_mask'] = self.pyg["val_mask"][:, self.split]
        self.graph.ndata['test_mask'] = self.pyg["test_mask"][:, self.split]