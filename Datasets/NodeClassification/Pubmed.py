import dgl
import os
import sys
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(THIS_DIR, "../../"))
from Datasets.NodeClassification.NodeClassificationDataset import NodeClassificationDataset


class Pubmed(NodeClassificationDataset):
    def __init__(self):
        super().__init__(name='Pubmed')

        self.epochs = 600

    def load_graph(self) -> dgl.DGLGraph:
        graph = dgl.data.PubmedGraphDataset(verbose=False)[0]
        graph.ndata["feat"] = torch.nn.functional.normalize(graph.ndata["feat"], p=1.0)
        return graph

    # def _create_splits(
    #     self,
    #     num_splits: int = 5,
    #     train_per_class: int = 20,
    #     val_size: int = 500,
    #     test_size: int = 1000,
    #     base_seed: int = 0,
    # ):
    #     labels = self.labels.cpu()
    #     num_nodes = labels.shape[0]
    #     classes = torch.unique(labels).tolist()

    #     # Sanity checks (will raise early if sizes don't fit)
    #     total_train = train_per_class * len(classes)
    #     assert total_train + val_size + test_size <= num_nodes, (
    #         f"Requested split sizes exceed number of nodes: "
    #         f"{total_train}+{val_size}+{test_size}>{num_nodes}"
    #     )

    #     self.splits = []
    #     for split in range(num_splits):
    #         g = torch.Generator(device="cpu").manual_seed(base_seed + split)

    #         # 1) Class-balanced train indices: 20 per class by default
    #         train_idx_per_class = []
    #         for c in classes:
    #             cls_idx = torch.nonzero(labels == c, as_tuple=False).view(-1)
    #             perm = cls_idx[torch.randperm(cls_idx.numel(), generator=g)]
    #             train_idx_per_class.append(perm[:train_per_class])
    #         train_idx = torch.cat(train_idx_per_class, dim=0)

    #         # 2) Remaining pool for val/test
    #         mask = torch.zeros(num_nodes, dtype=torch.bool)
    #         mask[train_idx] = True
    #         remaining = torch.nonzero(~mask, as_tuple=False).view(-1)
    #         remaining = remaining[torch.randperm(remaining.numel(), generator=g)]

    #         val_idx = remaining[:val_size]
    #         test_idx = remaining[val_size:val_size + test_size]

    #         self.splits.append({
    #             "Train Indices": train_idx.long(),
    #             "Validation Indices": val_idx.long(),
    #             "Test Indices": test_idx.long(),
    #         })

    # # (optional but handy) keep __len__ in sync with how many splits we made
    # def __len__(self):
    #     return len(getattr(self, "splits", [])) or 1