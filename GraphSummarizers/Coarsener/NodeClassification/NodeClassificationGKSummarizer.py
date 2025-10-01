import time, math, torch, dgl
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import numpy as np


def _lift_mask_to_coarse(S, mask_f_bool: torch.Tensor):
    m = mask_f_bool.to(dtype=torch.float32, device=S.device).unsqueeze(1)  # (n_f,1)
    hits = torch.sparse.mm(S.t(), m).squeeze(1)                            # (n_c,)
    return hits > 0

def repair_empty_columns(S_coo: torch.Tensor) -> torch.Tensor:
    S = S_coo.coalesce()
    n_f, n_c = S.size()
    col_sum = torch.sparse.sum(S, dim=0).to_dense()          # (n_c,)
    empty = (col_sum <= 0)
    if not empty.any():
        return S

    k = int(empty.sum().item())
    seeds = torch.randperm(n_f, device=S.device)[:k]

    row, col = S.indices()
    val = S.values()
    new_rows = torch.cat([row, seeds])
    new_cols = torch.cat([col, empty.nonzero(as_tuple=True)[0]])
    new_vals = torch.cat([val, torch.ones(k, device=S.device, dtype=val.dtype)])

    S_fixed = torch.sparse_coo_tensor(
        torch.stack([new_rows, new_cols]), new_vals, (n_f, n_c), device=S.device, dtype=val.dtype
    ).coalesce()
    return S_fixed

def _build_coarse_graph(
    g_fine: dgl.DGLGraph,
    S: torch.Tensor,
    eps: float = 1e-12,
):
    device = S.device
    n = g_fine.num_nodes()
    X = g_fine.ndata["feat"].to(device)
    nc = S.shape[1]
    assert nc > 0
    col = torch.sparse.sum(S, dim=0).to_dense()  # (nc,)

    src, dst = g_fine.edges()
    src = src.to(device); dst = dst.to(device)
    w = g_fine.edata["edge_weight"].to(device)

    A = torch.sparse_coo_tensor(torch.stack([src, dst]), w, (n, n), device=device).coalesce()
    AS = torch.sparse.mm(A, S)  # (n, nc)
    A_c = torch.sparse.mm(S.t(), AS)

    # Degree-normalize
    A_c = A_c.coalesce()
    s, d = A_c.indices()
    w_c  = A_c.values()
    deg  = torch.sparse.sum(A_c, dim=1).to_dense().clamp_min(eps)   # (n_c,)
    invs = deg.pow(-0.5)
    w_norm = w_c * invs[s] * invs[d]

    gc = dgl.graph((s, d), num_nodes=nc, device=device)
    gc.edata["edge_weight"] = w_c
    gc.edata["normalized_edge_weight"] = w_norm

    Xc = torch.sparse.mm(S.t(), X) / col.unsqueeze(1)  # (n_c, d)
    gc.ndata["feat"] = Xc

    train_mask_c = _lift_mask_to_coarse(S, g_fine.ndata["train_mask"])
    gc.ndata["train_mask"] = train_mask_c
    return gc


class NodeClassificationGKSummarizer:
    def __init__(self, dataset, graph: dgl.DGLGraph, recoarsen_every: int,
                 r=0.5, depth=0, **_):
        self.dataset = dataset
        self.original_graph = graph
        self.device = graph.device if hasattr(graph, "device") else (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.graph = graph.to(self.device)
        self.r = float(r)
        self.depth = int(depth)
        self.summarized_graph = self.graph.clone()  # For init
        self.S = None
        self.recoarsen_every = recoarsen_every
        self._teacher_model = None
        self.centers = "k-means++"


    def check_cached_summary(self, out_dir): 
        return False
    def load_cached_summary(self, out_dir): 
        raise NotImplementedError
        
    def bind_teacher(self, model: torch.nn.Module, dataset_obj, device: torch.device):
        self._teacher_model = model
        self._teacher_device = device
        self._teacher_out_key = dataset_obj.get_output_node_type()

    def summarize(self, out_dir=None, split=None):
        if self._teacher_model is None:
            return 0.0  # Backward compat. We need to init model in run() before summarize()
        t0 = time.time()
        g = self.graph

        n = g.num_nodes()
        nc = max(2, int(math.floor(self.r * n)))

        with torch.no_grad():
            x = {"_N": g.ndata["feat"]}
            idx = torch.arange(g.num_nodes(), device="cpu")
            Z = self._teacher_model.inference(g, x, idx, self._teacher_device, self._teacher_out_key)  # (n, d_out)
            if self._teacher_model.log_softmax:
                Z = Z.exp()

        n_init = 3 if isinstance(self.centers, str) else 1

        mbk = MiniBatchKMeans(
            n_clusters=nc,
            init=self.centers if isinstance(self.centers, np.ndarray) else "k-means++",
            n_init=n_init,
            batch_size=2048,
            max_iter=500,
            reassignment_ratio=0.01,
            #tol=1e-5,
            verbose=0,
        )
        mbk.fit(Z.numpy())
        self.centers = mbk.cluster_centers_
        row = torch.arange(n, dtype=torch.long)
        col = torch.as_tensor(mbk.labels_, dtype=torch.long)
        indices = torch.stack([row, col])  # [2, N]
        values = torch.ones(n, dtype=torch.float32)
        S = torch.sparse_coo_tensor(indices, values, size=(n, nc)).coalesce()
        self.S = S

        # Build coarse graph. Pass to train on theta
        with torch.no_grad():
            S_final = self.S
            # print(f"percent of zeros in P {(S_final == 0).sum() / ((S_final != 0).sum() + (S_final == 0).sum()):.3f}")
            S_final = repair_empty_columns(S_final)
            gc = _build_coarse_graph(g, S_final)
            self.summarized_graph = gc
            
        return time.time() - t0
