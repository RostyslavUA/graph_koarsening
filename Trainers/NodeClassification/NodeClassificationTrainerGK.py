import os, time, torch, tqdm, dgl
from typing import Tuple, Callable, Optional, Dict

from Trainers.NodeClassification.NodeClassificationTrainer import NodeClassificationTrainer
from Models.utils import save_node_classification_model_state
from Trainers.Losses.node_classification import cross_entropy, negative_log_likelihood


LEARNING_LOSSES = {
    # "CrossEntropy": cross_entropy,
    "NegativeLogLikelihood": negative_log_likelihood,  # Learning in log space
}

def check_drift_and_maybe_recoarsen(self,
                                    S_cache: torch.Tensor,
                                    device: torch.device,
                                    drift_threshold: float = 0.5):
    g = self.summarizer.summarized_graph
    g_dev = g.device
    if not hasattr(self, "_probe_ids") or self._probe_ids is None:
        nc = S_cache.size(1)
        probe_size = min(1024, nc)
        self._probe_ids = torch.arange(probe_size, device=g_dev)
        self._probe_prev = None

    probe_ids = self._probe_ids  # on g.device

    self.model.eval()
    with torch.no_grad():
        probe_out_c = _forward_coarse_ids_simple(self, probe_ids, out_device=device)  # [B_c, C]
        out_f_probe = spmm_subset_cols_scatter(S_cache, probe_ids.to(S_cache.device), probe_out_c)[0]
        probe_vec = out_f_probe  # [N_f, C]

        drift = 0.0
        should_recoarsen = False
        if (self._probe_prev is not None) and (self._probe_prev.shape == probe_vec.shape):
            drift = torch.linalg.norm(self._probe_prev - probe_vec) / torch.linalg.norm(self._probe_prev)
            if drift > drift_threshold:
                should_recoarsen = True
        self._probe_prev = probe_vec.detach()
    return should_recoarsen, drift

def _forward_coarse_ids_simple(self,
                               ids: torch.Tensor,
                               out_device: torch.device) -> torch.Tensor:
    g = self.summarizer.summarized_graph
    ids = ids.to(g.device)

    n_layers = int(getattr(self.model, "n_layers", 1))
    sampler = dgl.dataloading.MultiLayerNeighborSampler([-1] * n_layers)

    # Build seed spec
    if g.is_homogeneous:
        seeds = ids
    else:
        seeds = {self.output_node_type: ids}

    dl = dgl.dataloading.DataLoader(
        g,
        seeds,
        sampler,
        batch_size=ids.numel(),
        shuffle=False,
        device=out_device,
        use_uva=False,
    )

    for batch in dl:
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            _, _, blocks = batch
        else:
            blocks = batch

        mfgs = blocks if isinstance(blocks, (list, tuple)) else [blocks]

        b0 = mfgs[0]
        if g.is_homogeneous:
            feats = b0.srcdata["feat"]
            inputs = {self.output_node_type: feats}
        else:
            inputs = {nt: b0.srcnodes[nt].data["feat"] for nt in b0.srctypes}

        out_c = self.model(mfgs, inputs, self.output_node_type)  # [B_c, C]
        return out_c

def _get_device():
    if torch.cuda.is_available():
        dev = torch.device('cuda:0')
        torch.cuda.set_device(dev)
        return dev
    return torch.device('cpu')


def spmm_subset_cols_scatter(
    S_coo: torch.Tensor,
    col_ids: torch.Tensor,
    X: torch.Tensor,
):
    S = S_coo.coalesce()
    device = S.device
    n_rows = S.size(0)
    C = X.size(1)

    row, col = S.indices()  # [nnz], [nnz]
    val = S.values()  # [nnz]

    col_ids = col_ids.to(device=device, dtype=torch.long)
    B_c = col_ids.numel()

    lookup = torch.full((S.size(1),), -1, device=device, dtype=torch.long)
    lookup[col_ids] = torch.arange(B_c, device=device)

    keep = lookup[col] >= 0
    r = row[keep]  # [nnz_sel]
    j_new = lookup[col[keep]]  # [nnz_sel]
    w = val[keep]  # [nnz_sel]

    X_sub = X.index_select(0, j_new)  # [nnz_sel, C]
    contrib = w.unsqueeze(1) * X_sub  # [nnz_sel, C]

    Y = torch.zeros((n_rows, C), device=device, dtype=X.dtype)
    Y.index_add_(0, r, contrib)

    rowsum = torch.zeros(n_rows, device=device, dtype=w.dtype)
    rowsum.index_add_(0, r, w)  # sum of S over the selected columns
    return Y, rowsum


class NodeClassificationTrainerGK(NodeClassificationTrainer):
    def __init__(self, model: torch.nn.Module,
                 learning_loss_name: str, optimizer: torch.optim.Optimizer, rank: int,
                 training_dataloader: dgl.dataloading.DataLoader,
                 validation_graph: dgl.DGLGraph, validation_indices: torch.Tensor,
                 evaluator: Callable, output_node_type: str, out_directory: str,
                 dataset_obj=None, summarizer=None, recoarsen_every: int = 20):
        super().__init__(model, learning_loss_name, optimizer, rank, training_dataloader,
                         validation_graph, validation_indices, evaluator,
                         output_node_type, out_directory)
        self.dataset_obj = dataset_obj
        self.summarizer = summarizer
        self.recoarsen_every = int(recoarsen_every)
        device = _get_device()
        self.P_f_prev = None

        if self.summarizer is not None:
            self.summarizer.bind_teacher(self.model, self.dataset_obj, device)
    
    def train(self, n_epochs: int = 500, compute_period: int = 1):
        learning_convergence = ""

        validation_score = -1.0
        best_validation_score = -1.0
        best_loss = 0.0
        total_time = 0.0
        device = _get_device()

        last_recoarsen_epoch = 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        for epoch in tqdm.tqdm(range(n_epochs), "Training Model", leave=True, disable=True):
            epoch_start = time.time()
            if epoch == 0 or should_recoarsen:
                print(f"RECOARSENING")
                self.summarizer.summarize(out_dir=None, split=None)
                if hasattr(self.model, "train_graph_no_batching"):  # TAGCN does not support node batching
                    self.model.train_graph_no_batching = self.summarizer.summarized_graph
                self._probe_ids = None
                self._probe_prev = None
                new_graph = self.summarizer.summarized_graph
                S_cache = self.summarizer.S.to(device)  # [N_f, N_c]
                S_cache = S_cache.coalesce()
                # (Re)build dataloader with updated coarse graph
                new_loader = self.dataset_obj.get_training_data_loader(
                    new_graph, device, n_layers=self.model.n_layers, data_parallel=False
                )
                self.training_dataloader = new_loader
                last_recoarsen_epoch = epoch
                self.P_f_prev = None
            should_recoarsen = False

            # Train theta
            running_loss = 0.0
            for step, batch in enumerate(self.training_dataloader):
                self.optimizer.zero_grad(set_to_none=True)
                mfgs = batch[-1]
                inputs = {nt: mfgs[0].srcnodes[nt].data['feat'] for nt in mfgs[0].srctypes}
                self.model.train()
                out_c = self.model(mfgs, inputs, self.output_node_type)  # [B_c, C],

                coarse_ids = mfgs[-1].dstnodes[self.output_node_type].data["_ID"]  # [B_c]

                out_f, rowsum_slice = spmm_subset_cols_scatter(S_cache, coarse_ids, out_c)
                P_f = torch.nn.functional.softmax(out_f, dim=1)
                logP_f = (P_f.clamp_min(1e-12)).log()  # Assume the loss in log space
                
                covered = rowsum_slice > (1 - 1e-6)
                train_f = self.summarizer.graph.ndata["train_mask"].to(logP_f.device)
                train_f = train_f & covered
                targets = self.summarizer.graph.ndata["label"].to(logP_f.device)[train_f]
                logP_train = logP_f[train_f]

                loss = LEARNING_LOSSES[self.learning_loss_name](logP_train, targets)
                loss.backward()
                self.post_gradient_computation()
                self.optimizer.step()
                running_loss += loss.item()

            last_loss = running_loss/len(self.training_dataloader)
            should_recoarsen_probe, drift = check_drift_and_maybe_recoarsen(self, S_cache, device, drift_threshold=0.25)
            if should_recoarsen_probe:
                should_recoarsen = True


            # Validate
            if (self.rank == 0) and ((epoch % compute_period == 0) or epoch == n_epochs - 1):
                torch.cuda.empty_cache()
                validation_score = self.compute_validation_score()
                if best_validation_score <= validation_score:
                    best_validation_score = validation_score
                    best_loss = last_loss
                    save_node_classification_model_state(self.model, self.out_directory)
                    torch.save(S_cache, os.path.join(self.out_directory, "S.pth"))

            # Recompute P
            epochs_since = epoch - last_recoarsen_epoch
            # max interval cap
            if (self.summarizer.recoarsen_every > 0) and (epochs_since >= self.summarizer.recoarsen_every):
                should_recoarsen = True

            epoch_time = time.time() - epoch_start
            total_time += epoch_time
            learning_convergence += "{:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}\n".format(
                total_time, epoch_time, last_loss, validation_score, best_validation_score)
            with torch.no_grad():
                print(f"epoch: {epoch}, loss: {last_loss:.5f}, val: {validation_score:.5f}", 
                    f"[S] nc={S_cache.shape[1]}, epoch_time: {epoch_time:.3f}")

        max_gpu_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else -1
        if self.rank == 0:
            from Experiments.utils.file_utils import TRAINING_CONVERGENCE_FILENAME, TRAINING_SUMMARY_FILENAME
            with open(os.path.join(self.out_directory, TRAINING_CONVERGENCE_FILENAME), 'w') as f:
                f.write("Total Time(s), Epoch Time(s), Training Loss, Validation Evaluation, Best Validation Evaluation\n")
                f.write(learning_convergence)
            with open(os.path.join(self.out_directory, TRAINING_SUMMARY_FILENAME), 'w') as f:
                f.write("Training Loss,Validation Evaluation,Total Time(s),Max GPU memory (B)\n")
                f.write("{:.5f}, {:.5f}, {:.5f}, {:d}".format(
                    best_loss, best_validation_score, total_time, max_gpu_mem))
