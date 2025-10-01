from typing import Union, List, Optional

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import TAGConv


def _ensure_homograph(g: dgl.DGLGraph) -> dgl.DGLGraph:
    if g.is_homogeneous:
        return g
    assert len(g.ntypes) == 1 and len(g.etypes) == 1, \
        "TAGCN expects a single node/edge type graph."
    keep_ekeys = []
    for k in ("edge_weight", "normalized_edge_weight"):
        if k in g.edata:
            keep_ekeys.append(k)
    return dgl.to_homogeneous(g, edata=keep_ekeys)


class TAGCNLayer(nn.Module):
    def __init__(
        self,
        in_sizes: dict,
        out_sizes: dict,
        *,
        activation: bool = True,
        use_layer_norm: bool = True,
        use_edge_weights: bool = True,
        k: int = 2,
        bias: bool = True,
    ):
        super().__init__()
        self.use_edge_weights = use_edge_weights

        self.conv = TAGConv(
            in_feats=in_sizes["_N"],
            out_feats=out_sizes["_N"],
            k=k,
            bias=bias,
        )

        post = [nn.Identity()]
        if use_layer_norm:
            post.append(nn.LayerNorm(out_sizes["_N"]))
        if activation:
            post.append(nn.ReLU())
        self.post = nn.Sequential(*post)

    def _pick_edge_weight(self, g: dgl.DGLGraph) -> Optional[torch.Tensor]:
        if not self.use_edge_weights:
            return None
        if "edge_weight" in g.edata:
            return g.edata["edge_weight"]
        if "normalized_edge_weight" in g.edata:
            return g.edata["normalized_edge_weight"]
        return None

    def forward(self, graph: dgl.DGLGraph, h: torch.Tensor) -> torch.Tensor:
        assert graph.is_homogeneous, "TAGCNLayer expects a homogeneous graph."
        ew = self._pick_edge_weight(graph)
        if ew is not None and ew.device != h.device:
            ew = ew.to(h.device, non_blocking=True)
        x = self.conv(graph, h, edge_weight=ew) if ew is not None else self.conv(graph, h)
        return self.post(x)


class MLPInputLayer(nn.Module):
    def __init__(self, h_feats: int, hidden_layer_size: int, use_layer_norm: bool):
        super().__init__()
        self.W1 = nn.Linear(h_feats, hidden_layer_size)
        self.W2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.output_layer = nn.LayerNorm(hidden_layer_size) if use_layer_norm else nn.Identity()

    def forward(self, h: torch.Tensor):
        return self.output_layer(self.W2(F.relu(self.W1(h))).squeeze(1))


class TAGCN(nn.Module):
    def __init__(
        self,
        graph: dgl.DGLGraph,
        input_sizes: dict,
        out_sizes: Union[dict, int],
        dropout: float,
        use_edge_weights: bool,
        use_input_layer: bool,
        use_layer_norm: bool,
        hidden_layer_sizes: Union[dict, int],
        n_layers: int,
        log_softmax: bool,
        softmax_output: bool = False,
        k: int = 2,
        bias: bool = True,
        feature_key: str = "feat",
    ):
        super().__init__()
        assert n_layers >= 1

        self.softmax_output = softmax_output
        self.log_softmax = log_softmax
        self.dropout = dropout
        self.feature_key = feature_key

        # Normalize dict/int to dict
        if isinstance(hidden_layer_sizes, int):
            self.hidden_layer_sizes_dict = {nt: hidden_layer_sizes for nt in (graph.ntypes or ["_N"])}
        else:
            self.hidden_layer_sizes_dict = hidden_layer_sizes

        if isinstance(out_sizes, int):
            self.out_sizes_dict = {nt: out_sizes for nt in (graph.ntypes or ["_N"])}
        else:
            self.out_sizes_dict = out_sizes

        self.use_input_layer = use_input_layer
        self.input_layer = None
        if self.use_input_layer:
            self.input_layer = MLPInputLayer(input_sizes["_N"], input_sizes["_N"], use_layer_norm)

        self.train_graph_no_batching = _ensure_homograph(graph)

        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        if self.n_layers > 1:
            # Input
            self.layers.append(
                TAGCNLayer(
                    input_sizes, self.hidden_layer_sizes_dict,
                    activation=True, use_layer_norm=use_layer_norm, use_edge_weights=use_edge_weights,
                    k=k, bias=bias
                )
            )
            # Hidden
            for _ in range(1, self.n_layers - 1):
                self.layers.append(
                    TAGCNLayer(
                        self.hidden_layer_sizes_dict, self.hidden_layer_sizes_dict,
                        activation=True, use_layer_norm=use_layer_norm, use_edge_weights=use_edge_weights,
                        k=k, bias=bias
                    )
                )
            # Output
            self.layers.append(
                TAGCNLayer(
                    self.hidden_layer_sizes_dict, self.out_sizes_dict,
                    activation=False, use_layer_norm=use_layer_norm, use_edge_weights=use_edge_weights,
                    k=k, bias=bias
                )
            )
        else:
            self.layers.append(
                TAGCNLayer(
                    input_sizes, self.out_sizes_dict,
                    activation=False, use_layer_norm=use_layer_norm, use_edge_weights=use_edge_weights,
                    k=k, bias=bias
                )
            )

    def _full_feats(self, g: dgl.DGLGraph, device: torch.device) -> torch.Tensor:
        assert g.is_homogeneous, "TAGCN expects homogeneous graph for features."
        assert self.feature_key in g.ndata, f"Missing g.ndata['{self.feature_key}']"
        X = g.ndata[self.feature_key]
        return X.to(device, non_blocking=True)

    def _maybe_input(self, X: torch.Tensor) -> torch.Tensor:
        if self.use_input_layer:
            X = F.dropout(self.input_layer(X), p=self.dropout, training=self.training)
        return X

    def _run_full(self, g: dgl.DGLGraph, X: torch.Tensor) -> torch.Tensor:
        h = X
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            if i < self.n_layers - 1:
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def forward(self, mfgs: List[dgl.DGLGraph], x: dict, out_key: str):
        g = self.train_graph_no_batching.to(x["_N"].device, non_blocking=True)
        X = self._full_feats(g, x["_N"].device)
        X = self._maybe_input(X)
        H_all = self._run_full(g, X)  # [N_train, C]

        dst_gids = mfgs[-1].dstnodes["_N"].data["_ID"].to(H_all.device, non_blocking=True)
        out = H_all.index_select(0, dst_gids)

        if self.log_softmax:
            return F.log_softmax(out, dim=1)
        elif self.softmax_output:
            return F.softmax(out, dim=1)
        else:
            return out

    @torch.no_grad()
    def inference(self, graph: dgl.DGLGraph, x: dict, indices: torch.Tensor,
                  device: torch.device, out_key: str):
        g = _ensure_homograph(graph).to(device, non_blocking=True)
        X = self._full_feats(g, device)
        X = self._maybe_input(X)
        H_all = self._run_full(g, X)  # [N_infer, C]

        res = H_all.index_select(0, indices.to(device, non_blocking=True))
        if self.log_softmax:
            return F.log_softmax(res, dim=1).detach().to('cpu')
        elif self.softmax_output:
            return F.softmax(res, dim=1).detach().to('cpu')
        else:
            return res.detach().to('cpu')
