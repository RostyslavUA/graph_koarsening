import dgl
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(THIS_DIR, "../"))

from Datasets.NodeClassification.NodeClassificationDataset import NodeClassificationDataset
from Experiments.utils.model_config_utils import configure_model
from Experiments.utils.model_config_utils import configure_optimizer
from Experiments.utils.distibuted_utils import init_process_group
from Experiments.utils.distibuted_utils import set_torch_device
from Trainers.NodeClassification.NodeClassificationTrainer import NodeClassificationTrainer
from Trainers.NodeClassification.NodeClassificationTrainerGK import NodeClassificationTrainerGK


def run_distributed(rank: int, device_ids: list, split: int,
                    dataset: NodeClassificationDataset,
                    model_name: str, model_params: dict,
                    learning_loss_name: str,
                    optimizer_name: str, optimizer_params: dict,
                    train_graph: dgl.DGLGraph, validation_graph: dgl.DGLGraph,
                    out_directory: str,
                    learnable_p_cfg: dict = None,
                    summarizer = None):
    # Set up process resources.
    init_process_group(device_ids, rank)
    device = set_torch_device(rank=rank)

    # Set run parameters.
    model = configure_model(model_name, model_params, dataset, train_graph, device, data_parallel=True)
    optimizer = configure_optimizer(optimizer_name, optimizer_params, model.parameters())

    enable_lp = learnable_p_cfg["enable"]
    if enable_lp:
        dataloader = dataset.get_training_data_loader(train_graph, device,
                                            n_layers=model_params["n_layers"], data_parallel=True)  # we need the training mask from original graph
        trainer = NodeClassificationTrainerGK(
            model, learning_loss_name, optimizer, rank,
            dataloader, validation_graph, dataset.splits[split]["Validation Indices"],
            dataset.create_validation_evaluator(split),
            dataset.get_output_node_type(), out_directory,
            dataset_obj=dataset, summarizer=summarizer,
            recoarsen_every=int(learnable_p_cfg.get("recoarsen_every", 20)),
        )
    else:
        dataloader = dataset.get_training_data_loader(train_graph, device,
                                            n_layers=model_params["n_layers"], data_parallel=True)
        # Train the model using the summarized graph.
        trainer = NodeClassificationTrainer(
            model, learning_loss_name, optimizer, rank, dataloader,
            validation_graph, dataset.splits[split]["Validation Indices"], dataset.create_validation_evaluator(split),
            dataset.get_output_node_type(), out_directory)
    trainer.train(n_epochs=dataset.epochs, compute_period=dataset.compute_period)


def run(split: int, dataset: NodeClassificationDataset,
        model_name: str, model_params: dict,
        learning_loss_name: str,
        optimizer_name: str, optimizer_params: dict,
        train_graph: dgl.DGLGraph, validation_graph: dgl.DGLGraph,
        out_directory: str,
        graph_summarizer_name: str = None,
        summarizer = None):
    # Set up process resources.
    device = set_torch_device()

    # Set run parameters.
    model = configure_model(model_name, model_params, dataset, train_graph, device, data_parallel=False)
    optimizer = configure_optimizer(optimizer_name, optimizer_params, model.parameters())

    if graph_summarizer_name == "NodeClassificationGKSummarizer":
        assert model.log_softmax, "Assuming log-softmax output"
        model.log_softmax = False  # 1) get logits for K-Means and lifting; 2) apply log-softmax
        model.softmax_output = False
        dataloader = None
        trainer = NodeClassificationTrainerGK(
            model, learning_loss_name, optimizer, 0,
            dataloader, validation_graph, dataset.splits[split]["Validation Indices"],
            dataset.create_validation_evaluator(split),
            dataset.get_output_node_type(), out_directory,
            dataset_obj=dataset, summarizer=summarizer,
        )
    else:
        # Train the model using the summarized graph.
        dataloader = dataset.get_training_data_loader(train_graph, device,
                                                    n_layers=model_params["n_layers"], data_parallel=False)
        trainer = NodeClassificationTrainer(
            model, learning_loss_name, optimizer, 0, dataloader,
            validation_graph, dataset.splits[split]["Validation Indices"], dataset.create_validation_evaluator(split),
            dataset.get_output_node_type(), out_directory)
    trainer.train(n_epochs=dataset.epochs, compute_period=dataset.compute_period)
