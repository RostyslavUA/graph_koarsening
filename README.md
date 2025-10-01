# Adaptive Graph Coarsening for Efficient GNN Training

This repository builds on the [Convolution Matching Graph Summarization](https://github.com/amazon-science/convolution-matching) framework and extends it with a novel algorithm called **Graph Koarsening (GK)**. The GK is a method for adaptive graph coarsening that is based on the K-means clustering algorithm.

In addition to the algorithms and datasets supported by the original repository, this work also integrates **heterophilic datasets** such as *Chameleon, Squirrel, Texas, Wisconsin, and Cornell*.

---

## Usage

The overall codebase and structure follow the conventions of the original repository, but have been extended to include the **Graph Koarsening (GK)** algorithm. Feel free to check the original repository if something remains unclear after reading this README.

### Running Experiments

The main entry point for running **node classification experiments with GK** is:
```
python3 ./Experiments/node_classification_graph_summarization.py
```
Experiments are configured using the YAML configuration files located under `./Experiments/config`
To run the experiment with GK on e.g., node classification on the Cornell dataset
- Set `graph_summarizers: [NodeClassificationGKSummarizer]` under `CornellNodeClassificationGraphSummarization` entry in `./Experiments/config/run_config.yaml`
- Set the parameters of the coarsener ('r', 'recoarsen_every') under  `CornellNodeClassificationGraphSummarization` entry in `./Experiments/config/params.yaml`
- Execute `python3 ./Experiments/node_classification_graph_summarization.py CornellNodeClassificationGraphSummarization`

### Key Modifications
Two new components were added to integrate GK into the framework:
- Graph Koarsening Summarizer
  - File `./GraphSummarizers/Coarsener/NodeClassification/NodeClassificationGKSummarizer.py`
  - Implements the GK algorithm
- Custom trainer for GK
  - File `./Trainers/NodeClassification/NodeClassificationTrainerGK.py`
  - Implements workflow specific to GK
### Results and Analysis
Experiment outputs are stored in the `./results` directory
- Use `./Analysis/parse_results.py` to combine the results from different experiments into `results.csv` and `convergence.csv` files
- Use `./Analysis/NodeClassificationGraphSummarizationAnalysis.ipynb` to analyze the results

## Contact
For questions and comments, feel free to reach out to me - ro22@rice.edu

## Citation
```
@article{olshevskyi2025adaptivegraphcoarseningefficient,
      title={Adaptive Graph Coarsening for Efficient GNN Training}, 
      author={Rostyslav Olshevskyi and Madeline Navarro and Santiago Segarra},
      journal={arXiv preprint arXiv:2509.25706},
      year={2025},
      url={https://arxiv.org/abs/2509.25706}
}
```
