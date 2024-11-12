## 0. Python environment setup with Conda
```
conda create --name TAPE python=3.8
conda activate TAPE

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda install -c pyg pytorch-sparse
conda install -c pyg pytorch-scatter
conda install -c pyg pytorch-cluster
conda install -c pyg pyg
pip install ogb
conda install -c dglteam/label/cu113 dgl
pip install yacs
pip install transformers
pip install --upgrade accelerate
```


## 1. Download TAG datasets

### A. Original text attributes

| Dataset | Description |
| ----- |  ---- |
| ogbn-arxiv  | The [OGB](https://ogb.stanford.edu/docs/nodeprop/) provides the mapping from MAG paper IDs into the raw texts of titles and abstracts. <br/>Download the dataset [here](https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz), unzip and move it to `dataset/ogbn_arxiv_orig`.|
| ogbn-products (subset) |  The dataset is located under `dataset/ogbn_products_orig`.|


### B. LLM responses
| Dataset | Description |
| ----- |  ---- |
| ogbn-arxiv  | Download the dataset [here](https://drive.google.com/file/d/1A6mZSFzDIhJU795497R6mAAM2Y9qutI5/view?usp=sharing), unzip and move it to `gpt_responses/ogbn-arxiv`.|
| ogbn-products (subset)  | Download the dataset [here](https://drive.google.com/file/d/1C769tlhd8pT0s7I3vXIEUI-PK7A4BB1p/view?usp=sharing), unzip and move it to `gpt_responses/ogbn-products`.|


## 2. Fine-tuning the LMs
### To use the orginal text attributes
```
WANDB_DISABLED=True TOKENIZERS_PARALLELISM=False CUDA_VISIBLE_DEVICES=0,1,2,3 python -m core.trainLM dataset ogbn-arxiv
```

### To use the GPT responses
```
WANDB_DISABLED=True TOKENIZERS_PARALLELISM=False CUDA_VISIBLE_DEVICES=0,1,2,3 python -m core.trainLM dataset ogbn-arxiv lm.train.use_gpt True
```


## 3. Training the GNNs

### To use different GNN models
```
python -m core.trainEnsemble gnn.model.name MLP
python -m core.trainEnsemble gnn.model.name GCN
python -m core.trainEnsemble gnn.model.name SAGE
python -m core.trainEnsemble gnn.model.name RevGAT gnn.train.lr 0.002 gnn.train.dropout 0.75
```


## 4. Reproducibility
Use `run.sh` to run the codes and reproduce the published results.

This repository also provides the checkpoints for all trained models `(*.ckpt)` and the TAPE features `(*.emb)` used in the project. Please donwload them [here](https://drive.google.com/drive/folders/1nF8NDGObIqU0kCkzVaisWooGEQlcNSIN?usp=sharing).

