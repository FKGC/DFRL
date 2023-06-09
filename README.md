# DFRL
Source code for ISWC2023 paper: Dynamic Relational Learning For Few-Shot Knowledge Graph Completion

Few-shot Knowledge Graph (KG) completion is a focus of current research, where each task aims at querying unseen facts of a relation given few-shot reference entity pairs. 
This work proposes an adaptive attentional network for few-shot KG completion by learning adaptive entity and reference representations. Evaluation in link prediction on two public datasets shows that our approach achieves new state-of-the-art results with different few-shot sizes.

# Requirements

```
python 3.6
Pytorch == 1.13.1
CUDA: 11.6
GPU: NVIDIA GeForce RTX 3090
```

# Datasets

We adopt Nell and Wiki datasets to evaluate our model, DFRL.
The orginal datasets and pretrain embeddings are provided from [xiong's repo](https://github.com/xwhan/One-shot-Relational-Learning). 
For convenience, the datasets can be downloaded from [Nell data](https://sites.cs.ucsb.edu/~xwhan/datasets/nell.tar.gz)
and [Wiki data](https://sites.cs.ucsb.edu/~xwhan/datasets/wiki.tar.gz). 
The pre-trained embeddings can be downloaded from [Nell embeddings](https://drive.google.com/file/d/1XXvYpTSTyCnN-PBdUkWBXwXBI99Chbps/view?usp=sharing)
 and [Wiki embeddings](https://drive.google.com/file/d/1_3HBJde2KVMhBgJeGN1-wyvW88gRU1iL/view?usp=sharing).
Note that all these files were provided by xiong and we just select what we need here. 
All the dataset files and the pre-trained TransE embeddings should be put into the directory ./NELL and ./Wiki, respectively.

# How to run
The model in our source code is based on Bi-LSTM interaction. To achieve the best performance, pls train the models as follows:

#### Nell

```
python trainer.py --weight_decay 0.0 --prefix nell.5shot
```

#### Wiki

```
python trainer.py --dataset wiki --embed_dim 50 --BiLSTM_hidden_size 50 --BiLSTM_input_size 50 --dropout_input 0.3 --dropout_layers 0.2 --lr 6e-5 --prefix wiki.5shot
```

To test the trained models, pls run as follows:

#### Nell

```
python trainer.py --weight_decay 0.0 --prefix nell.5shot_best --test
```

#### Wiki

```
python trainer.py --dataset wiki --embed_dim 50  --BiLSTM_hidden_size 50 --BiLSTM_input_size 50 --dropout_input 0.3 --dropout_layers 0.2 --lr 6e-5 --prefix wiki.5shot --test
```

