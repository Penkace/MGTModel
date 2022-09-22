# MGT Model

This is the source code for paper [xxx](xxx).

## Abstract
Given start-ups' high-risk and high-reward nature, identifying the ones that will eventually succeed is literally a million-dollar question for practitioners in the \$63-billion venture capital industry and for policy makers worldwide, especially at an early stage such that investment returns can be exponential, and policies can better guide and promote the innovation ecosystem for long-term economic growth.


Although various empirical studies and data-driven modeling work have been done, the predictive power of complex networks of stakeholders including venture capital investors, start-ups, and start-ups' managing members has not been thoroughly explored. We design an effective graph representation learning model where node embeddings are incrementally updated by unsupervised graph self-attention and optimized with fine-tuning by supervised link prediction and node classification. Our model uses network structures, temporal dependencies among time periods, and rich node-level attributes for success prediction. Overall, our method achieves superior performance on a real dataset of global venture capital investments, almost twice as human investors. In addition, our model excels at prediction for start-ups in industries such as healthcare and IT. Meanwhile, we shed light on the impacts on start-up success from observable factors including gender, education, and networking, which can be of value for practitioners as well as policy makers when they screen ventures of high growth potential.


## Data

## Prerequisites
The code has been successfully tested in the following environment. (For older dgl versions, you may need to modify the code)
- Python 3.8.12
- PyTorch 1.11.0
- Pytorch Geometric 2.0.4
- Sklearn 1.0.2
- Pandas 1.3.5

## Getting Started

### Prepare your data

The input of our model is as follows:

* `graph_edges` includes the edges of each time step. The shape is [Time_num x 2 x Edge_num]. Time_num is the number of time steps. Edge_num is the number of the edge in this time step.
* `edge_date` is the time step corresponding to each edge and the length is equal to the number of all edges.
* `edge_type` is the edge type corresponding to each edge and the length is equal to the number of all edges.
* `all_nodes` is the number of nodes.
* `new_companies` is the index of the newly added node at each time. The shape is [(Time_num - 1) x new_add_node_length].
* `labels` is the label of the newly added node at each time. The shape is [(Time_num - 1 ) x new_add_node_length].
* `nodetypes` is the set of node types corresponding to all nodes.

We provide samples of our data in the `./Data` folder.


**Node Representation Learning**

`node_representation_learning.py` : File for generating node representations in VC networks by node classification and link prediction tasks

```python
python node_representation_learning.py --embedding_dim 64 --n_layers_clf 3 --train_embed --loss_type 'LPNC'
```

**Start-up Success Prediction**

`startup_success_prediction.py` : Code that dynamically updates newly added nodes and predicts the success of startups

```python
python startup_success_prediction.py --dynamic_clf --gpus 'cuda:0'
```

**File Statement**
Run the node_representation_learning.py file to generate the representation of the nodes and save the embedding in the file `Save_model`. Then run the startup_success_prediction.py file to make predictions about the success of the startups.
Model/Convs.py contains **MGTConvs**, which is the layer to update the nodes dynamically. **Predict_model** in `Model/Model.py` is the model for startup success prediction.

## Cite

Please cite our paper if you find this code useful for your research:

```
citation
```


