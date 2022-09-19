# MGT Model

This is the source code for paper [xxx](xxx).

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

* `graph_edges` : The shape is [Time_num x 2 x Edge_num]. Time_num is the number of time steps. Edge_num is the number of the edge in this time step.
* `edge_date` : edge_date is the time step corresponding to each edge.
* `edge_type` : edge_type is the edge type corresponding to each edge.
* `all_nodes` : all_nodes is the number of nodes.
* `new_companies` : The shape is [(Time_num - 1) x new_add_node_length]. It is the index of the newly added node at each time.
* `labels` : The shape is [(Time_num - 1 ) x new_add_node_length]. It is the label of the newly added node at each time.
* `nodetypes` : The set of node types corresponding to all nodes.


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


