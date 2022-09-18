# MGT Model

This is the source code for paper [xxx](xxx) appeared in xxx.

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
* `nodetypes` : It is the set of node types corresponding to all nodes.


**Node Representation Learning**

```python
python main_embedding_training.py --data_dir ./Data --Model_dir ./Save_model
```

**Start-up Success Prediction**

```
python startup_prediction.py --data_dir ./Data --Model_dir ./Save_model
```


1. main_embedding_training.py : File for generating node representations in VC networks by node classification and link prediction tasks
2. main.py : Code that dynamically updates newly added nodes and predicts the success of startups
3. You need to run the main_node_traininig.py file to generate a representation of the nodes and then run the main.py file to make predictions about the success of the startups.
4. Model/Convs.py contains **MGTConvs**, which is the layer to dynamically update the nodes. Model/Model.py contains **Predict_model**.
