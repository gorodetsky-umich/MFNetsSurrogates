import numpy as np
import torch
import networkx as nx

from mfnets_surrogates.net_torch import *

def make_graph_8():
    """A graph with 8 nodes.

    3 -> 7 -> 8
              ^
              |
         1 -> 4
            / ^
           /  |
    2 -> 5 -> 6
    """

    graph = nx.DiGraph()

    dinput = 1
    for node in range(1, 9):
        graph.add_node(node, func=torch.nn.Linear(dinput, 1, bias=True), dim_in=1, dim_out=1)

    graph.add_edge(1, 4, func=torch.nn.Linear(dinput, 1, bias=True), out_rows=1, out_cols=1, dim_in=1)
    graph.add_edge(2, 5, func=torch.nn.Linear(dinput, 1, bias=True), out_rows=1, out_cols=1, dim_in=1)
    graph.add_edge(5, 6, func=torch.nn.Linear(dinput, 1, bias=True), out_rows=1, out_cols=1, dim_in=1)
    graph.add_edge(6, 4, func=torch.nn.Linear(dinput, 1, bias=True), out_rows=1, out_cols=1, dim_in=1)
    graph.add_edge(3, 7, func=torch.nn.Linear(dinput, 1, bias=True), out_rows=1, out_cols=1, dim_in=1)
    graph.add_edge(7, 8, func=torch.nn.Linear(dinput, 1, bias=True), out_rows=1, out_cols=1, dim_in=1)
    graph.add_edge(4, 8, func=torch.nn.Linear(dinput, 1, bias=True), out_rows=1, out_cols=1, dim_in=1)
    graph.add_edge(5, 4, func=torch.nn.Linear(dinput, 1, bias=True), out_rows=1, out_cols=1, dim_in=1)

    roots = set([1, 2, 3])
    return graph, roots


def make_graph_4():
    """A graph with 4 nodes with different output dims

    1- > 4 <- 2 <- 3
    """

    graph = nx.DiGraph()

    dim_in = 1
    dim_out = [2, 3, 6, 4]
    
    graph.add_node(1, func=torch.nn.Linear(dim_in, dim_out[0], bias=True), dim_in=dim_in, dim_out=dim_out[0])
    graph.add_node(2, func=torch.nn.Linear(dim_in, dim_out[1], bias=True), dim_in=dim_in, dim_out=dim_out[1])
    graph.add_node(3, func=torch.nn.Linear(dim_in, dim_out[2], bias=True), dim_in=dim_in, dim_out=dim_out[2])
    graph.add_node(4, func=torch.nn.Linear(dim_in, dim_out[3], bias=True), dim_in=dim_in, dim_out=dim_out[3])
    

    graph.add_edge(1, 4, func=torch.nn.Linear(dim_in, dim_out[0] * dim_out[3], bias=True), out_rows=dim_out[3], out_cols=dim_out[0], dim_in=1)
    graph.add_edge(2, 4, func=torch.nn.Linear(dim_in, dim_out[1] * dim_out[3], bias=True), out_rows=dim_out[3], out_cols=dim_out[1], dim_in=1)
    graph.add_edge(3, 2, func=torch.nn.Linear(dim_in, dim_out[2] * dim_out[1], bias=True), out_rows=dim_out[1], out_cols=dim_out[2], dim_in=1)

    roots = set([1, 3])
    return graph, roots, dim_out

def make_graph_4gen():
    """A graph with 4 nodes with different output dims and generic edge functions

    1- > 4 <- 2 <- 3
    """

    graph = nx.DiGraph()

    dim_in = 1
    dim_out = [2, 3, 6, 4]
    
    graph.add_node(1, func=torch.nn.Linear(dim_in, dim_out[0], bias=True))
    graph.add_node(2, func=LinearScaleShift(dim_in, dim_out[1], dim_out[2]))
    graph.add_node(3, func=torch.nn.Linear(dim_in, dim_out[2], bias=True))
    graph.add_node(4, func=LinearScaleShift(dim_in, dim_out[3],dim_out[0]+dim_out[1]))
    

    graph.add_edge(1, 4)
    graph.add_edge(2, 4)
    graph.add_edge(3, 2)

    roots = set([1, 3])
    return graph, roots, dim_out

def make_graph_4gen_nn():
    """A graph with 4 nodes with different output dims and generic edge functions as fully connected neural networks

    1- > 4 <- 2 <- 3
    """

    graph = nx.DiGraph()

    dim_in = 1
    dim_out = [2, 3, 6, 4]
    
    graph.add_node(1, func=torch.nn.Linear(dim_in, dim_out[0], bias=True))
    graph.add_node(2, func=FullyConnectedNNEdge(dim_in, dim_out[1], dim_out[2],
                                                hidden_layer_sizes=[100, 100,
                                                                    100, 20]))
    graph.add_node(3, func=torch.nn.Linear(dim_in, dim_out[2], bias=True))
    graph.add_node(4, func=FullyConnectedNNEdge(dim_in, dim_out[3],
                                                dim_out[0]+dim_out[1],
                                                hidden_layer_sizes=[100,
                                                                    100,
                                                                    100,
                                                                    100]))

    graph.add_edge(1, 4)
    graph.add_edge(2, 4)
    graph.add_edge(3, 2)

    roots = set([1, 3])
    return graph, roots, dim_out


def make_graph_4gen_nn_equal_model_average():
    """A graph with 4 nodes with different output dims and generic edge functions as fully connected neural networks

    1- > 4 <- 2 <- 3
    """

    graph = nx.DiGraph()

    dim_in = 1
    dim_out = [3, 3, 3, 3]
    
    graph.add_node(1, func=torch.nn.Linear(dim_in, dim_out[0], bias=True))
    graph.add_node(2, func=EqualModelAverageEdge(dim_in, dim_out[1], 1, 
                                                 FeedForwardNet(dim_in,
                                                                dim_out[1],
                                                                [100, 100, 20])))

    graph.add_node(3, func=torch.nn.Linear(dim_in, dim_out[2], bias=True))
    graph.add_node(4, func=EqualModelAverageEdge(dim_in, dim_out[3], 2, 
                                                 FeedForwardNet(dim_in,
                                                                dim_out[3],
                                                                [100, 100, 20])))

    graph.add_edge(1, 4)
    graph.add_edge(2, 4)
    graph.add_edge(3, 2)

    roots = set([1, 3])
    return graph, roots, dim_out


def make_graph_single():
    """Make a graph with a single node

    1
    """
    graph = nx.DiGraph()

    # pnodes = torch.randn((2, 2), device=device, dtype=dtype)
    # pedges = torch.randn((1, 2), device=device, dtype=dtype)

    dim_in = 1
    graph.add_node(1, func=FeedForwardNet(dim_in, 1, [10, 10]))
    return graph, set([1])
