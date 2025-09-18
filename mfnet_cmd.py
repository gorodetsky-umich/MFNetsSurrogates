"""Command Line Utility for MFNETS."""
import sys
import yaml
import json
import os
import logging
import pathlib
from collections import namedtuple

from typing import Literal, Any

import argparse

import pydantic as pyd

import networkx as nx
import torch
import numpy as np
from sklearn import preprocessing
# import matplotlib.pyplot as plt

from mfnets_surrogates import net_torch as net
from mfnets_surrogates import net_pyro, pce_model

from pyro.infer import MCMC, NUTS, Predictive, SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.infer.autoguide import (
    # AutoDelta,
    AutoNormal,
    AutoMultivariateNormal,
    # AutoLowRankMultivariateNormal,
    # AutoGuideList,
    AutoIAFNormal,
    # init_to_feasible,
)


import pandas as pd


logger = logging.getLogger(__name__)

ModelTrainData = namedtuple('ModelTrainData', ('train_in', 'train_out', 'dim_in', 'dim_out', 'output_dir'))

class ModelDescription(pyd.BaseModel):

    name: str | int = pyd.Field(description="Name")
    desc: str = pyd.Field(description="Description.", default="no description")
    train_input: pyd.FilePath = pyd.Field(description="Training input file.")
    train_output: pyd.FilePath = pyd.Field(description="Training output file.")
    output_dir: str = pyd.Field(description="Path to output")
    test_output: None | pyd.FilePath | list[pyd.FilePath] = pyd.Field(description="Testing input file.",
                                                                      default=None)


class MCMCParams(pyd.BaseModel):
    burnin: int = pyd.Field(description="Burnin", default=100, gt=0)
    num_samples: int = pyd.Field(description='number of samples', default=100, gt=0)

class IAFParams(pyd.BaseModel):
    hidden_dim: int = pyd.Field(description="Burnin", default=10, gt=0)
    num_transforms_dim: int = pyd.Field(description="Burnin", default=10, gt=0)    
    
    
class Algorithm(pyd.BaseModel):

    noise_std: float = pyd.Field(description="Noise of data.", gt=0)
    noise_std_predict: None | float = pyd.Field(description="Predict with noise addition", default=None,
                                                gt=0)    
    parametrization: None | Literal['svi-normal', 'mcmc', 'svi-multinormal', 'svi-iafflow'] = pyd.Field(description="inference algorithm", default='svi-normal')
    mcmc_params: None | MCMCParams = pyd.Field(description="MCMC parameters.", default=None)
    iaf_params: None | IAFParams = pyd.Field(description="IAF parameters.", default=None)
    num_optimization_steps: int = pyd.Field(description="number of optimization steps", default=1000, gt=0)
    num_pred_samples: None | int = pyd.Field(description="Number of samples for Bayesian inference.", gt=0,
                                             default=None)
    sample_output_files: None | str = pyd.Field(description="File where to save samples.", default=None)
    

class ConnectionModel(pyd.BaseModel):

    name: int | str = pyd.Field(description="model whose structure is being described")
    node_type: Literal['linear', 'feedforward', 'polynomial', 'poly-linear-scale-shift'] = pyd.Field(description="Node/edge model type")
    hidden_layers: None | list[int] = pyd.Field(description="hidden layer architecture in an mlp model", default=None)
    poly_order: None | int = pyd.Field(description="polynomial order", default=None)
    poly_name: None | Literal['hermite', 'legendre']  = pyd.Field(description="polynomial type", default=None)
    edge_type: Literal['equal_model_average', 'normal'] = pyd.Field(description="special edge type", default='normal')
    
    

class Graph(pyd.BaseModel):
    structure: pyd.FilePath = pyd.Field(description="graph structure")
    structure_format: Literal['edge list', 'adjacency list'] = pyd.Field(description="format of graph structure", default='edge list')
    connection_type: Literal['scale-shift', 'general'] = pyd.Field(description="graph connection type")    
    node_model: None | Literal['linear'] = pyd.Field(description="Node model for scale-shift model", default=None)
    edge_model: None | Literal['linear'] = pyd.Field(description="Edge model for scale-shift model", default=None)    
    connection_models: None | list[ConnectionModel] = pyd.Field(description="node and edge types")

class Config(pyd.BaseModel):

    # below line is needed because I have a model_info and model_ is protected by pydantic for some reason
    model_config  = pyd.ConfigDict(protected_namespaces=())

    
    num_models: int =  pyd.Field(description="Number of models", gt=0)
    save_dir: str = pyd.Field(description="Save directory.")
    model_info: list[ModelDescription]
    inference_type: Literal['regression', 'bayes']
    
    algorithm: Algorithm

    graph: Graph

def parse_model_info(config: Config) -> dict[str | int, ModelTrainData]:
    """Parse data files."""
    logger.info(f"Number of models: {config.num_models}")

    models = {}
    for model in config.model_info:

        name = model.name
        try: 
            train_input = pd.read_csv(model.train_input, sep='\s+')
        except FileNotFoundError:
            print(f"Cannot open training inputs for model {name} in file {model.train_input}")
            exit(1)

        try: 
            train_output = pd.read_csv(model.train_output, sep='\s+')
        except FileNotFoundError:
            print(f"Cannot open training outputs for model {name} in file {model.train_output}")
            exit(1)            
            
        assert train_input.shape[0] == train_output.shape[0]

        dim_in = train_input.shape[1]
        dim_out = train_output.shape[1]

        output_dir = os.path.join(os.getcwd(), config.save_dir, model.output_dir)
        
        models[name] = ModelTrainData(train_input, train_output, dim_in, dim_out, output_dir)
        logger.info(f"Model {name}: number of inputs = {dim_in}, number of outputs = {dim_out}, ntrain = {train_output.shape[0]}, output_dir = {output_dir}")

    return models

    
def fill_graph(graph: nx.Graph, config: Config, model_info: dict[int | str, ModelTrainData]) -> nx.Graph:
    """Assign node and edge functions."""

    # add nodes that were not included in the edge list
    model_names = list(model_info.keys())
    for name in model_names:
        if name not in graph.nodes:
            graph.add_node(name)

    if config.graph.connection_type == 'scale-shift':
        logger.info('Scale-shift edge functions are used')
        for node in graph.nodes:

            # works because model names must match in the input file and in the graph.edge_list file
            dim_in = model_info[node].dim_in
            dim_out = model_info[node].dim_out
            logger.info(f"Updating function for graph node {node}: dim_in = {dim_in}, dim_out = {dim_out}")
            graph.nodes[node]['func'] = torch.nn.Linear(dim_in, dim_out, bias=True)
            graph.nodes[node]['dim_in'] = dim_in
            graph.nodes[node]['dim_out'] = dim_out

        for e1, e2 in graph.edges:

            # rho needs to multiply output of lower fidelity model and be of the dimension of output of high-fidelity model
            dim_in = model_info[e2].dim_in
            dim_out_rows = model_info[e2].dim_out
            dim_out_cols = model_info[e1].dim_out
            logger.info(f"Updating function for graph edge {e1}->{e2} (rho_[e1->e2](x)): dim_in = {dim_in}, dim_out = {dim_out_rows} x {dim_out_cols}, but flattened")
            graph.edges[e1, e2]['func'] = torch.nn.Linear(dim_in, dim_out_rows * dim_out_cols, bias=True)
            graph.edges[e1, e2]['out_rows'] = dim_out_rows
            graph.edges[e1, e2]['out_cols'] = dim_out_cols
            graph.edges[e1, e2]['dim_in'] = dim_in

    elif config.graph.connection_type == "general":
        logger.info('General edge functions are used')
        for node in graph.nodes:
            dim_in = model_info[node].dim_in
            dim_out = model_info[node].dim_out
            # print(list(graph.predecessors(node)
            num_inputs_parents = np.sum([model_info[p].dim_out for p in graph.predecessors(node)])
            num_parents = len([p for p in graph.predecessors(node)])
            
            logger.info(f'Assigning model for node {node}')
            logger.info(f'Number of parents for node {node} = {num_parents}')
            # exit(1)
            # so far only use linear functions to test interface
            if num_inputs_parents == 0:
                for model in config.graph.connection_models:
                    if model.name == node:
                        logger.info(f"Leaf node with type: {model.node_type}")
                        if model.node_type == "linear":
                            graph.nodes[node]['func'] = torch.nn.Linear(dim_in, dim_out, bias=True)
                        elif model.node_type == "polynomial":
                            poly_order = model.poly_order
                            poly_name = model.poly_name  # HG or LU
                            graph.nodes[node]['func'] = pce_model.PCE(dim_in,
                                                                      dim_out,
                                                                      poly_order,
                                                                      poly_name)
                        elif model.node_type == "feedforward":
                            hidden_layer = model.hidden_layers
                            logger.info(f'Feedforward with hidden layer sizes: {hidden_layer}')
                            graph.nodes[node]['func'] = net.FeedForwardNet(dim_in, dim_out,
                                                                           hidden_layer_sizes=hidden_layer)
                        else:
                            raise Exception(f"Node type {model.node_type} unknown")
                        break
                                
            else:
                for model in config.graph.connection_models:
                    if model.name == node:
                        logger.info(f"Regular node with type: {model.node_type}")
                        try:
                            et = model.edge_type
                        except KeyError:
                            et = None
                        if et == 'equal_model_average':
                            logger.info(f"Processing model averaged edge")
                            if model.node_type == "linear":
                                graph.nodes[node]['func'] = \
                                    net.EqualModelAverageEdge(dim_in, dim_out,
                                                              num_parents,
                                                              torch.nn.Linear(dim_in, dim_out, bias=True))                                
                            elif model.node_type == "feedforward":
                                hidden_layer = model.hidden_layers
                                logger.info(f'Feedforward with hidden layer sizes: {hidden_layer}')
                                graph.nodes[node]['func'] = \
                                    net.EqualModelAverageEdge(dim_in, dim_out,
                                                              num_parents,
                                                              net.FeedForwardNet(dim_in, dim_out,
                                                                           hidden_layer_sizes=hidden_layer))
                            else:
                                raise Exception(f"Node type {model.node_type} unknown")
                        else:
                            logger.info(f"Processing learned edge")
                            if model.node_type == "linear":
                                graph.nodes[node]['func'] = net.LinearScaleShift(dim_in, dim_out, num_inputs_parents)
                            elif model.node_type == "poly-linear-scale-shift":
                                poly_order = model.poly_order
                                poly_name = model.poly_name  # HG or LU
                                graph.nodes[node]['func'] = net.PolyScaleShift(
                                    dim_in,
                                    dim_out,
                                    num_inputs_parents,
                                    poly_order,
                                    poly_name)
                            elif model.node_type == "feedforward":
                                hidden_layer = model.hidden_layers
                                logger.info(f'Feedforward with hidden layer sizes: {hidden_layer}')
                                graph.nodes[node]['func'] = net.FullyConnectedNNEdge(dim_in, dim_out,
                                                                                     num_inputs_parents,
                                                                                     hidden_layer_sizes=hidden_layer)
                            else:
                                raise Exception(f"Node type {model.node_type} unknown")

                
            graph.nodes[node]['dim_in'] = dim_in
            graph.nodes[node]['dim_out'] = dim_out
            
            
    else:
        logger.error(f"Connection type {model_info.graph.connection_type} is not recognized")
        exit(1)

    return graph


def parse_graph(config: Config, model_info: dict[str | int, ModelTrainData]) -> tuple[nx.Graph, set[Any]]:
    """Parse the graph."""
    try:
        with open(config.graph.structure) as f:
            graph_read = f.read().splitlines()
    except FileNotFoundError:
        print(f"Cannot open file {config.graph.structure}")
        exit(1)

    logger.info(f"Graph file type: {config.graph.structure_format}")
    if config.graph.structure_format == "edge list":
        graph = fill_graph(nx.parse_edgelist(graph_read, create_using=nx.DiGraph, nodetype=int),
                           config, model_info)
    elif config.graph.structure_format == "adjacency list":
        graph = fill_graph(nx.parse_adjlist(graph_read, create_using=nx.DiGraph, nodetype=int),
                           config, model_info)
    else:
        logger.error(f"File type unrecognized")
        exit(1)

    roots = set([x for x in graph.nodes() if graph.in_degree(x) == 0])
    logger.info(f"Root models: {roots}")
    return graph, roots



def parse_evaluation_locations(config: Config) -> dict[str | int, None | list[tuple[str, pd.DataFrame]]]:
    """Parse eval_locations."""
    model_evals = {} 
    for ii, model in enumerate(config.model_info):

        name = model.name

        if model.test_output is not None:

            filename = model.test_output
            logger.info(f"Will evaluate model {name} at inputs of file(s) {filename}")

            if isinstance(filename, list):
                model_evals[name] = []
                for fname in filename:
                    try: 
                        test_input = pd.read_csv(fname, sep='\s+')
                    except FileNotFoundError:
                        print(f"Cannot open test inputs for model {name} in file {fname}")
                        exit(1)
                    model_evals[name].append((fname.name, test_input))
            else:
                try: 
                    test_input = pd.read_csv(filename, sep='\s+')
                except FileNotFoundError:
                    print(f"Cannot open test inputs for model {name} in file {filename}")
                    exit(1)
                    
                model_evals[name] = [(filename.name, test_input)]
        else:
            model_evals[name] = None
            
    return model_evals

def model_info_to_dataloaders(model_info, graph_nodes):
    """Convert datasets to dataloaders for pytorch training."""
    data_loaders = []
    scalers_in = {}
    scalers_out ={}
    for node in graph.nodes:
        model = model_info[node]
        # print(model)
        x = model.train_in.to_numpy()
        if x.ndim == 1:
            x = x[:, np.newaxis]
        y = model.train_out.to_numpy()
        if y.ndim == 1:
            y = y[:, np.newaxis] 


        scaler_in = preprocessing.StandardScaler().fit(x)
        x_scaled = scaler_in.transform(x)

        scaler_out = preprocessing.StandardScaler().fit(y)
        y_scaled = scaler_out.transform(y)

        scalers_in[node]  = scaler_in
        scalers_out[node] = scaler_out
        # dataset = net.ArrayDataset(torch.Tensor(x), torch.Tensor(y))
        dataset = net.ArrayDataset(torch.Tensor(x_scaled), torch.Tensor(y_scaled))
        data_loaders.append(torch.utils.data.DataLoader(dataset, batch_size=x.shape[0], shuffle=False))        

    return data_loaders, scalers_in, scalers_out





if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='mfnet_cmd',
        description="Perform MFNETS",
    )

    parser.add_argument('input_file',  type=str, nargs=1, help='YAML input file')

    #########################
    ## Parse Arguments
    args = parser.parse_args()

    input_file = args.input_file[0]
    logger.info(f"Reading input Specs: {input_file}")
    try:
        with open(input_file, 'r') as file:
            input_spec = yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Cannot open input file {input_file}")
        exit(1)

    # main_model_schema = Config.model_json_schema()  # (1)!
    # print(json.dumps(main_model_schema, indent=2))  # (2)!        

    # print(input_spec)
    config_in = Config(**input_spec)
    # print(config.model_dump_json(indent=2))

        
    save_dir = pathlib.Path(input_spec['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(filename=save_dir / "log.log", level=logging.INFO)
    # logger.FileHandler(save_dir / "log.log", 'w+')

    with open(save_dir / "input.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config_in.model_dump(), f)
        
    # print(input_spec)

    model_info = parse_model_info(config_in)
    graph, roots = parse_graph(config_in, model_info)
    target_nodes = list(graph.nodes)
    num_nodes = len(target_nodes)
    logger.info(f"Node names: {target_nodes}")

    model_test_inputs = parse_evaluation_locations(config_in)


    # print(model_info)

    data_loaders, scalers_in, scalers_out = model_info_to_dataloaders(model_info, graph.nodes)    

    # exit(1)
    
    #########################
    ## Run algorithms
    if config_in.inference_type == "regression":
        logger.info("Performing Regression")
        #################
        ## Pytorch
        model = net.MFNetTorch(graph, roots, edge_type=config_in.graph.connection_type)
        logger.info(f"Model: {model}")
        
        ## Train
        loss_fns = net.construct_loss_funcs(model)        
        obj_func = model.train(data_loaders, target_nodes, loss_fns)
        logger.info(f"Model Loss: {obj_func}")


    elif config_in.inference_type == "bayes": # 

        logger.info("Running Bayesian Inference")
        noise_std = float(config_in.algorithm.noise_std)

        model = net_pyro.MFNetProbModel(graph, roots, noise_std=noise_std,
                                        edge_type=config_in.graph.connection_type)
        logger.info(f"Model: {model}")
        
        ## Algorithm parameters
        alg = config_in.algorithm.parametrization
        num_samples = config_in.algorithm.num_pred_samples

        ## Data setup
        if alg[:3] == 'svi':
            # SVI
            logger.info("Running Stochastic Variational Inference")
            
            adam_params = {"lr": 0.005, "betas": (0.95, 0.999)}
            num_steps = input_spec['algorithm']['num_optimization_steps']
                        
            optimizer = Adam(adam_params)
            if alg == 'svi-normal':
                logger.info("Approximating with an AutoNormal Guide")
                guide = AutoNormal(model)
            elif alg == 'svi-multinormal':
                guide = AutoMultivariateNormal(model)
            elif alg == 'svi-iafflow':
                hidden_dim = input_spec['algorithm']['iaf_params']['hidden_dim']
                num_transforms = input_spec['algorithm']['iaf_params']['num_transforms']
                guide = AutoIAFNormal(model,
                                      hidden_dim=hidden_dim,
                                      num_transforms=num_transforms)
            else:
                logger.info(f"Algorithm \'{alg}\' is not recognized")
                exit(1)


            logger.info(f"Number of steps = {num_steps}") 
            model.train_svi(data_loaders, target_nodes, guide, adam_params, max_steps=num_steps,
                            logger=logger)

            # logger.info(f"Iteration {step}\t Elbo loss: {elbo}")            
        elif alg == 'mcmc':
            raise InputError("Cannot Run MCMC yet")
            # MCMC
            num_chains = 1
            warmup = input_spec.algorithm.mcmc_params.burnin
            
            nuts_kernel = NUTS(model, jit_compile=False, full_mass=True)
            mcmc = MCMC(
                nuts_kernel,
                num_samples=num_samples,
                warmup_steps=warmup,
                num_chains=num_chains,
            )
            print("\n")
            mcmc.run(X, target_nodes, Y)
            print("\n")

            param_samples = mcmc.get_samples()

        else:
            logger.info(f"Algorithm \'{alg}\' is not recognized")
            exit(1)


    ## Evaluate and save to file
    logger.info(f"Evaluating Test data")
    if config_in.inference_type == "bayes":
        logger.info(f"Number of prediction samples: {num_samples}")
        
    for node in graph.nodes:
        test_pts = model_test_inputs[node]
        if test_pts is not None:
            logger.info(f"Evaluating model {node} at test points")

            for (fname, data) in test_pts:
                # print("fname = ", fname)
                x = torch.Tensor(data.to_numpy())
                x_scaled = torch.Tensor(scalers_in[node].transform(x))

                dirname =  model_info[node].output_dir
                os.makedirs(dirname, exist_ok=True)
                filename = os.path.join(dirname, f"{fname}.evals")
                
                if config_in.inference_type == "regression":
                    vals = model([x_scaled], [node])[0].detach().numpy()
                    vals_unscaled = scalers_out[node].inverse_transform(vals)
                    results = pd.DataFrame(vals_unscaled, columns=model_info[node].train_out.columns)
                    results.to_csv(filename, sep=' ', index=False)
                elif config_in.inference_type == "bayes":
                    if config_in.algorithm.noise_std_predict is not None:
                        model.update_noise_std(config_in.algorithm.noise_std_predict)
                    vals_pred = model.predict([x_scaled], [node], num_samples)[1][0].detach().numpy()
                    for jj in range(num_samples):
                        filename_jj = filename + f"_{jj}"
                        vals_unscaled = scalers_out[node].inverse_transform(vals_pred[jj, :, :])
                        results = pd.DataFrame(vals_unscaled, columns=model_info[node].train_out.columns)
                        results.to_csv(filename_jj, sep=' ', index=False)
                else:
                    raise InputError(f"Inference type {input_spec['inference_type']} unrecognized")
                        
