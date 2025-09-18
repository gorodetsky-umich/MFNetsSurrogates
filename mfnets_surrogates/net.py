"""Multifidelity Surrogate Modeling via Directed Networks.

Author: Alex Gorodetsky, goroda@umich.edu

Copyright (c) 2020, Alex Gorodetsky

License: MIT
"""

import copy

try:
    import queue.SimpleQueue as SimpleQueue
except ImportError:
    from queue import Queue as SimpleQueue
from functools import partial

import numpy as np

import networkx as nx
# print("NetworkX Version = ", nx.__version__)

import scipy.optimize as sciopt

from functools import partial

pyapprox_is_installed = True
try:
    from pyapprox.l1_minimization import nonlinear_basis_pursuit, lasso
except ModuleNotFoundError:
    pyapprox_is_installed = False


def least_squares(target, predicted, std=1e0):
    """Evaluate the least squares objective function.

    Parameters
    ----------
    target : np.ndarray (nobs)
        The observations

    predicted : np.ndarray (nobs)
        The model predictions of the observations

    std : float
        The standard deviation of the I.I.D noise

    Returns
    -------
    obj : float
        The value of the least squares objective function

    grad : np.ndarray (nobs)
        The gradient of ``obj``
    """
    resid = target - predicted
    obj = np.dot(resid, resid) * 0.5 * std**-2
    grad = -(std**-2) * resid
    return obj, grad


def lin(param, xinput):
    """Compute with a linear parametric model.

    Parameters
    ----------
    param : np.ndarray (nparams)
       The parameters of the model

    xinput : np.ndarray (nsamples,nparams)
       The independent variables of the model

    Returns
    -------
    vals : np.ndarray (nsamples)
      Evaluation of the linear model

    grad : np.ndarray (nsamples,nparams)
      gradient of the linear model with respect to the model parameters
    """
    one = np.ones((xinput.shape[0], 1))
    grad = np.concatenate((one, xinput), axis=1)
    return param[0] + np.dot(param[1:], xinput.T), grad


def monomial_1d_lin(param, xinput):
    """Compute with a Linear Model with Monomial basis functions.

    p[0]+sum(x**p[1:])

    Parameters
    ----------
    param : np.ndarray (nparams)
       The parameters of the model

    xinput : np.ndarray (nsamples,nparams)
       The independent variables of the model

    Returns
    -------
    vals : np.ndarray (nsamples)
        Evaluation of the linear model

    grad : np.ndarray (nsamples,nparams)
      gradient of the linear model with respect to the model parameters
    """
    basis = xinput ** np.arange(param.shape[0])[np.newaxis, :]
    vals = basis.dot(param)
    grad = basis
    return vals, grad


class MFSurrogate:
    """Multifidelity surrogate.

    A surrogate consists of a graph where the edges and nodes are functions
    Each node represents a particular information sources and the edges
    describe the relationships between the information sources
    """

    def __init__(self, graph, roots, copy_data=True):
        """Initialize a multifidelity surrogate via a graph and its roots.

        Parameters
        ----------
        graph : networkx.graph
            The graphical representation of the MF network

        roots : list
            The ids of every root node in the network

        copy_data : boolean
           True - perform a deep copy of graph and roots
           False - just use a shallow copy (this is dangerous as many functions
           change the internal shape)
        """
        if copy_data:
            self.roots = copy.deepcopy(roots)
            self.graph = copy.deepcopy(graph)
        else:
            self.roots = roots
            self.graph = graph
            print("warning MFSurrogate not copying data. proceed with caution")
        self.nparam = graph_to_vec(self.graph).shape[0]

    def get_nparam(self):
        """Get the number of parameters parameterizing the graph.

        Returns
        -------
        nparam : integer
            The number of all the unknown parameters in the MF surrogate"""
        return self.nparam

    def forward(self, xinput, target_node):
        """Evaluate the surrogate output at target_node by considering the
        subgraph of all ancestors of this node

        Parameters
        ----------
        xinput : np.ndarray (nsamples,nparams)
            The independent variables of the model

        target_node : integer
            The id of the node under consideration

        Returns
        -------
        This function adds the following attributes to the underlying graph

        eval :
            stores the evaluation of the function represented by the
            particular node / edge the evaluations at the nodes are
            cumulative (summing up all ancestors) whereas the edges are local

        pre-grad : np.ndarray
            internal attribute needed for accounting

        parents-left : set
            internal attribute needed for accounting

        """
        anc = nx.ancestors(self.graph, target_node)
        anc_and_target = anc.union(set([target_node]))
        relevant_root_nodes = anc.intersection(self.roots)

        # Evaluate the target nodes and all ancestral nodes and put the root
        # nodes in a queue
        queue = SimpleQueue()
        for node in anc_and_target:
            pval, pgrad = self.graph.nodes[node]["func"](
                self.graph.nodes[node]["param"], xinput
            )
            self.graph.nodes[node]["eval"] = pval
            self.graph.nodes[node]["pre_grad"] = pgrad
            self.graph.nodes[node]["parents_left"] = set(
                self.graph.predecessors(node)
            )

            if node in relevant_root_nodes:
                queue.put(node)

        while not queue.empty():
            node = queue.get()
            feval = self.graph.nodes[node]["eval"]
            for child in self.graph.successors(node):
                if child in anc_and_target:
                    pval, pgrad = self.graph.edges[node, child]["func"](
                        self.graph.edges[node, child]["param"], xinput
                    )

                    self.graph.nodes[child]["eval"] += feval * pval

                    ftile = np.tile(
                        feval.reshape((feval.shape[0], 1)), (1, pgrad.shape[1])
                    )
                    self.graph.edges[node, child]["pre_grad"] = ftile * pgrad
                    self.graph.edges[node, child]["eval"] = pval

                    self.graph.nodes[child]["parents_left"].remove(node)

                    if self.graph.nodes[child]["parents_left"] == set():
                        queue.put(child)

        return self.graph.nodes[node]["eval"], anc

    def backward(self, target_node, deriv_pass, ancestors=None):
        """Perform a backward computation to compute the derivatives for all
        parameters that affect the target node

        Parameters
        ----------
        target_node : integer
            The id of the node under consideration

        deriv_pass : np.ndarray (nparams)
            A gradient vector

        Returns
        -------
        derivative : np.ndarray(nparams)
            A vector containing the derivative of all parameters
        """

        if ancestors is None:
            ancestors = nx.ancestors(self.graph, target_node)

        anc_and_target = ancestors.union(set([target_node]))

        # Evaluate the node
        self.graph.nodes[target_node]["pass_down"] = deriv_pass

        # Gradient with respect to beta
        self.graph.nodes[target_node]["derivative"] = np.dot(
            self.graph.nodes[target_node]["pass_down"],
            self.graph.nodes[target_node]["pre_grad"],
        )

        queue = SimpleQueue()
        queue.put(target_node)

        for node in ancestors:
            self.graph.nodes[node]["children_left"] = set(
                self.graph.successors(node)
            ).intersection(anc_and_target)
            self.graph.nodes[node]["pass_down"] = 0.0
            self.graph.nodes[node]["derivative"] = 0.0

        while not queue.empty():
            node = queue.get()

            pass_down = self.graph.nodes[node]["pass_down"]
            for parent in self.graph.predecessors(node):
                self.graph.nodes[parent]["pass_down"] += (
                    pass_down * self.graph.edges[parent, node]["eval"]
                )
                self.graph.edges[parent, node]["derivative"] = np.dot(
                    pass_down, self.graph.edges[parent, node]["pre_grad"]
                )
                self.graph.nodes[parent]["derivative"] += np.dot(
                    pass_down * self.graph.edges[parent, node]["eval"],
                    self.graph.nodes[parent]["pre_grad"],
                )

                self.graph.nodes[parent]["children_left"].remove(node)
                if self.graph.nodes[parent]["children_left"] == set():
                    queue.put(parent)

        return self.get_derivative()

    def set_param(self, param):
        """Set the parameters for the graph

        Parameters
        ----------
        param : np.ndarray (nparams)
            A flattened array containing all parameters of the MF surrogate
        """
        self.graph = vec_to_graph(param, self.graph, attribute="param")

    def get_param(self):
        """Get the parameters of the graph"""
        return graph_to_vec(self.graph, attribute="param")

    def get_derivative(self):
        """Get a vector of derivatives of each parameter"""
        return graph_to_vec(self.graph, attribute="derivative")

    def get_evals(self):
        """Get the evaluations at each node"""
        return [self.graph.nodes[node]["eval"] for node in self.graph.nodes]

    def zero_derivatives(self):
        """Set all the derivative attributes to zero

        Used prior to computing a new derivative to clear out previous sweep
        """
        self.graph = vec_to_graph(
            np.zeros(self.nparam), self.graph, attribute="derivative"
        )

    def zero_attributes(self):
        """Zero all attributes except 'func' and 'param'"""

        atts = [
            "eval",
            "pass_down",
            "pre_grad",
            "derivative",
            "children_left",
            "parents_left",
        ]
        for att in atts:
            for node in self.graph.nodes:
                try:
                    self.graph.nodes[node][att] = 0.0
                except:  # what exception is it?
                    continue
            for edge in self.graph.edges:
                try:
                    self.graph.edges[edge][att] = 0.0
                except:  # what exception is it?
                    continue

    def train(
        self,
        param0in,
        nodes,
        xtrain,
        ytrain,
        stdtrain,
        niters=200,
        func=least_squares,
        verbose=False,
        warmup=True,
        opts=dict(),
    ):
        """Train the multifidelity surrogate.

        This is the main entrance point for data-driven training.

        Parameters
        ----------
        param0in : np.ndarray (nparams)
            The initial guess for the parameters

        nodes : list
            A list of nodes for which data is available

        xtrain : list
            A list of input features for each node in *nodes*

        ytrain : list
            A list of output values for each node in *nodes*

        stdtrain : float
            The standard devaition for data for each node in *nodes*

        niters : integer
            The number of optimization iterations

        func : callable
            A scalar valued objective function with the signature

            ``func(target, predicted) ->  val (float), grad (np.ndarray)``

            where ``target`` is a np.ndarray of shape (nobs)
            containing the observations and ``predicted`` is a np.ndarray of
            shape (nobs) containing the model predictions of the observations

        verbose : integer
            The verbosity level

        warmup : boolean
            Specify whether or not to progressively find a good guess before
            optimizing

        opts : dictionary
            Specify the type of loss function: 'lstsq' for squared error, anything else for L1 regularization, 'lambda' for regularization value

        Returns
        -------
        Upon completion of this function, the parameters of the graph are set
        to the values that best fit the data, as defined by *func*
        """
        bounds = list(zip([-np.inf] * self.nparam, [np.inf] * self.nparam))
        param0 = copy.deepcopy(param0in)

        # options = {'maxiter':20, 'disp':False, 'gtol':1e-10, 'ftol':1e-18}
        options = {"maxiter": 20, "disp": False, "gtol": 1e-10, "ftol": 1e-18}

        # Warming up
        if warmup is True:
            for node in nodes:
                node_list = nodes[node - 1 : node]
                x_list = xtrain[node - 1 : node]
                y_list = ytrain[node - 1 : node]
                std_list = stdtrain[node - 1 : node]

                res = sciopt.minimize(
                    optimize_obj,
                    param0,
                    args=(func, self, node_list, x_list, y_list, std_list),
                    method="L-BFGS-B",
                    jac=True,
                    bounds=bounds,
                    options=options,
                )

                param0 = res.x
                for ii in range(self.nparam):
                    if np.abs(param0[ii]) > 1e-10:
                        bounds[ii] = (param0[ii] - 1e-10, param0[ii] + 1e-10)
                # print("bounds", bounds)

        # Final Training
        lossfunc = opts.get("lossfunc", "lstsq")
        if lossfunc == "lstsq":
            options = {"maxiter": niters, "disp": verbose, "gtol": 1e-10}
            res = sciopt.minimize(
                optimize_obj,
                param0,
                args=(func, self, nodes, xtrain, ytrain, stdtrain),
                method="L-BFGS-B",
                jac=True,
                options=options,
            )
        elif pyapprox_is_installed is True:
            obj = partial(
                optimize_obj,
                optf=least_squares,
                graph=self,
                nodes=nodes,
                xin_l=xtrain,
                yin_l=ytrain,
                std_l=stdtrain,
            )
            lamda = opts["lambda"]
            options = {
                "ftol": 1e-12,
                "disp": False,
                "maxiter": 1e3,
                "method": "slsqp",
            }
            l1_coef, res = lasso(obj, True, None, param0, lamda, options)
            # res.x includes slack variables so remove these
            res.x = l1_coef
        else:
            raise Exception("Specified loss is not accepted")

        self.set_param(res.x)
        return self


def graph_to_vec(graph, attribute="param"):
    """Extract the multifidelity surrogate parameters from the graph

    Parameters
    ----------
    graph : networkx.graph
        The graphical representation of the MF network

    Returns
    -------
        vec : np.ndarray (nparams)
        A flattened array containing all the parameters of the MF network
    """
    nodes = graph.nodes
    node_params_dict = nx.get_node_attributes(graph, attribute)
    node_params = np.concatenate([node_params_dict[n] for n in nodes])

    edges = graph.edges
    edge_params_dict = nx.get_edge_attributes(graph, attribute)
    edge_params = np.concatenate([edge_params_dict[e] for e in edges])

    return np.concatenate((node_params, edge_params))


def vec_to_graph(vec, graph, attribute="param"):
    """
    Update the parameters of a multifidelity surrogate

    Parameters
    ----------
    vec : np.ndarray (nparams)
        A flattened array containing all the parameters of the MF network

    graph : networkx.graph
        The graphical representation of the MF network

    Returns
    -------
    graph : networkx.graph
        The updated graphical representation of the MF network with the
        parameter values given by ``vec``.
    """
    nodes = graph.nodes
    ind = 0
    for node in nodes:
        try:
            offset = graph.nodes[node][attribute].shape[0]
        except:  # What is the exception?
            offset = graph.nodes[node]["param"].shape[0]
        graph.nodes[node][attribute] = vec[ind : ind + offset]
        ind = ind + offset

    edges = graph.edges
    for edge in edges:
        try:
            offset = graph.edges[edge][attribute].shape[0]
        except:  # What is the exception?
            offset = graph.edges[edge]["param"].shape[0]

        graph.edges[edge][attribute] = vec[ind : ind + offset]
        ind = ind + offset

    return graph


def optimize_obj(param, optf, graph, nodes, xin_l, yin_l, std_l):
    """Composite optimization objective for a set of nodes

    Parameters
    ----------
    param : np.ndarray (nparams)
        The parameter values at which to compute the objective value and
        gradient

    optf : callable
        A scalar valued objective function with the signature

        ``optf(target, predicted) ->  float``

        where ``target`` is a np.ndarray of shape (nobs)
        containing the observations and ``predicted`` is a np.ndarray of
        shape (nobs) containing the model predictions of the observations

    graph : networkx.graph
        The graphical representation of the MF network

    nodes : list
        A list of nodes for which data is available

    xin_l : list
        A list of input features for each node in *nodes*

    yin_l : list
        A list of output values for each node in *nodes*

    std_l : float
        The standard devaition for data for each node in *nodes*

    Returns
    -------
    final_val : float
        The value of the least squares objective function

    final_derivative : np.ndarray (nobs)
        The gradient of ``obj``
    """
    final_derivative = np.zeros((param.shape[0]))
    final_val = 0.0

    # print("nodes =", nodes)
    for node, xin, yout, std in zip(nodes, xin_l, yin_l, std_l):
        graph.zero_attributes()
        graph.zero_derivatives()
        graph.set_param(param)
        val, anc = graph.forward(xin, node)

        ## optimization function takes
        new_val, obj_grad = optf(yout, val, std=std)
        derivative = graph.backward(node, obj_grad, ancestors=anc)

        final_val += new_val
        final_derivative += derivative

    return final_val, final_derivative


def learn_obj(param, graph, node, x, y, std):
    """
    Return the least squares learning objective function

    Parameters
    ----------
    param : np.ndarray (nparams)
        The parameter values at which to compute the objective value and
        gradient

    graph : networkx.graph
        The graphical representation of the MF network

    nodes : list
        A list of nodes for which data is available

    x : list
        A list of input features for each node in *nodes*

    y : list
        A list of output values for each node in *nodes*

    std : float
        The standard devaition for data for each node in *nodes*

    Returns
    -------
    final_val : float
        The value of the least squares objective function
    """
    graph.set_param(param)
    predict, _ = graph.forward(x, node)
    val, _ = least_squares(y, predict, std=std)

    return val


def learn_obj_grad(param, graph, node, x, y, std):
    """
    Return the gradient of the least squares learning objective function

    Parameters
    ----------
    param : np.ndarray (nparams)
        The parameter values at which to compute the objective value and
        gradient

    graph : networkx.graph
        The graphical representation of the MF network

    nodes : list
        A list of nodes for which data is available

    x : list
        A list of input features for each node in *nodes*

    y : list
        A list of output values for each node in *nodes*

    std : float
        The standard devaition for data for each node in *nodes*

    Returns
    -------
    final_derivative : np.ndarray (nobs)
        The gradient of the objective
    """
    graph.zero_derivatives()
    graph.set_param(param)
    predict, anc = graph.forward(x, node)
    _, grad = least_squares(y, predict, std=std)
    graph.backward(node, grad, ancestors=anc)
    return graph.get_derivative()


def learn_obj_grad_both(param, graph, node, x, y, std):
    """
    Return the value and gradient of the least squares learning objective
    function

    Parameters
    ----------
    param : np.ndarray (nparams)
        The parameter values at which to compute the objective value and
        gradient

    graph : networkx.graph
        The graphical representation of the MF network

    nodes : list
        A list of nodes for which data is available

    x : list
        A list of input features for each node in *nodes*

    y : list
        A list of output values for each node in *nodes*

    std : float
        The standard devaition for data for each node in *nodes*

    Returns
    -------
    val : float
        The value of the least squares objective function

    final_derivative : np.ndarray (nobs)
        The gradient of the objective
    """

    graph.zero_derivatives()
    graph.set_param(param)
    predict, A = graph.forward(x, node)
    # print("predict = ", predict.shape)
    # print("y = ", y.shape)
    val, grad = least_squares(y, predict, std=std)
    graph.backward(node, grad, ancestors=A)

    return val, graph.get_derivative()


# --------------------------------#
# Functions useful for debugging #


def identity(ynotused, predict, std=None):
    """Identity output function"""
    # f(predict) = predict
    return predict[0], np.ones(predict.shape)


def identity_obj(param, graph, node, x):
    graph.set_param(param)
    predict, _ = graph.forward(x, node)
    return predict[0]


def identity_obj_grad(param, graph, node, x):
    graph.zero_derivatives()
    graph.set_param(param)
    predict, A = graph.forward(x, node)
    # print("predict = ", predict)
    _, pass_back = identity(None, predict, std=None)
    graph.backward(node, pass_back, ancestors=A)

    return graph.get_derivative()
