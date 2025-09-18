"""MFnets using pytorch."""

import itertools
import copy

import torch
import torch.utils
import torch.nn as nn
import networkx as nx

try:
    import queue.SimpleQueue as SimpleQueue
except ImportError:
    from queue import Queue as SimpleQueue

# dtype = torch.float
device = torch.device("cpu")

from .pce_model import PCE


__all__ = [
    "ArrayDataset",
    "MFNetTorch",
    "construct_loss_funcs",
    "FeedForwardNet",
    "FullyConnectedNNEdge",
    "EqualModelAverageEdge",
    "LinearScaleShift",
    "PolyScaleShift",
]


class ArrayDataset(torch.utils.data.Dataset):
    """Dataset from an array of data."""

    def __init__(self, x, y):
        """Initialize. Rows of x are different data points."""
        self.x = x
        self.y = y

    def __getitem__(self, key):
        """Get a row."""
        assert isinstance(key, int)
        return self.x[key, :], self.y[key, :]

    def __len__(self):
        """Get number of data points."""
        return self.x.size(dim=0)


class FeedForwardNet(nn.Module):
    """A flexible generalized model with fully connected layers"""

    def __init__(self, dim_in, dim_out, hidden_layer_sizes):
        super().__init__()
        assert len(hidden_layer_sizes) > 0, "must have at least 1 hidden layer"

        layers = [
            torch.nn.Linear(dim_in, hidden_layer_sizes[0], bias=True),
            nn.Tanh(),
        ]
        for ii in range(len(hidden_layer_sizes) - 1):
            layers.append(
                torch.nn.Linear(
                    hidden_layer_sizes[ii],
                    hidden_layer_sizes[ii + 1],
                    bias=True,
                )
            )
            layers.append(nn.Tanh())
        layers.append(
            torch.nn.Linear(hidden_layer_sizes[-1], dim_out, bias=True)
        )
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FullyConnectedNNEdge(nn.Module):
    """A fully connected NN to model edges/nodes for the MFNET"""

    def __init__(
        self, dim_in, dim_out, num_parent_vals_in, hidden_layer_sizes=[10]
    ):
        """Initialize the LinearScaleShift node for the MFNET

        f_{j} = NN(x, vec{parent_outputs})

        Parameters
        ----------
        dim_in : integer
            Dimension of the input space

        dim_out : integer
            Dimension of the output space

        num_parent_vals_in : integer
            Dimension of outputs of the parents
        """
        super().__init__()

        self.dim_out = dim_out
        self.num_parent_vals_in = num_parent_vals_in

        # edge model
        self.node = FeedForwardNet(
            dim_in + num_parent_vals_in, dim_out, hidden_layer_sizes
        )

    def forward(self, xinput, parent_vals):
        """Evaluate the fully connected NN model."""

        xin = torch.cat((xinput, parent_vals), dim=-1)
        return self.node(xin)


class EqualModelAverageEdge(nn.Module):
    """The edge model averages the parent model values and then learns a discrepancy."""

    def __init__(self, dim_in, dim_out, num_parents, model):
        """Initialize the EqualModeAvergeEdge

        f_{j} = \frac{1}{num_parents} \sum  f_{i} + delta_j(x)

        Parameters
        ----------
        dim_in : integer
            Dimension of the input space

        dim_out : integer
            Dimension of the output space

        num_parents : integer
            Number of parents.

        model : PytorchModule
            Represents the node model

        Note
        ----
        The parents *must* have equal number of outputs as the current node
        """
        super().__init__()

        self.dim_out = dim_out
        self.num_parents = num_parents

        # edge model
        self.node = copy.deepcopy(model)

    def forward(self, xinput, parent_vals):
        """Evaluate the fully connected NN model."""

        # print("xin.shape = ", xinput.shape)
        out1 = self.node(xinput)
        # print("parent_vals.shape = ", parent_vals.shape)
        # print("out1.shape = ", out1.shape)
        # print(self.dim_out)
        out2 = 0.0
        start = 0
        end = self.dim_out
        for ii in range(self.num_parents):
            out2 += parent_vals[:, start:end]
            start = end
            end = start + self.dim_out

        out2 /= float(self.num_parents)
        # print("out2 shape = ", out2.shape)
        # exit(1)
        out = out1 + out2

        return out


class LinearScaleShift(nn.Module):
    """A generalized scale and shift operator for the MFNET"""

    def __init__(self, dim_in, dim_out, num_parent_vals_in):
        """Initialize the LinearScaleShift node for the MFNET

        f_{j} = edge(x) \vec{parent_outputs} + node(x)

        Parameters
        ----------
        dim_in : integer
            Dimension of the input space

        dim_out : integer
            Dimension of the output space

        num_parent_vals_in : integer
            Dimension of outputs of the parents
        """
        super().__init__()

        self.dim_out = dim_out
        self.num_parent_vals_in = num_parent_vals_in

        # edge model
        self.edge = torch.nn.Linear(
            dim_in, dim_out * num_parent_vals_in, bias=True
        )

        # node model
        self.node = torch.nn.Linear(dim_in, dim_out, bias=True)

    def forward(self, xinput, parent_vals):
        """Evaluate the linear scale shift node model

        Parent values are expected to be (num_samples, flattened)
        """

        edge_eval = self.edge(
            xinput
        )  # should be num data x (num_out * num_parents_val_in)
        edge_eval = edge_eval.reshape(
            edge_eval.size(dim=0), self.dim_out, self.num_parent_vals_in
        )

        # print("input size = ", xinput.size())
        # print("parent_val size = ", parent_vals.size())
        edge_func = torch.einsum("ijk,ik->ij", edge_eval, parent_vals)

        node_func = self.node(xinput)

        return edge_func + node_func


class PolyScaleShift(nn.Module):
    """A generalized scale and shift operator for the MFNET"""

    def __init__(
        self, dim_in, dim_out, num_parent_vals_in, poly_order, poly_name
    ):
        """Initialize the PolynomialScaleShift node for the MFNET

        f_{j} = edge(x) \vec{parent_outputs} + node(x)

        node could be a polynomial. Edge is linear

        Parameters
        ----------
        dim_in : integer
            Dimension of the input space

        dim_out : integer
            Dimension of the output space

        num_parent_vals_in : integer
            Dimension of outputs of the parents

        poly_order: integer
            polynomial order

        poly_name: str 'Hermite' or 'Legendre'
        """
        super().__init__()

        self.dim_out = dim_out
        self.num_parent_vals_in = num_parent_vals_in

        # edge model
        self.edge = torch.nn.Linear(
            dim_in, dim_out * num_parent_vals_in, bias=True
        )

        # node model
        self.node = PCE(dim_in, dim_out, poly_order, poly_name)

    def forward(self, xinput, parent_vals):
        """Evaluate the linear scale shift node model

        Parent values are expected to be (num_samples, flattened)
        """

        edge_eval = self.edge(
            xinput
        )  # should be num data x (num_out * num_parents_val_in)
        edge_eval = edge_eval.reshape(
            edge_eval.size(dim=0), self.dim_out, self.num_parent_vals_in
        )

        # print("input size = ", xinput.size())
        # print("parent_val size = ", parent_vals.size())
        edge_func = torch.einsum("ijk,ik->ij", edge_eval, parent_vals)

        node_func = self.node(xinput)

        return edge_func + node_func


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


def make_graph_2():
    """Make a graph with two nodes (scale and shift).

    1 -> 2
    """
    graph = nx.DiGraph()

    # pnodes = torch.randn((2, 2), device=device, dtype=dtype)
    # pedges = torch.randn((1, 2), device=device, dtype=dtype)

    dim_in = 1
    graph.add_node(1, func=torch.nn.Linear(dim_in, 1, bias=True), dim_in=1)
    graph.add_node(2, func=torch.nn.Linear(dim_in, 1, bias=True), dim_out=1)
    graph.add_edge(
        1,
        2,
        func=torch.nn.Linear(dim_in, 1, bias=True),
        out_rows=1,
        out_cols=1,
        dim_in=1,
    )
    return graph, set([1])


def make_graph_2gen():
    """Make a graph with two nodes (generalized scale-shift)

    1 -> 2
    """
    graph = nx.DiGraph()

    # pnodes = torch.randn((2, 2), device=device, dtype=dtype)
    # pedges = torch.randn((1, 2), device=device, dtype=dtype)

    dim_in = 1
    dim_out1 = 1
    dim_out2 = 1
    graph.add_node(1, func=torch.nn.Linear(dim_in, dim_out1, bias=True))
    graph.add_node(2, func=LinearScaleShift(dim_in, dim_out2, dim_out1))
    graph.add_edge(1, 2)
    return graph, set([1])


def make_graph_2gen_nn():
    """Make a graph with two nodes (generalized neural network)

    1 -> 2
    """
    graph = nx.DiGraph()

    # pnodes = torch.randn((2, 2), device=device, dtype=dtype)
    # pedges = torch.randn((1, 2), device=device, dtype=dtype)

    dim_in = 1
    dim_out1 = 1
    dim_out2 = 1
    graph.add_node(1, func=torch.nn.Linear(dim_in, dim_out1, bias=True))
    graph.add_node(
        2,
        func=FullyConnectedNNEdge(
            dim_in, dim_out2, dim_out1, hidden_layer_sizes=[20, 20, 20]
        ),
    )
    graph.add_edge(1, 2)
    return graph, set([1])


def make_graph_2gen_nn_fixed():
    """Make a graph with two nodes (generalized neural network) with a fixed edge

    1 -> 2
    """
    graph = nx.DiGraph()

    # pnodes = torch.randn((2, 2), device=device, dtype=dtype)
    # pedges = torch.randn((1, 2), device=device, dtype=dtype)

    dim_in = 1
    dim_out1 = 1
    dim_out2 = 1
    graph.add_node(1, func=FeedForwardNet(dim_in, dim_out1, [10, 10]))
    graph.add_node(
        2,
        func=EqualModelAverageEdge(
            dim_in, dim_out2, 1, FeedForwardNet(dim_in, dim_out2, [10, 10])
        ),
    )

    graph.add_edge(1, 2)
    return graph, set([1])


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
        graph.add_node(
            node,
            func=torch.nn.Linear(dinput, 1, bias=True),
            dim_in=1,
            dim_out=1,
        )

    graph.add_edge(
        1,
        4,
        func=torch.nn.Linear(dinput, 1, bias=True),
        out_rows=1,
        out_cols=1,
        dim_in=1,
    )
    graph.add_edge(
        2,
        5,
        func=torch.nn.Linear(dinput, 1, bias=True),
        out_rows=1,
        out_cols=1,
        dim_in=1,
    )
    graph.add_edge(
        5,
        6,
        func=torch.nn.Linear(dinput, 1, bias=True),
        out_rows=1,
        out_cols=1,
        dim_in=1,
    )
    graph.add_edge(
        6,
        4,
        func=torch.nn.Linear(dinput, 1, bias=True),
        out_rows=1,
        out_cols=1,
        dim_in=1,
    )
    graph.add_edge(
        3,
        7,
        func=torch.nn.Linear(dinput, 1, bias=True),
        out_rows=1,
        out_cols=1,
        dim_in=1,
    )
    graph.add_edge(
        7,
        8,
        func=torch.nn.Linear(dinput, 1, bias=True),
        out_rows=1,
        out_cols=1,
        dim_in=1,
    )
    graph.add_edge(
        4,
        8,
        func=torch.nn.Linear(dinput, 1, bias=True),
        out_rows=1,
        out_cols=1,
        dim_in=1,
    )
    graph.add_edge(
        5,
        4,
        func=torch.nn.Linear(dinput, 1, bias=True),
        out_rows=1,
        out_cols=1,
        dim_in=1,
    )

    roots = set([1, 2, 3])
    return graph, roots


class MFNetTorch(nn.Module):
    """Multifidelity Network."""

    def __init__(self, graph, roots, edge_type="scale-shift"):
        """Initialize a multifidelity surrogate via a graph and its roots.

        Parameters
        ----------
        graph : networkx.graph
            The graphical representation of the MF network

        roots : list
            The ids of every root node in the network

        edge_type: string
            Either "scale-shift" or "general"

        """
        super(MFNetTorch, self).__init__()
        self.graph = graph
        self.roots = roots
        self.target_node = None
        self.edge_type = edge_type
        self.modules_list = nn.ModuleList()

        for node, func in self.graph.nodes.data("func"):
            self.modules_list.add_module(f"node{node}", func)

        for from_n, to_n, func in self.graph.edges.data("func"):
            # print("from_n", from_n, "to_n", to_n, "func ", func)
            if func is not None:
                self.modules_list.add_module(f"edge{from_n}->{to_n}", func)

    def zero_attributes(self):
        """Zero all attributes except 'func' and 'param'."""
        atts = ["eval", "parents_left"]
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

    def eval_node_recurse_(self, xinput, node):
        """Private function recursively evaluating all nodes"""

        parents = list(self.graph.predecessors(node))
        # print(f"Parents of {node}: {parents}")
        if len(parents) == 0:  # leaf node
            # print("\t leaf node: ", node)
            pval = self.graph.nodes[node]["func"](xinput)
            self.graph.nodes[node]["eval"] = pval
            # print("\t xinput shape = ", xinput.size())
            # print("\t eval shape = ", pval.size())
            return pval
        else:
            for parent in parents:
                self.eval_node_recurse_(xinput, parent)

            # need to flatten
            parent_evals = torch.cat(
                [self.graph.nodes[p]["eval"] for p in parents], dim=-1
            )
            # print("parent size = ", parent_evals.size())
            # print(xinput.size())
            pval = self.graph.nodes[node]["func"](xinput, parent_evals)
            self.graph.nodes[node]["eval"] = pval
            return pval

    def eval_target_node_general_(self, xinput, target_node):
        """Evaluate the surrogate output at target_node.

        Parameters
        ----------
        xinput :  np.ndarray (nsamples, ndim)
            The independent variables of the model at which to evaluate target node

        target_node : integer
            The id of the nodes to evaluate

        Returns
        -------
        This function adds the following attributes to the underlying graph

        eval :
            stores the evaluation of the function represented by the
            particular node / edge the evaluations at the nodes are
            cumulative (summing up all ancestors) whereas the edges are local

        """

        self.zero_attributes()
        eval = self.eval_node_recurse_(xinput, target_node)
        return eval

    def eval_target_node_scale_shift_(self, xinput, target_node):
        """Evaluate the surrogate output at target_node.

        Parameters
        ----------
        xinput :  np.ndarray (nsamples, ndim)
            The independent variables of the model at which to evaluate target node

        target_node : integer
            The id of the nodes to evaluate

        Returns
        -------
        This function adds the following attributes to the underlying graph

        eval :
            stores the evaluation of the function represented by the
            particular node / edge the evaluations at the nodes are
            cumulative (summing up all ancestors) whereas the edges are local

        parents-left : set
            internal attribute needed for accounting

        """
        self.zero_attributes()
        anc = nx.ancestors(self.graph, target_node)
        anc_and_target = anc.union(set([target_node]))
        relevant_root_nodes = anc.intersection(self.roots)

        # Evaluate the target nodes and all ancestral nodes and put the root
        # nodes in a queue
        queue = SimpleQueue()
        for node in anc_and_target:
            pval = self.graph.nodes[node]["func"](xinput)

            # print(f"node {node}, pval size = {pval.size()}")
            self.graph.nodes[node]["eval"] = pval
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
                    pval = self.graph.edges[node, child]["func"](xinput)

                    # print("\n")
                    # print("node = ", node)
                    # print("child = ", child)
                    # print("pval shape", pval.shape)
                    # print(self.graph.edges[node, child]['out_rows'],  self.graph.edges[node, child]['out_cols'])
                    pval = pval.reshape(
                        pval.size(dim=0),
                        self.graph.edges[node, child]["out_rows"],
                        self.graph.edges[node, child]["out_cols"],
                    )

                    # print("feval.shape = ", feval.shape)
                    # print("pval.reshaped = ", pval.shape)

                    rho_f = torch.einsum("ijk,ik->ij", pval, feval)

                    # print(f"child = {child}, eval prev size = {self.graph.nodes[child]['eval'].size()}, rho_f.size = {rho_f.size()}")
                    # self.graph.nodes[child]['eval'] += feval * pval
                    # print("pval shape =
                    self.graph.nodes[child]["eval"] += rho_f

                    # print("child shape = ", self.graph.nodes[child]['eval'].shape)

                    self.graph.edges[node, child]["eval"] = pval

                    self.graph.nodes[child]["parents_left"].remove(node)

                    if self.graph.nodes[child]["parents_left"] == set():
                        queue.put(child)

        return self.graph.nodes[target_node]["eval"]

    def eval_target_node(self, xinput, target_node):
        if self.edge_type == "scale-shift":
            return self.eval_target_node_scale_shift_(xinput, target_node)
        elif self.edge_type == "general":
            return self.eval_target_node_general_(xinput, target_node)
        else:
            raise Exception(f"Edge type {self.edge_type} unrecognized")

    def forward(self, xinput, target_nodes):
        """Evaluate the surrogate output at target_node.

        Parameters
        ----------
        xinput : List of np.ndarrays (nsamples,nparams)
            The independent variables of the model at which to evaluate target nodes

        target_node : List of integers
            The id of the nodes to evaluate

        Returns
        -------
        list of evaluations for each of the target nodes
        """
        vals = [
            self.eval_target_node(x, t) for x, t in zip(xinput, target_nodes)
        ]
        return vals

    def eval_loss(self, data, targets, loss_fns):
        """Evaluate loss function."""
        loss = 0
        x = []
        y = []
        for dat in data:
            # should only be one batch because only LBFGS can be used
            for batch, (X, Y) in enumerate(dat):
                x.append(X)
                y.append(Y)

        pred = self(x, targets)
        loss = 0
        for p, ydat, loss_fn in zip(pred, y, loss_fns):
            loss += loss_fn(p.reshape(ydat.shape), ydat)

        return loss

    def train(self, data, targets, loss_fns, max_iter=100):
        """Train the model."""
        optimizer = torch.optim.LBFGS(
            self.parameters(),
            max_iter=max_iter,
            tolerance_grad=1e-15,
            tolerance_change=1e-15,
            history_size=500,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            self.zero_attributes()
            loss = self.eval_loss(data, targets, loss_fns)
            loss.backward(retain_graph=True)
            return loss

        self.zero_grad()

        optimizer.step(closure)
        return self.eval_loss(data, targets, loss_fns)


def generate_data(model, ndata):
    """Generate data."""
    ndata_lf, ndata_hf = ndata
    xlf = torch.rand(ndata_lf, 1) * 6 - 3
    xhf = torch.rand(ndata_hf, 1) * 6 - 3
    with torch.no_grad():
        y = model([xlf, xhf], [1, 2])

    d1 = ArrayDataset(xlf, y[0])
    d2 = ArrayDataset(xhf, y[1])
    data = (d1, d2)
    return data


def generate_data_1model(model, ndata):
    """Generate data."""

    x = torch.rand(ndata, 1) * 6 - 3
    with torch.no_grad():
        y = model([x], [1])

    d1 = ArrayDataset(x, y[0])
    data = [d1]
    return data


def construct_loss_funcs(model):
    """Create loss functions."""
    num_nodes = len(model.graph.nodes)
    return [torch.nn.MSELoss() for _ in range(num_nodes)]


def plot_funcs(model, x, data=None, title=None):
    """Plot the univariate functions."""
    nnodes = len(model.graph.nodes)
    with torch.no_grad():
        y = model([x] * nnodes, list(range(1, nnodes + 1)))
        # ylf = model(x)

        # yhf = model(x)

    plt.figure()
    plt.plot(x, y[0], label="low-fidelity")
    if nnodes > 1:
        plt.plot(x, y[1], label="high-fidelity")

    if data != None:
        plt.plot(data[0].x, data[0].y, "ro", label="Low-fidelity Data")
        if nnodes > 1:
            plt.plot(data[1].x, data[1].y, "ko", label="High-fidelity Data")

    if title is not None:
        plt.title(title)
    plt.legend()


def run_scale_shift():
    torch.manual_seed(1)
    graph, root = make_graph_2()

    model = MFNetTorch(graph, root)

    # for param in model.named_parameters():
    #     print(param)

    x = torch.linspace(-3, 3, 10).reshape(10, 1)
    plot_funcs(model, x)

    data = generate_data(model, [20, 20])
    data_loaders = [
        torch.utils.data.DataLoader(d, batch_size=len(d), shuffle=False)
        for d in data
    ]

    graph2, root2 = make_graph_2()
    model_trained = MFNetTorch(graph2, root2)
    plot_funcs(model_trained, x)

    loss_fns = construct_loss_funcs(model_trained)
    model_trained.train(data_loaders, [1, 2], loss_fns)

    plot_funcs(model_trained, x, data)

    with torch.no_grad():
        loss2 = model_trained.eval_loss(data_loaders, [1, 2], loss_fns)
        print("loss2 = ", loss2)
        print(model_trained)
        print("Trained parameters")
        print(torch.nn.utils.parameters_to_vector(model_trained.parameters()))
        print("True parameters")
        print(torch.nn.utils.parameters_to_vector(model.parameters()))

    plt.show()


def run_generalized():
    torch.manual_seed(1)
    graph, root = make_graph_2gen()

    model = MFNetTorch(graph, root, edge_type="general")

    # val = model.eval_target_node_general_(torch.Tensor([0.2]), 2)
    # print("val = ", val)
    # exit()

    x = torch.linspace(-3, 3, 10).reshape(10, 1)
    plot_funcs(model, x, title="True Model")

    data = generate_data(model, [20, 20])
    data_loaders = [
        torch.utils.data.DataLoader(d, batch_size=len(d), shuffle=False)
        for d in data
    ]

    graph2, root2 = make_graph_2gen()
    model_trained = MFNetTorch(graph2, root2, edge_type="general")
    plot_funcs(model_trained, x, title="Un-trained initial mfnet")

    loss_fns = construct_loss_funcs(model_trained)
    model_trained.train(data_loaders, [1, 2], loss_fns)

    plot_funcs(model_trained, x, data, title="Trained mfnet")

    with torch.no_grad():
        loss2 = model_trained.eval_loss(data_loaders, [1, 2], loss_fns)
        print("loss2 = ", loss2)
        print(model_trained)
        print("Trained parameters")
        print(torch.nn.utils.parameters_to_vector(model_trained.parameters()))
        print("True parameters")
        print(torch.nn.utils.parameters_to_vector(model.parameters()))

    plt.show()


def run_generalized_nn():
    torch.manual_seed(1)
    graph, root = make_graph_2gen()  # train with non nn

    model = MFNetTorch(graph, root, edge_type="general")

    # val = model.eval_target_node_general_(torch.Tensor([0.2]), 2)
    # print("val = ", val)
    # exit()

    x = torch.linspace(-3, 3, 10).reshape(10, 1)
    plot_funcs(model, x, title="True Model")

    data = generate_data(model, [20, 20])
    data_loaders = [
        torch.utils.data.DataLoader(d, batch_size=len(d), shuffle=False)
        for d in data
    ]

    graph2, root2 = make_graph_2gen_nn()
    model_trained = MFNetTorch(graph2, root2, edge_type="general")
    loss_fns = construct_loss_funcs(model_trained)
    plot_funcs(model_trained, x, title="Un-trained initial mfnet")

    model_trained.train(data_loaders, [1, 2], loss_fns)

    plot_funcs(model_trained, x, data, title="Trained mfnet")

    with torch.no_grad():
        loss2 = model_trained.eval_loss(data_loaders, [1, 2], loss_fns)
        print("loss2 = ", loss2)
        print(model_trained)
        # print("Trained parameters")
        # print(torch.nn.utils.parameters_to_vector(model_trained.parameters()))
        # print("True parameters")
        # print(torch.nn.utils.parameters_to_vector(model.parameters()))

    plt.show()


def run_generalized_nn_fixed_edge():
    torch.manual_seed(1)
    graph, root = make_graph_2gen_nn_fixed()  # train with non nn

    model = MFNetTorch(graph, root, edge_type="general")

    # val = model.eval_target_node_general_(torch.Tensor([0.2]), 2)
    # print("val = ", val)
    # exit()

    x = torch.linspace(-3, 3, 10).reshape(10, 1)
    plot_funcs(model, x, title="True Model")

    data = generate_data(model, [20, 20])
    data_loaders = [
        torch.utils.data.DataLoader(d, batch_size=len(d), shuffle=False)
        for d in data
    ]

    graph2, root2 = make_graph_2gen_nn_fixed()
    model_trained = MFNetTorch(graph2, root2, edge_type="general")
    loss_fns = construct_loss_funcs(model_trained)
    plot_funcs(model_trained, x, title="Un-trained initial mfnet")

    model_trained.train(data_loaders, [1, 2], loss_fns)

    plot_funcs(model_trained, x, data, title="Trained mfnet")

    with torch.no_grad():
        loss2 = model_trained.eval_loss(data_loaders, [1, 2], loss_fns)
        print("loss2 = ", loss2)
        print(model_trained)
        # print("Trained parameters")
        # print(torch.nn.utils.parameters_to_vector(model_trained.parameters()))
        # print("True parameters")
        # print(torch.nn.utils.parameters_to_vector(model.parameters()))

    plt.show()


def run_single_fidelity():
    torch.manual_seed(1)
    graph, root = make_graph_single()  # train with non nn

    model = MFNetTorch(graph, root, edge_type="general")

    # val = model.eval_target_node_general_(torch.Tensor([0.2]), 2)
    # print("val = ", val)
    # exit()

    x = torch.linspace(-3, 3, 10).reshape(10, 1)
    plot_funcs(model, x, title="True Model")

    data = generate_data_1model(model, 20)
    data_loaders = [
        torch.utils.data.DataLoader(d, batch_size=len(d), shuffle=False)
        for d in data
    ]

    graph2, root2 = make_graph_single()
    model_trained = MFNetTorch(graph2, root2, edge_type="general")
    loss_fns = construct_loss_funcs(model_trained)
    plot_funcs(model_trained, x, title="Un-trained initial mfnet")

    model_trained.train(data_loaders, [1], loss_fns)

    plot_funcs(model_trained, x, data, title="Trained mfnet")

    with torch.no_grad():
        loss2 = model_trained.eval_loss(data_loaders, [1], loss_fns)
        print("loss2 = ", loss2)
        print(model_trained)
        # print("Trained parameters")
        # print(torch.nn.utils.parameters_to_vector(model_trained.parameters()))
        # print("True parameters")
        # print(torch.nn.utils.parameters_to_vector(model.parameters()))

    plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # run_scale_shift()
    # run_generalized()
    # run_generalized_nn()
    # run_single_fidelity()
    run_generalized_nn_fixed_edge()
