"""Probabilistic MFNets."""

import functools
import itertools
import logging

import pandas as pd

import torch
import numpy as np

from pyro.infer import Predictive
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO

import pyro
import pyro.distributions as dist
from pyro.nn.module import to_pyro_module_

from mfnets_surrogates.net_torch import MFNetTorch


__all__ = [
    "MFNetProbModel",
]


def nestgetattr(obj, name):
    """Nested getattr."""

    def func(current, new):
        return getattr(current, new)

    res = functools.reduce(func, name.split("."), obj)
    return res


def nestsetattr(obj, name, value):
    """Nested setattr."""

    def func(current, new):
        return getattr(current, new)

    names = name.split(".")
    if len(names) == 1:
        setattr(obj, name, value)
    else:
        last_obj = functools.reduce(func, names[:-1], obj)
        setattr(last_obj, names[-1], value)


def convert_to_pyro(model):
    """Convert an MFNetTorch model to a Pyro Model.

    Also sets all internal parameters to uncertain random variables (priors)
    """
    to_pyro_module_(model)
    names = []
    for name, param in model.named_parameters():
        names.append(name)

    for name in names:
        # print("NAME = ", name)
        param = nestgetattr(model, name)
        size = list(param.size())
        dim = param.dim()
        # print(size, dim)
        new_param = pyro.nn.PyroSample(
            dist.Normal(0, 1).expand(size).to_event(dim)
        )
        # print(f"\t size={size}, dim={dim}")
        nestsetattr(model, name, new_param)


def param_samples_to_pandas(samples):
    """Convert samples of parameters from pyro dictionary output to pandas dataframe.


    Parameters
    ----------
    samples: dictionary
             output of pyro Predictive call that containts a list of model parameters and their samples

    Returns
    -------
    df_params: Pandas Dataframe
      A pandas dataframe that extracts outputs related to the model parameters, does some minimial
      checks and processing and returns a dataframe

    Notes
    -----
    See MFNetProbModel.predict  for usage

    """
    param_dict = {}
    for key, val in samples.items():
        val = val.squeeze()
        if val.dim() == 1:
            val = val.reshape(val.size(dim=0), 1)

        name = key.split("modules_list.")
        name = name[-1]
        for idx in itertools.product(
            *map(range, val.shape[1:])
        ):  # first dimension is samples
            idx_str = "[{}]".format(",".join(map(str, idx)))
            new_name = name + idx_str
            if len(idx) == 1:
                v = val[:, idx[0]]
            elif len(idx) == 2:
                v = val[:, idx[0], idx[1]]
            else:
                raise ValueError(
                    f"Param Sample Key {key} has value with dimension larger than 2"
                )
            param_dict[new_name] = v.detach().numpy()

    df_params = pd.DataFrame(param_dict)
    return df_params


class MFNetProbModel(pyro.nn.PyroModule):
    """Probabilistic MFNet."""

    # def __init__(self, model, noise_std=1.0):
    #     self.model = model
    #     self.sigma = noise_std
    #     self.guide = None # for variational inference
    #     self.mcmc = None # for mcmc
    #     convert_to_pyro(self.model)

    def __init__(self, graph, roots, noise_std=1.0, **kwargs):
        """Initialize Probabilistic MFNET."""
        super().__init__()
        self.model = MFNetTorch(graph, roots, **kwargs)
        self.sigma = noise_std
        self.guide = None  # for variational inference
        self.mcmc = None  # for mcmc
        convert_to_pyro(self.model)

    def update_noise_std(self, noise_std):
        self.sigma = noise_std

    def forward(self, x, targets, y=None):
        """Evaluate model.

        Parameters
        ----------
        x: list of inputs at target nodes
            See MFNetTorch.forward

        targets: list of target nodes
                 See MFNetTorch.forward

        y: list of observations for each of the nodes
        """

        means = self.model(x, targets)
        if y == None:
            for ii, (m, xx) in enumerate(zip(means, x)):
                dout = m.shape[1]
                cov = self.sigma**2 * torch.eye(dout)
                # for jj in pyro.plate(f"data_loop_{ii}", m.shape[0]):
                for kk in range(dout):
                    with pyro.plate(f"data_loop_{ii}_{kk}", m.shape[0]) as jj:
                        obs = pyro.sample(
                            f"obs/{targets[ii]}/{jj}/{kk}",
                            dist.Normal(m[jj, kk], cov[kk, kk]),
                            obs=None,
                        )
        else:
            for ii, (m, xx, yy) in enumerate(zip(means, x, y)):
                dout = yy.shape[1]
                cov = self.sigma**2 * torch.eye(dout)
                for kk in range(dout):
                    yyy = yy[:, kk]
                    mm = m[:, kk]
                    with pyro.plate(f"data_loop_{ii}_{kk}", yy.shape[0]) as jj:
                        obs = pyro.sample(
                            f"obs/{targets[ii]}/{jj}/{kk}",
                            dist.Normal(mm[jj], cov[kk, kk]),
                            obs=yyy[jj],
                        )

        return [m for m in means]

    def train_svi(
        self,
        data,
        targets,
        guide,
        adam_params={"lr": 0.01, "betas": (0.95, 0.999)},
        max_steps=1000,
        print_frac=0.1,
        logger=None,
    ):
        """Train the model.

        Parameters
        ----------
        data: list of data loaders for the data associated with each node
        targets: list of target nodes (same length as data)
        guide: guide to use (this is then stored internally)
        adam_params: parameters to use for optimization
        max_steps: number of optimization steps to take
        print_frac: printing frequency

        Returns
        -------
        self
        """

        optimizer = Adam(adam_params)
        svi = SVI(self, guide, optimizer, loss=Trace_ELBO())
        self.guide = guide
        x = []
        y = []
        for dat in data:
            assert len(dat) == 1
            for batch, (X, Y) in enumerate(dat):
                x.append(X)
                y.append(Y)

        print_increment = int(print_frac * max_steps)
        for step in range(max_steps):
            # self.model.zero_grad()
            elbo = svi.step(x, targets, y)
            if step % print_increment == 0:
                if logger is not None:
                    logger.info(f"Iteration {step}\t Elbo loss: {elbo}")
                else:
                    print(f"Iteration {step}\t Elbo loss: {elbo}")

        return self

    def train_mcmc(
        self, data, targets, num_samples=1000, warmup_frac=0.1, num_chains=1
    ):
        """Train the model.

        Parameters
        ----------
        data: list of data loaders for the data associated with each node
        targets: list of target nodes (same length as data)
        guide: guide to use (this is then stored internally)
        adam_params: parameters to use for optimization
        max_steps: number of optimization steps to take
        print_frac: printing frequency

        Returns
        -------
        self

        Note
        ------
        Stores n mcmc object
        """

        assert (
            num_chains == 1
        ), "Havent implemented merging chains yet in predict, so dont generate data with multiple chains here"
        x = []
        y = []
        for dat in data:
            assert len(dat) == 1
            for batch, (X, Y) in enumerate(dat):
                x.append(X)
                y.append(Y)

        warmup_steps = int(warmup_frac * num_samples)
        nuts_kernel = NUTS(self, jit_compile=False, full_mass=True)
        mcmc = MCMC(
            nuts_kernel,
            num_samples=num_samples,
            warmup_steps=warmup_steps,
            num_chains=num_chains,
        )
        mcmc.run(x, targets, y)
        self.mcmc = mcmc

        return self

    def predict(self, xpred, targets, num_samples):
        """Generate predictions.

        Parameters
        ---------
        xpred: list of locations to predict for models for which predictions are desired
        targets: list of models to predict
        guide: guide to use for predictions
        num_samples: number of samples to predict

        Returns
        -------
        param_samples: samples of the number of parmaeters
        pred_samples: list of size of targets corresponding to predictions
                     pred_samples = num_samples x num_loc_pred x num_outputs

        """
        if self.guide != None:
            predictive = Predictive(
                self, guide=self.guide, num_samples=num_samples
            )
        elif self.mcmc != None:
            predictive = Predictive(self, self.mcmc.get_samples())

        else:
            raise "must run either SVI or MCMC"

        pred = predictive(xpred, targets)

        if self.guide != None:
            param_samples = {k: v for k, v in pred.items() if k[:3] != "obs"}
        elif self.mcmc != None:
            param_samples = self.mcmc.get_samples()

        df_params = param_samples_to_pandas(param_samples)
        pred_samples = [None] * len(targets)
        # print("pred items = ", param_samples)
        for ii in range(len(targets)):
            pred_samples[ii] = [
                v
                for k, v in pred.items()
                if k.startswith(f"obs/{targets[ii]}/")
            ]

            pred_samples[ii] = torch.stack(pred_samples[ii], dim=-1)  #
        return df_params, pred_samples


### Everything below is examples


def run_scale_shift():
    torch.manual_seed(1)
    pyro.clear_param_store()

    # Create the data
    graph, root = make_graph_2()
    model = MFNetTorch(graph, root)

    data = generate_data(model, [20, 20])
    data_loaders = [
        torch.utils.data.DataLoader(d, batch_size=len(d), shuffle=False)
        for d in data
    ]

    # Now train
    num_models = 2
    graph, roots = make_graph_2()
    model_trained = MFNetProbModel(graph, roots, noise_std=1e-3)

    # Plot Prior Predictive
    plt.figure()
    targets = [1, 2]
    x = torch.linspace(-1, 1, 10).reshape(10, 1)
    for ii in range(1000):
        evals = model_trained([x] * num_models, targets)
        plt.plot(x, evals[0].flatten(), color="blue", alpha=0.2)
        plt.plot(x, evals[1].flatten(), color="red", alpha=0.2)

    # TRAIN
    num_samples = 1000
    run_svi = True
    run_mcmc = False

    if run_svi == True:
        num_steps = 10000
        adam_params = {"lr": 0.001, "betas": (0.95, 0.999)}

        guide = AutoNormal(model_trained)
        # guide = AutoMultivariateNormal(model_trained)
        # guide = AutoIAFNormal(model_trained, hidden_dim=[100], num_transforms=4)
        model_trained.train_svi(
            data_loaders, targets, guide, adam_params, max_steps=num_steps
        )

    if run_mcmc == True:
        num_chains = 1
        warmup_frac = 0.2
        model_trained.train_mcmc(
            data_loaders,
            targets,
            num_samples=num_samples,
            warmup_frac=warmup_frac,
        )

    # TEST
    xtest = torch.linspace(-3, 3, 100).reshape(100, 1)

    param_samples, pred_samples = model_trained.predict(
        [xtest] * num_models, targets, num_samples
    )
    # pred_samples = num_samples x num_outputs
    plt.figure()
    for ii in range(num_samples):
        plt.plot(xtest.flatten(), pred_samples[0][ii, :], "blue", alpha=0.2)
        plt.plot(xtest.flatten(), pred_samples[1][ii, :], "red", alpha=0.2)

    plt.plot(data[0].x, data[0].y, "ko")
    plt.plot(data[1].x, data[1].y, "mo")

    print(param_samples)
    # df = samples_to_pandas(param_samples)
    plt.figure()
    plt.plot(param_samples.iloc[:, 0])
    plt.plot(param_samples.iloc[:, 1])

    pd.plotting.scatter_matrix(
        param_samples, alpha=0.2, figsize=(6, 6), diagonal="kde"
    )

    plt.show()


def run_generalized():
    torch.manual_seed(1)

    pyro.clear_param_store()

    # Generate Data
    graph, root = make_graph_2gen()
    model = MFNetTorch(graph, root, edge_type="general")
    x = torch.linspace(-3, 3, 10).reshape(10, 1)
    # plot_funcs(model, x, title="True Model")

    data = generate_data(model, [60, 60])
    data_loaders = [
        torch.utils.data.DataLoader(d, batch_size=len(d), shuffle=False)
        for d in data
    ]

    # graph, roots = make_graph_2()
    # model = MFNetProbModel(graph, roots, noise_std=1e-4)

    # Train
    graph2, root2 = make_graph_2gen()
    model_trained = MFNetProbModel(
        graph2, root2, noise_std=1e-4, edge_type="general"
    )
    targets = [1, 2]
    num_models = len(targets)

    # Plot Prior Predictive
    plt.figure()
    x = torch.linspace(-1, 1, 40).reshape(40, 1)
    for ii in range(1000):
        evals = model_trained([x] * num_models, targets)
        plt.plot(
            x, evals[0].detach().numpy().flatten(), color="blue", alpha=0.2
        )
        plt.plot(
            x, evals[1].detach().numpy().flatten(), color="red", alpha=0.2
        )
    plt.title("Prior predictive")

    # Train
    # guide = AutoNormal(model_trained)
    guide = AutoIAFNormal(model_trained, hidden_dim=[20], num_transforms=2)
    model_trained.train_svi(data_loaders, targets, guide, max_steps=1000)

    # Test
    num_samples = 1000
    xtest = torch.linspace(-3, 3, 100).reshape(100, 1)
    param_samples, pred_samples = model_trained.predict(
        [xtest] * num_models, targets, num_samples
    )
    # pred_samples = num_samples x num_outputs
    plt.figure()
    for ii in range(num_samples):
        plt.plot(xtest.flatten(), pred_samples[0][ii, :], "blue", alpha=0.2)
        plt.plot(xtest.flatten(), pred_samples[1][ii, :], "red", alpha=0.2)

    plt.plot(data[0].x, data[0].y, "ko")
    plt.plot(data[1].x, data[1].y, "mo")

    print(param_samples)
    # df = samples_to_pandas(param_samples)
    plt.figure()
    plt.plot(param_samples.iloc[:, 0])
    plt.plot(param_samples.iloc[:, 1])

    pd.plotting.scatter_matrix(
        param_samples, alpha=0.2, figsize=(6, 6), diagonal="kde"
    )

    plt.show()


def run_generalized_nn():
    torch.manual_seed(1)

    pyro.clear_param_store()

    # Generate Data
    graph, root = make_graph_2gen()
    model = MFNetTorch(graph, root, edge_type="general")

    # plot_funcs(model, x, title="True Model")

    data = generate_data(model, [20, 20])
    data_loaders = [
        torch.utils.data.DataLoader(d, batch_size=len(d), shuffle=False)
        for d in data
    ]

    # graph, roots = make_graph_2()
    # model = MFNetProbModel(graph, roots, noise_std=1e-4)

    # Train
    graph2, root2 = make_graph_2gen_nn()
    model_trained = MFNetProbModel(
        graph2, root2, noise_std=1e-2, edge_type="general"
    )
    targets = [1, 2]
    num_models = len(targets)

    # Plot Prior Predictive
    plt.figure()
    x = torch.linspace(-3, 3, 40).reshape(40, 1)
    # x = torch.linspace(-3, 3, 10).reshape(10, 1)
    for ii in range(1000):
        evals = model_trained([x] * num_models, targets)
        plt.plot(
            x, evals[0].detach().numpy().flatten(), color="blue", alpha=0.2
        )
        plt.plot(
            x, evals[1].detach().numpy().flatten(), color="red", alpha=0.2
        )
    plt.title("Prior predictive")

    # Train
    guide = AutoNormal(model_trained)
    # guide = AutoIAFNormal(model_trained, hidden_dim=[20], num_transforms=2)
    model_trained.train_svi(data_loaders, targets, guide, max_steps=10000)

    # Test
    num_samples = 1000
    xtest = torch.linspace(-3, 3, 100).reshape(100, 1)
    param_samples, pred_samples = model_trained.predict(
        [xtest] * num_models, targets, num_samples
    )
    # pred_samples = num_samples x num_outputs
    plt.figure()
    for ii in range(num_samples):
        plt.plot(xtest.flatten(), pred_samples[0][ii, :], "blue", alpha=0.2)
        plt.plot(xtest.flatten(), pred_samples[1][ii, :], "red", alpha=0.2)

    plt.plot(data[0].x, data[0].y, "ko")
    plt.plot(data[1].x, data[1].y, "mo")

    # print(param_samples.summary())
    # df = samples_to_pandas(param_samples)
    # plt.figure()
    # plt.plot(param_samples.iloc[:, 0])
    # plt.plot(param_samples.iloc[:, 1])

    # pd.plotting.scatter_matrix(param_samples, alpha=0.2, figsize=(6,6), diagonal='kde')

    plt.show()


def run_single_fidelity():
    torch.manual_seed(1)

    pyro.clear_param_store()

    # Generate Data
    graph, root = make_graph_single()  # train with non nn

    model = MFNetTorch(graph, root, edge_type="general")
    # plot_funcs(model, x, title="True Model")
    data = generate_data_1model(model, 20)
    data_loaders = [
        torch.utils.data.DataLoader(d, batch_size=len(d), shuffle=False)
        for d in data
    ]

    # graph, roots = make_graph_2()
    # model = MFNetProbModel(graph, roots, noise_std=1e-4)

    # Train
    graph2, root2 = make_graph_single()
    model_trained = MFNetProbModel(
        graph2, root2, noise_std=1e-4, edge_type="general"
    )
    targets = [1]
    num_models = len(targets)

    # Plot Prior Predictive
    plt.figure()
    x = torch.linspace(-3, 3, 40).reshape(40, 1)
    for ii in range(1000):
        evals = model_trained([x] * num_models, targets)
        plt.plot(
            x, evals[0].detach().numpy().flatten(), color="blue", alpha=0.2
        )
    plt.title("Prior predictive")

    # Train
    guide = AutoNormal(model_trained)
    # guide = AutoIAFNormal(model_trained, hidden_dim=[20], num_transforms=2)
    model_trained.train_svi(data_loaders, targets, guide, max_steps=14000)

    # Test
    num_samples = 1000
    xtest = torch.linspace(-3, 3, 100).reshape(100, 1)
    param_samples, pred_samples = model_trained.predict(
        [xtest] * num_models, targets, num_samples
    )
    # pred_samples = num_samples x num_outputs
    plt.figure()
    for ii in range(num_samples):
        plt.plot(xtest.flatten(), pred_samples[0][ii, :], "blue", alpha=0.2)

    plt.plot(data[0].x, data[0].y, "ko")

    # print(param_samples.summary())
    # df = samples_to_pandas(param_samples)
    # plt.figure()
    # plt.plot(param_samples.iloc[:, 0])
    # plt.plot(param_samples.iloc[:, 1])

    # pd.plotting.scatter_matrix(param_samples, alpha=0.2, figsize=(6,6), diagonal='kde')

    plt.show()


def run_generalized_nn_fixed_edge():
    torch.manual_seed(1)
    graph, root = make_graph_2gen_nn_fixed()  # train with non nn
    model = MFNetTorch(graph, root, edge_type="general")

    data = generate_data(model, [20, 20])
    data_loaders = [
        torch.utils.data.DataLoader(d, batch_size=len(d), shuffle=False)
        for d in data
    ]

    # graph, roots = make_graph_2()
    # model = MFNetProbModel(graph, roots, noise_std=1e-4)

    # Train
    graph2, root2 = make_graph_2gen_nn_fixed()
    model_trained = MFNetProbModel(
        graph2, root2, noise_std=1e-4, edge_type="general"
    )
    targets = [1, 2]
    num_models = len(targets)

    # Plot Prior Predictive
    plt.figure()
    x = torch.linspace(-3, 3, 40).reshape(40, 1)
    for ii in range(1000):
        evals = model_trained([x] * num_models, targets)
        plt.plot(
            x, evals[0].detach().numpy().flatten(), color="blue", alpha=0.2
        )
        plt.plot(
            x, evals[1].detach().numpy().flatten(), color="red", alpha=0.2
        )
    plt.title("Prior predictive")

    # Train
    guide = AutoNormal(model_trained)
    # guide = AutoDelta(model_trained)
    # guide = AutoIAFNormal(model_trained, hidden_dim=[20], num_transforms=2)
    model_trained.train_svi(data_loaders, targets, guide, max_steps=10000)

    # Test
    num_samples = 1000
    xtest = torch.linspace(-3, 3, 100).reshape(100, 1)
    param_samples, pred_samples = model_trained.predict(
        [xtest] * num_models, targets, num_samples
    )
    # pred_samples = num_samples x num_outputs
    plt.figure()
    for ii in range(num_samples):
        plt.plot(xtest.flatten(), pred_samples[0][ii, :], "blue", alpha=0.2)
        plt.plot(xtest.flatten(), pred_samples[1][ii, :], "red", alpha=0.2)

    plt.plot(data[0].x, data[0].y, "ko")
    plt.plot(data[1].x, data[1].y, "mo")

    # print(param_samples.summary())
    # df = samples_to_pandas(param_samples)
    # plt.figure()
    # plt.plot(param_samples.iloc[:, 0])
    # plt.plot(param_samples.iloc[:, 1])

    # pd.plotting.scatter_matrix(param_samples, alpha=0.2, figsize=(6,6), diagonal='kde')

    plt.show()


def run_8model():
    graph, roots = make_graph_8()

    node = 8

    ## Truth
    mfsurr_true = MFNetTorch(graph, roots)

    dx = 1
    ndata = [0] * 8
    ndata[7] = 10
    x = torch.rand(ndata[7], 1)
    # y =  mfsurr_true.forward([x],[8])[0].detach()
    y = x**2

    plt.figure()
    plt.plot(x, y, "o", color="blue", alpha=0.2)
    dataset = ArrayDataset(x, y)
    data_loaders = [
        torch.utils.data.DataLoader(
            dataset, batch_size=ndata[7], shuffle=False
        )
    ]

    graph_learn, roots_learn = make_graph_8()
    # graph_learn, roots_learn = make_graph_single() # train with non nn
    model_trained = MFNetProbModel(graph_learn, roots_learn, noise_std=1e-4)
    # guide = AutoIAFNormal(model_trained, hidden_dim=[100], num_transforms=2)
    guide = AutoNormal(model_trained)
    # guide = AutoDelta(model_trained)
    # targets = [1, 2, 3, 4, 5, 6, 7, 8]
    targets = [8]
    # targets = [1]
    adam_params = {"lr": 0.1, "betas": (0.9, 0.999)}
    model_trained.train_svi(
        data_loaders, targets, guide, adam_params, max_steps=2000
    )

    # print(mfsurr_learn)
    # mfsurr_learn.train(data_loaders, [8], loss_fns[7:])

    # print("\n")
    # with torch.no_grad():
    # predict = mfsurr_learn([x],[8])[0]
    num_samples = 1000
    pred_samples = model_trained.predict([x], targets, num_samples)[1]
    print("pred_samples shape = ", pred_samples[0].shape)
    mean_predict = torch.mean(pred_samples[0], dim=0)
    plt.figure()
    for ii in range(num_samples):
        plt.plot(x, pred_samples[0][ii, :, :], "o", color="red", alpha=0.2)
    plt.plot(x, y, "o", color="blue")
    plt.plot(x, mean_predict, "ko")

    # print(predict.size())
    err = torch.linalg.norm(mean_predict - y) ** 2 / torch.linalg.norm(y) ** 2
    print("err = ", err)
    plt.show()
    exit(1)


def make_graph_4():
    """A graph with 4 nodes with different output dims

    1- > 4 <- 2 <- 3
    """

    graph = nx.DiGraph()

    dim_in = 1
    dim_out = [2, 3, 6, 4]

    graph.add_node(
        1,
        func=torch.nn.Linear(dim_in, dim_out[0], bias=True),
        dim_in=dim_in,
        dim_out=dim_out[0],
    )
    graph.add_node(
        2,
        func=torch.nn.Linear(dim_in, dim_out[1], bias=True),
        dim_in=dim_in,
        dim_out=dim_out[1],
    )
    graph.add_node(
        3,
        func=torch.nn.Linear(dim_in, dim_out[2], bias=True),
        dim_in=dim_in,
        dim_out=dim_out[2],
    )
    graph.add_node(
        4,
        func=torch.nn.Linear(dim_in, dim_out[3], bias=True),
        dim_in=dim_in,
        dim_out=dim_out[3],
    )

    graph.add_edge(
        1,
        4,
        func=torch.nn.Linear(dim_in, dim_out[0] * dim_out[3], bias=True),
        out_rows=dim_out[3],
        out_cols=dim_out[0],
        dim_in=1,
    )
    graph.add_edge(
        2,
        4,
        func=torch.nn.Linear(dim_in, dim_out[1] * dim_out[3], bias=True),
        out_rows=dim_out[3],
        out_cols=dim_out[1],
        dim_in=1,
    )
    graph.add_edge(
        3,
        2,
        func=torch.nn.Linear(dim_in, dim_out[2] * dim_out[1], bias=True),
        out_rows=dim_out[1],
        out_cols=dim_out[2],
        dim_in=1,
    )

    roots = set([1, 3])
    return graph, roots, dim_out


def run_multi_output_gen():
    graph, roots, dim_out = make_graph_4()
    node = 4

    ## Truth
    mfsurr_true = MFNetTorch(graph, roots)

    dx = 1
    ndata = [0] * 4
    ndata[3] = 10
    x = torch.rand(ndata[3], 1)
    # y =  mfsurr_true.forward([x]*4,[1, 2, 3, 4])
    y = mfsurr_true.forward([x], [4])[0].detach()
    # print("\n")
    # print("yshapes = ", [yy.size() for yy in y])
    # print("yshape = ", y.shape)
    # exit(1)

    dataset = ArrayDataset(x, y)
    data_loaders = [
        torch.utils.data.DataLoader(
            dataset, batch_size=ndata[3], shuffle=False
        )
    ]

    plt.figure()
    for ii in range(dim_out[3]):
        plt.plot(x, y[:, ii], "o")

    graph_learn, roots_learn, dim_out = make_graph_4()
    model_trained = MFNetProbModel(graph_learn, roots_learn, noise_std=1e-3)
    # guide = AutoIAFNormal(model_trained, hidden_dim=[20], num_transforms=2)
    # guide = AutoNormal(model_trained)
    guide = AutoMultivariateNormal(model_trained)
    targets = [4]
    adam_params = {"lr": 0.1, "betas": (0.9, 0.999)}
    model_trained.train_svi(
        data_loaders, targets, guide, adam_params, max_steps=1000
    )

    # print("\n")
    with torch.no_grad():
        num_samples = 100
        pred_samples = model_trained.predict([x], targets, num_samples)[1]
        predict = torch.mean(pred_samples[0], dim=0)

        plt.figure()
        for ii in range(dim_out[3]):
            plt.plot(x, y[:, ii], "r")
            plt.plot(x, predict[:, ii], "k")
        plt.show()
        assert predict.size(dim=1) == dim_out[node - 1]

        err = torch.linalg.norm(predict - y) ** 2 / (ndata[3] * dim_out[3])
        print("err = ", err)
        # assert err<1e-3


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns
    import networkx as nx

    from pyro.infer import MCMC, NUTS

    from pyro.infer.autoguide import (
        AutoDelta,
        AutoNormal,
        AutoMultivariateNormal,
        AutoLowRankMultivariateNormal,
        AutoGuideList,
        AutoIAFNormal,
        init_to_feasible,
    )

    from mfnets_surrogates.net_torch import (
        make_graph_2,
        make_graph_8,
        make_graph_2gen,
        make_graph_2gen_nn,
        make_graph_2gen_nn_fixed,
        make_graph_single,
        ArrayDataset,
        generate_data,
        generate_data_1model,
    )

    # run_scale_shift()
    run_generalized()
    # run_generalized_nn()
    # run_single_fidelity()
    # run_generalized_nn_fixed_edge()
    # run_8model()
    # run_multi_output_gen()
