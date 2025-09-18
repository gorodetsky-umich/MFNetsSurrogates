import unittest

# import numpy as np
# import torch
# import networkx as nx

# from mfnets_surrogates.net_torch import *

from .test_utils import *
from mfnets_surrogates.net_pyro import *

from pyro.infer.autoguide import (
    AutoDelta,
    AutoNormal,
    AutoMultivariateNormal,
    # AutoLowRankMultivariateNormal,
    # AutoGuideList,
    AutoIAFNormal,
    # init_to_feasible,
)

class TestMfnetPyro(unittest.TestCase):

    # @unittest.skip('testing other')
    def test_vi_mean_pred(self):
        torch.manual_seed(2)

        graph, roots = make_graph_8()

        node = 8

        ## Truth
        mfsurr_true = MFNetTorch(graph, roots)

        dx = 1
        ndata = [0] * 8
        ndata[7] = 20
        x = torch.rand(ndata[7], 1)
        y =  mfsurr_true.forward([x],[8])[0].detach()
        # y =  x**2

        # import matplotlib.pyplot as plt
        
        # plt.figure()
        # plt.plot(x, y, 'o', color='blue', alpha=0.2)


        # print("y = ", y)

        dataset = ArrayDataset(x, y)
        data_loaders = [torch.utils.data.DataLoader(dataset,
                                                    batch_size=ndata[7],
                                                    shuffle=False)]
        
        
        graph_learn, roots_learn = make_graph_8()
        model_trained = MFNetProbModel(graph_learn, roots_learn, noise_std=1e-3)
        # guide = AutoIAFNormal(model_trained, hidden_dim=[100], num_transforms=2)    
        guide = AutoNormal(model_trained)
        # guide = AutoDelta(model_trained)
        targets = [8]
        adam_params = {"lr": 0.1, "betas": (0.9, 0.999)}
        model_trained.train_svi(data_loaders, targets, guide, adam_params, max_steps=1000)

        # print(mfsurr_learn)
        # mfsurr_learn.train(data_loaders, [8], loss_fns[7:])

        # print("\n")
        # with torch.no_grad():
            # predict = mfsurr_learn([x],[8])[0]
        num_samples = 1000
        pred_samples = model_trained.predict([x],targets, num_samples)[1]
        # print("pred_samples shape = ", pred_samples[0].shape)
        mean_predict = torch.mean(pred_samples[0], dim=0)
        # plt.figure()
        # plt.plot(x.flatten(), pred_samples[0].T, 'o', color='red', alpha=0.2)
        # plt.plot(x, y, 'o', color='blue')
        # plt.plot(x, mean_predict, 'ko')

        # print("y size = ", y.shape)
        # print("mean_predict.shape = ", mean_predict.shape)
        
        # print("error = ", mean_predict.flatten() - y.flatten())
        err = torch.linalg.norm(mean_predict-y)**2 / ndata[7]  # / torch.linalg.norm(y)**2
        # print("err = ", err)
        # plt.show()
        # exit(1)

        assert err<1e-3, f"training error is {err}"

        ntest = 100
        x_test = torch.rand(ntest, dx)
        with torch.no_grad():
            y_test =  mfsurr_true.forward([x_test],[8])[0]
            
            predict_test = torch.mean(model_trained.predict([x_test], [8], num_samples)[1][0], dim=0)
            err = torch.linalg.norm(predict_test-y_test)**2 / ntest
            # print("err = ", err)
            # plt.figure()
            # plt.plot(x_test, y_test, 'o', color='blue')
            # plt.plot(x_test, predict_test, 'ko')
            # plt.show()
            assert err<1e-3, f"testing error is {err}"

    # @unittest.skip('testing other')
    def test_vi_multi_out(self):
        torch.manual_seed(2)

        # print("\n")
        graph, roots, dim_out = make_graph_4()

        node = 4

        ## Truth
        mfsurr_true = MFNetTorch(graph, roots)

        dx = 1
        ndata = [0] * 4
        ndata[3] = 500
        x = torch.rand(ndata[3], 1)
        # y =  mfsurr_true.forward([x]*4,[1, 2, 3, 4])
        y =  mfsurr_true.forward([x], [4])[0].detach()
        # print("\n")
        # print("yshapes = ", [yy.size() for yy in y])
        # print("y = ", y)
        # exit(1)
        
        dataset = ArrayDataset(x, y)
        data_loaders = [torch.utils.data.DataLoader(dataset,
                                                    batch_size=ndata[3],
                                                    shuffle=False)]
        

        graph_learn, roots_learn, dim_out = make_graph_4()
        model_trained = MFNetProbModel(graph_learn, roots_learn, noise_std=1e-2)
        # guide = AutoIAFNormal(model_trained, hidden_dim=[100], num_transforms=2)    
        guide = AutoNormal(model_trained)
        # guide = AutoDelta(model_trained)
        targets = [4]
        adam_params = {"lr": 0.1, "betas": (0.9, 0.999)}
        model_trained.train_svi(data_loaders, targets, guide, adam_params, max_steps=1000)

        # print("\n")
        with torch.no_grad():

            num_samples = 10
            pred_samples = model_trained.predict([x], targets, num_samples)[1]
            predict = torch.mean(pred_samples[0], dim=0)
            
            assert predict.size(dim=1) == dim_out[node-1]

            err = torch.linalg.norm(predict-y)**2/ (dim_out[node-1] * ndata[-1])
            print("err = ", err)
            assert err<1e-3, f"error = {err}"

        ntest = 100
        x_test = torch.rand(ntest, dx)
        with torch.no_grad():
            y_test =  mfsurr_true.forward([x_test],[node])[0].detach()
            predict_samples = model_trained.predict([x_test], [node], num_samples)[1]
            predict = torch.mean(predict_samples[0], dim=0)
            err = torch.linalg.norm(predict - y_test)**2/(ntest * dim_out[node-1])
            print("err = ", err)
            assert err<1e-3, f"Testing error = {err}"

    # @unittest.skip('testing other')
    def test_vi_multi_out_gen(self):
        torch.manual_seed(2)

        # print("\n")
        graph, roots, dim_out = make_graph_4()

        node = 4

        ## Truth
        mfsurr_true = MFNetTorch(graph, roots)

        dx = 1
        ndata = [0] * 4
        ndata[3] = 500
        x = torch.rand(ndata[3], 1)
        # y =  mfsurr_true.forward([x]*4,[1, 2, 3, 4])
        y =  mfsurr_true.forward([x], [4])[0].detach()

        dataset = ArrayDataset(x, y)
        data_loaders = [torch.utils.data.DataLoader(dataset,
                                                    batch_size=ndata[3],
                                                    shuffle=False)]
        # learning
        graph_learn, roots_learn, _ = make_graph_4gen()
        model_trained = MFNetProbModel(graph_learn, roots_learn, edge_type="general", noise_std=1e-2)
        guide = AutoNormal(model_trained)
        targets = [4]
        adam_params = {"lr": 0.1, "betas": (0.9, 0.999)}
        model_trained.train_svi(data_loaders, targets, guide, adam_params, max_steps=1000)
        
        print("\n")
        with torch.no_grad():
            num_samples = 10
            pred_samples = model_trained.predict([x], targets, num_samples)[1]
            predict = torch.mean(pred_samples[0], dim=0)
            
            assert predict.size(dim=1) == dim_out[node-1]
            err = torch.linalg.norm(predict-y)**2 / (ndata[3] * dim_out[3])
            print("err = ", err)
            assert err<1e-3, f"Train error = {err}"

        ntest=1000
        x_test = torch.rand(ntest, dx)
        with torch.no_grad():
            y_test =  mfsurr_true.forward([x_test],[node])[0]
            predict_samples = model_trained.predict([x_test], [node], num_samples)[1]
            predict = torch.mean(predict_samples[0], dim=0)
            err = torch.linalg.norm(predict - y_test)**2/(ntest * dim_out[node-1])
            print("err = ", err)
            assert err<1e-3, f"Test error = {err}"

    # @unittest.skip('testing other')
    def test_vi_multi_out_gen_nn(self):
        torch.manual_seed(2)

        # print("\n")
        # graph, roots, dim_out = make_graph_4gen_nn()
        graph, roots, dim_out = make_graph_4()

        node = 4

        ## Truth
        # mfsurr_true = MFNetTorch(graph, roots, edge_type="general")
        mfsurr_true = MFNetTorch(graph, roots)        

        dx = 1
        ndata = [0] * 4
        ndata[3] = 5
        x = torch.rand(ndata[3], 1)
        # y =  mfsurr_true.forward([x]*4,[1, 2, 3, 4])
        y =  mfsurr_true.forward([x], [4])[0].detach()

        dataset = ArrayDataset(x, y)
        data_loaders = [torch.utils.data.DataLoader(dataset,
                                                    batch_size=ndata[3],
                                                    shuffle=False)]
        
        # learning
        graph_learn, roots_learn, _ = make_graph_4gen_nn()
        model_trained = MFNetProbModel(graph_learn, roots_learn, edge_type="general", noise_std=1e-0)
        # guide = AutoIAFNormal(model_trained)    
        guide = AutoNormal(model_trained)
        targets = [4]
        adam_params = {"lr": 0.1, "betas": (0.95, 0.999)}
        print("train")
        model_trained.train_svi(data_loaders, targets, guide, adam_params, max_steps=1000)
        
        print("\n")
        with torch.no_grad():
            num_samples = 100
            pred_samples = model_trained.predict([x], targets, num_samples)[1]
            predict = torch.mean(pred_samples[0], dim=0)

            assert predict.size(dim=1) == dim_out[node-1]
            err = torch.linalg.norm(predict-y)**2 / (ndata[3] * dim_out[3])
            print("err = ", err)
            assert err < 1e-3, f"Training error = {err}"

        ntest=1000
        x_test = torch.rand(ntest, dx)
        with torch.no_grad():
            y_test =  mfsurr_true.forward([x_test],[node])[0]
            predict_samples = model_trained.predict([x_test], [node], num_samples)[1]
            predict = torch.mean(predict_samples[0], dim=0)
            err = torch.linalg.norm(predict - y_test)**2/(ntest * dim_out[node-1])            
            print("err = ", err)
            assert err<1e-3, f"Testing error = {err}"


    # @unittest.skip('testing other')
    def test_vi_multi_out_gen_nn_model_average(self):
        torch.manual_seed(2)

        # print("\n")
        # graph, roots, dim_out = make_graph_4gen_nn()
        graph, roots, dim_out = make_graph_4gen_nn_equal_model_average()

        node = 4

        ## Truth
        # mfsurr_true = MFNetTorch(graph, roots, edge_type="general")
        mfsurr_true = MFNetTorch(graph, roots, edge_type="general")

        dx = 1
        ndata = [0] * 4
        ndata[3] = 500
        x = torch.rand(ndata[3], 1)
        y =  mfsurr_true.forward([x], [4])[0].detach()

        dataset = ArrayDataset(x, y)
        data_loaders = [torch.utils.data.DataLoader(dataset,
                                                    batch_size=ndata[3],
                                                    shuffle=False)]        

        # learning
        graph_learn, roots_learn, _ = make_graph_4gen_nn_equal_model_average()
        model_trained = MFNetProbModel(graph_learn, roots_learn, edge_type="general", noise_std=1e0)
        guide = AutoNormal(model_trained)
        targets = [4]
        adam_params = {"lr": 0.001, "betas": (0.5, 0.999)}
        model_trained.train_svi(data_loaders, targets, guide, adam_params, max_steps=5000)        


        print("\n")
        with torch.no_grad():
            num_samples = 100
            pred_samples = model_trained.predict([x], targets, num_samples)[1]
            predict = torch.mean(pred_samples[0], dim=0)

            assert predict.size(dim=1) == dim_out[node-1]
            err = torch.linalg.norm(predict-y)**2/2
            print("err = ", err)
            assert err<1e-4, f"Training error = {err}"

        ntest=1000
        x_test = torch.rand(ntest, dx)
        with torch.no_grad():
            y_test =  mfsurr_true.forward([x_test],[node])[0]
            predict_samples = model_trained.predict([x_test], [node], num_samples)[1]
            predict = torch.mean(predict_samples[0], dim=0)
            err = torch.linalg.norm(predict - y_test)**2/(ntest * dim_out[node-1])
            print("err = ", err)
            assert err<1e-3, f"Testing error = {err}"
            
if __name__== "__main__":    
    mfnet_pyro_test_suite = unittest.TestLoader().loadTestsFromTestCase(TestMfnetPyro)
    unittest.TextTestRunner(verbosity=2).run(mfnet_pyro_test_suite)
