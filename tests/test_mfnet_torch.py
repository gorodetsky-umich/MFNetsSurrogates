import unittest
# import numpy as np
# import torch
# import networkx as nx

# from mfnets_surrogates.net_torch import *
from .test_utils import *

class TestMfnet(unittest.TestCase):

    # @unittest.skip('testing other')
    def test_least_squares_opt(self):
        torch.manual_seed(2)

        graph, roots = make_graph_8()

        node = 8

        ## Truth
        mfsurr_true = MFNetTorch(graph, roots)

        dx = 1
        ndata = [0] * 8
        ndata[7] = 500
        x = torch.rand(ndata[7], 1)
        y =  mfsurr_true.forward([x],[8])[0]
        std = 1.#1e-4

        graph_learn, roots_learn = make_graph_8()        
        mfsurr_learn = MFNetTorch(graph_learn, roots_learn)

        loss_fns = construct_loss_funcs(mfsurr_learn)    
        dataset = ArrayDataset(x, y)
        data_loaders = [torch.utils.data.DataLoader(dataset,
                                                    batch_size=ndata[7],
                                                    shuffle=False)]
        # print(mfsurr_learn)
        mfsurr_learn.train(data_loaders, [8], loss_fns[7:])

        print("\n")
        with torch.no_grad():
            predict = mfsurr_learn([x],[8])[0]
            # print(predict.size())
            err = torch.linalg.norm(predict-y)**2/2
            print("err = ", err)
            assert err<1e-4

        ntest=1000
        x_test = torch.rand(ntest, dx)
        with torch.no_grad():
            y_test =  mfsurr_true.forward([x_test],[8])[0]
            predict_test = mfsurr_learn.forward([x_test], [8])[0]
            err = torch.linalg.norm(predict_test-y_test)/np.sqrt(ntest)
            print("err = ", err)
            assert err<1e-3

    # @unittest.skip('testing other')
    def test_least_squares_opt_multi_out(self):
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
        y =  mfsurr_true.forward([x], [4])[0]
        # print("\n")
        # print("yshapes = ", [yy.size() for yy in y])
        # print("y = ", y)
        # exit(1)
        std = 1.#1e-4
        
        graph_learn, roots_learn, _ = make_graph_4()        
        mfsurr_learn = MFNetTorch(graph_learn, roots_learn)

        loss_fns = construct_loss_funcs(mfsurr_learn)    
        dataset = ArrayDataset(x, y)
        data_loaders = [torch.utils.data.DataLoader(dataset,
                                                    batch_size=ndata[3],
                                                    shuffle=False)]
        # print(mfsurr_learn)
        mfsurr_learn.train(data_loaders, [node], loss_fns[(node-1):])

        print("\n")
        with torch.no_grad():
            predict = mfsurr_learn([x],[node])[0]
            # print(dim_out)
            # print(predict.size())
            assert predict.size(dim=1) == dim_out[node-1]

            err = torch.linalg.norm(predict-y)**2/2
            print("err = ", err)
            assert err<1e-4

        ntest=1000
        x_test = torch.rand(ntest, dx)
        with torch.no_grad():
            y_test =  mfsurr_true.forward([x_test],[node])[0]
            predict_test = mfsurr_learn.forward([x_test], [node])[0]
            err = torch.linalg.norm(predict_test-y_test)/np.sqrt(ntest)
            print("err = ", err)
            assert err<1e-3

    # @unittest.skip('testing other')
    def test_least_squares_opt_multi_out_gen(self):
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
        y =  mfsurr_true.forward([x], [4])[0]
        # print("\n")
        # print("yshapes = ", [yy.size() for yy in y])
        # print("y = ", y)
        # exit(1)
        std = 1.#1e-4

        # learning
        graph_learn, roots_learn, _ = make_graph_4gen()        
        mfsurr_learn = MFNetTorch(graph_learn, roots_learn, edge_type="general")

        loss_fns = construct_loss_funcs(mfsurr_learn)    
        dataset = ArrayDataset(x, y)
        data_loaders = [torch.utils.data.DataLoader(dataset,
                                                    batch_size=ndata[3],
                                                    shuffle=False)]
        # print(mfsurr_learn)
        mfsurr_learn.train(data_loaders, [node], loss_fns[(node-1):])

        print("\n")
        with torch.no_grad():
            predict = mfsurr_learn([x],[node])[0]
            # print(dim_out)
            # print(predict.size())
            assert predict.size(dim=1) == dim_out[node-1]

            err = torch.linalg.norm(predict-y)**2/2
            print("err = ", err)
            assert err<1e-4

        ntest=1000
        x_test = torch.rand(ntest, dx)
        with torch.no_grad():
            y_test =  mfsurr_true.forward([x_test],[node])[0]
            predict_test = mfsurr_learn.forward([x_test], [node])[0]
            err = torch.linalg.norm(predict_test-y_test)/np.sqrt(ntest)
            print("err = ", err)
            assert err<1e-3

    @unittest.skip('testing other')
    def test_least_squares_opt_multi_out_gen_nn(self):
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
        ndata[3] = 500
        x = torch.rand(ndata[3], 1)
        # y =  mfsurr_true.forward([x]*4,[1, 2, 3, 4])
        y =  mfsurr_true.forward([x], [4])[0]
        # print("\n")
        # print("yshapes = ", [yy.size() for yy in y])
        # print("y = ", y)
        # exit(1)
        std = 1e-4

        # learning
        graph_learn, roots_learn, _ = make_graph_4gen_nn()        
        mfsurr_learn = MFNetTorch(graph_learn, roots_learn, edge_type="general")

        loss_fns = construct_loss_funcs(mfsurr_learn)    
        dataset = ArrayDataset(x, y)
        data_loaders = [torch.utils.data.DataLoader(dataset,
                                                    batch_size=ndata[3],
                                                    shuffle=False)]
        # print(mfsurr_learn)
        mfsurr_learn.train(data_loaders, [node], loss_fns[(node-1):],
                           max_iter=1000)

        print("\n")
        with torch.no_grad():
            predict = mfsurr_learn([x],[node])[0]
            # print(dim_out)
            # print(predict.size())
            assert predict.size(dim=1) == dim_out[node-1]

            err = torch.linalg.norm(predict-y)**2/2
            print("err = ", err)
            assert err<1e-4

        ntest=1000
        x_test = torch.rand(ntest, dx)
        with torch.no_grad():
            y_test =  mfsurr_true.forward([x_test],[node])[0]
            predict_test = mfsurr_learn.forward([x_test], [node])[0]
            err = torch.linalg.norm(predict_test-y_test)/np.sqrt(ntest)
            print("err = ", err)
            assert err<1e-3

    # @unittest.skip('testing other')
    def test_least_squares_opt_multi_out_gen_nn_model_average(self):
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
        # y =  mfsurr_true.forward([x]*4,[1, 2, 3, 4])
        y =  mfsurr_true.forward([x], [4])[0]
        # print("\n")
        # print("yshapes = ", [yy.size() for yy in y])
        # print("y = ", y)
        # exit(1)
        std = 1.#1e-4

        # learning
        graph_learn, roots_learn, _ = make_graph_4gen_nn_equal_model_average()
        mfsurr_learn = MFNetTorch(graph_learn, roots_learn, edge_type="general")

        loss_fns = construct_loss_funcs(mfsurr_learn)    
        dataset = ArrayDataset(x, y)
        data_loaders = [torch.utils.data.DataLoader(dataset,
                                                    batch_size=ndata[3],
                                                    shuffle=False)]
        # print(mfsurr_learn)
        mfsurr_learn.train(data_loaders, [node], loss_fns[(node-1):],
                           max_iter=5000)

        print("\n")
        with torch.no_grad():
            predict = mfsurr_learn([x],[node])[0]
            # print(dim_out)
            # print(predict.size())
            assert predict.size(dim=1) == dim_out[node-1]

            err = torch.linalg.norm(predict-y)**2/2
            print("err = ", err)
            assert err<1e-4

        ntest=1000
        x_test = torch.rand(ntest, dx)
        with torch.no_grad():
            y_test =  mfsurr_true.forward([x_test],[node])[0]
            predict_test = mfsurr_learn.forward([x_test], [node])[0]
            err = torch.linalg.norm(predict_test-y_test)/np.sqrt(ntest)
            print("err = ", err)
            assert err<1e-3            
        
if __name__== "__main__":    
    mfnet_test_suite = unittest.TestLoader().loadTestsFromTestCase(TestMfnet)
    unittest.TextTestRunner(verbosity=2).run(mfnet_test_suite)


