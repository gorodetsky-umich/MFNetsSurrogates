import unittest
import networkx as nx

from mfnets_surrogates.net import *

def make_graph_2():
    """Make a graph with two nodes

    1 -> 2
    """

    graph = nx.DiGraph()

    pnodes = np.random.randn(2, 2)
    pedges = np.random.randn(1, 2)

    graph.add_node(1, param=pnodes[0, :], func=lin)
    graph.add_node(2, param=pnodes[1, :], func=lin)
    graph.add_edge(1, 2, param=pedges[0, :], func=lin)
    return graph, set([1])

def make_graph_8(nnode_param=2, nedge_param=2, linfunc=lin):
    """A graph with 8 nodes

    3 -> 7 -> 8
              ^
              |
         1 -> 4
            / ^
           /  |
    2 -> 5 -> 6
    """

    graph = nx.DiGraph()

    pnodes = np.random.randn(10, nnode_param)
    pedges = np.random.randn(8, nedge_param)

    for node in range(1, 9):
        graph.add_node(node, param=pnodes[node-1], func=linfunc)

    graph.add_edge(1, 4, param=pedges[0, :], func=linfunc)
    graph.add_edge(2, 5, param=pedges[1, :], func=linfunc)
    graph.add_edge(5, 6, param=pedges[2, :], func=linfunc)
    graph.add_edge(6, 4, param=pedges[3, :], func=linfunc)
    graph.add_edge(3, 7, param=pedges[4, :], func=linfunc)
    graph.add_edge(7, 8, param=pedges[5, :], func=linfunc)
    graph.add_edge(4, 8, param=pedges[6, :], func=linfunc)
    graph.add_edge(5, 4, param=pedges[7, :], func=linfunc)

    roots = set([1, 2, 3])
    return graph, roots

try:
    from pyapprox.l1_minimization import lasso, nonlinear_basis_pursuit
    pyapprox_available=True
except ImportError:
    pyapprox_available=False

class TestMfnet(unittest.TestCase):

    def test_gradient(self):
        dx = 1
        np.random.seed(2)
        graph, roots = make_graph_8()
        node = 8

        mfsurr = MFSurrogate(graph, roots)
        nparam = mfsurr.get_nparam()

        x = np.random.rand(5, dx)
        y = np.random.rand(5)
        std = 1e-2

        x0 = np.random.randn(nparam)
        # agrad = learn_obj_grad(x0, mfsurr, node, x, y, std)
        fgrad = sciopt.approx_fprime(
            x0, learn_obj, 1e-8, mfsurr, node, x, y, std)

        err = sciopt.check_grad(
            learn_obj, learn_obj_grad, x0, mfsurr, node, x, y, std)
        rel_err = err / np.linalg.norm(fgrad)

        assert rel_err <1e-7, f"Relative error of finite difference gradient is {rel_err}"

    def test_jac(self):
        """Test building block for jacobian calculation """
        dx = 1
        np.random.seed(2)
        graph, roots = make_graph_8()
        
        mfsurr = MFSurrogate(graph, roots)
        nparam = mfsurr.get_nparam()

        npts = 1
        x = np.random.rand(npts, dx)
        y = np.random.rand(npts)
        std = 1e-2

        #print("nparam = ", nparam)
        x0 = np.random.randn(nparam)

        #return the row of the jacobian corresponding to node 
        for node in range(1,9):
            fgrad = sciopt.approx_fprime(
                x0, identity_obj, 1e-10, mfsurr, node, x)
            err = sciopt.check_grad(
                identity_obj, identity_obj_grad, x0, mfsurr, node, x)
            rel_err = err / np.linalg.norm(fgrad)
            #print("err = ", err, rel_err)
            assert np.all(err<1e-7)

    def test_least_squares_opt(self):
        np.random.seed(2)

        nnode_param, nedge_param = 2, 1
        graph, roots = make_graph_8(
            nnode_param,nedge_param,linfunc=monomial_1d_lin)
        node = 8

        ## Truth
        mfsurr_true = MFSurrogate(graph, roots)
        nparam = mfsurr_true.get_nparam()
        sparsity = nparam // 4
        true_param = np.zeros(nparam)
        true_param[np.random.permutation(nparam)[:sparsity]]=np.random.normal(
            0,1,sparsity)
        #make sure to set edge coefficients to be non-zero
        mfsurr_true.set_param(true_param)
        for e in mfsurr_true.graph.edges:
            if np.count_nonzero(mfsurr_true.graph.edges[e]['param'])==0:
                mfsurr_true.graph.edges[e]['param'][np.random.randint(
                    0,nedge_param)]=np.random.randn()

        for n in mfsurr_true.graph.nodes:
            if np.count_nonzero(mfsurr_true.graph.nodes[n]['param'])==0:
                mfsurr_true.graph.nodes[n]['param'][np.random.randint(
                    0,nnode_param)]=np.random.randn()
            #print(mfsurr_true.graph.nodes[n]['param'],n)
        true_param = mfsurr_true.get_param()


        dx = 1
        ndata = 500
        x = np.random.rand(ndata, dx)
        y, _ =  mfsurr_true.forward(x, node)
        std = 1.#1e-4

        mfsurr_learn = MFSurrogate(graph, roots)
        param_start = np.random.randn(nparam)

        res = sciopt.minimize(learn_obj_grad_both, param_start,
                              args=(mfsurr_learn, node, x, y,std),
                              method='BFGS', jac=True)
        lstsq_coef = res.x
        #print('est param',lstsq_coef)
        #print('true param',mfsurr_true.get_param())
        mfsurr_learn.set_param(lstsq_coef)
        predict = mfsurr_learn.forward(x, node)[0]
        assert np.linalg.norm(predict-y)**2/2<2e-15

        ntest=100
        x_test = np.random.rand(ntest, dx)
        y_test, _ =  mfsurr_true.forward(x_test, node)
        predict_test = mfsurr_learn.forward(x_test, node)[0]
        assert np.linalg.norm(predict_test-y_test)/np.sqrt(ntest)<1e-8

    @unittest.skipUnless(pyapprox_available, "PyApprox not available")
    def test_l1_opt(self):
        np.random.seed(2)

        nnode_param, nedge_param = 2, 1
        graph, roots = make_graph_8(nnode_param,nedge_param,linfunc=monomial_1d_lin)
        node = 8

        ## Truth
        mfsurr_true = MFSurrogate(graph, roots)
        nparam = mfsurr_true.get_nparam()
        sparsity = nparam // 4
        true_param = np.zeros(nparam)
        true_param[np.random.permutation(nparam)[:sparsity]]=np.random.normal(
            0,1,sparsity)
        #make sure to set edge coefficients to be non-zero
        mfsurr_true.set_param(true_param)
        for e in mfsurr_true.graph.edges:
            if np.count_nonzero(mfsurr_true.graph.edges[e]['param'])==0:
                mfsurr_true.graph.edges[e]['param'][np.random.randint(
                    0,nedge_param)]=np.random.randn()

        for n in mfsurr_true.graph.nodes:
            if np.count_nonzero(mfsurr_true.graph.nodes[n]['param'])==0:
                mfsurr_true.graph.nodes[n]['param'][np.random.randint(
                    0,nnode_param)]=np.random.randn()
            #print(mfsurr_true.graph.nodes[n]['param'],n)
        true_param = mfsurr_true.get_param()


        dx = 1
        ndata = 500
        x = np.random.rand(ndata, dx)
        y, _ =  mfsurr_true.forward(x, node)
        std = 1.#1e-4

        mfsurr_learn = MFSurrogate(graph, roots)
        param_start = np.random.randn(nparam)
        
        obj = partial(
            learn_obj_grad_both, graph = mfsurr_learn, node=node, x=x, y=y, std=std)

        lamda = 1e-4
        eps=1e-8
        options = {'ftol':1e-8,'disp':False,'maxiter':1000,'iprint':0, 'method':'slsqp'}

        l1_coef = nonlinear_basis_pursuit(obj,True,None,param_start,options,eps**2)

        mfsurr_learn.set_param(l1_coef)
        predict = mfsurr_learn.forward(x, node)[0]

        assert np.linalg.norm(predict-y)**2/2<2e-14

        ntest=100
        x_test = np.random.rand(ntest, dx)
        y_test, _ =  mfsurr_true.forward(x_test, node)
        predict_test = mfsurr_learn.forward(x_test, node)[0]
        assert np.linalg.norm(predict_test-y_test)/np.sqrt(ntest)<1e-8
        

    
    
if __name__== "__main__":    
    mfnet_test_suite = unittest.TestLoader().loadTestsFromTestCase(TestMfnet)
    unittest.TextTestRunner(verbosity=2).run(mfnet_test_suite)

