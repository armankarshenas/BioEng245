"""
School: University of California, Berkeley
Course: BIOENG 145/245
Author: Yorick Chern
Instructor: Liana Lareau
Assignment 5
"""
import numpy as np

def linear_forward(X, w, b):
    """
    Q:  calculate the output of a node with no activation function: Z = Xw + b

    Inputs
    - X: input matrix (N, D)
    - w: linear node weight matrix (D, D')
    - b: the bias vector of shape (D', )
    """

    output = np.matmul(X,w)+b
    return output

def relu_forward(X):
    """
    Q:  Z = relu(X) = max(X, 0)

    Inputs
    - X: input matrix (N, D)
    """
    Z = np.maximum(X,0)
    return Z

def linear_backward(X, w, b):
    """
    Q:  with Z = Xw + b, find dZ/dw and dZ/db

    Outputs a tuple of...
    - (dZdw, dZdb)
    """
    dzdw = X
    dzdb = 1+np.zeros_like(b)
    return (dzdw,dzdb)

def relu_backward(X):
    """
    Q:  Z = relu(X) = max(X, 0), find dZ/dX

    Inputs
    - X: input matrix (N, D)
    """
    idx = X > 0
    output = np.array(idx).astype(np.double)
    return output

def softmax_forward(X):
    """
    Q:  Z = softmax(X)

    Inputs
    - X: input matrix (N, D)
    """
    N, D = X.shape
    exp_arr = np.exp(X)
    exp_sum = np.sum(exp_arr,axis=1)
    exp_sum = np.repeat(exp_sum,D)
    exp_sum = np.reshape(exp_sum,[N,D])
    output = np.divide(exp_arr,exp_sum)
    return output

def softmax_backward(X):
    """
    Q:  Z = softmax(X), find dZ/dX
    """
    X = np.array(X)
    fn = softmax_forward(X)
    N, D = X.shape
    size = np.shape(X)
    if len(size) > 1:
        out = np.zeros([1, size[1]])
        for i in range(size[0]):
            fn_t = fn[i, :]
            s = fn_t.reshape(-1, 1)
            temp = (np.diagflat(s) - np.dot(s, s.T))
            out = np.vstack((out, temp))
        out = out[1:, :]
        out = np.reshape(out,[N,D,D])
    else:
        s = fn.reshape(-1, 1)
        out = np.diagflat(s) - np.dot(s, s.T)

    return out

if __name__ == '__main__':

    # due to popular demand, we have increased the number and clarity of test cases available to you

    # we will use this (5, 8) matrix to test all the functions above!
    X = np.array([[-0.38314889,  0.35891731,  0.09037955,  0.98397352, -0.74292248, -0.5883056,  0.54354937,  0.79001348],
                  [ 0.58758113, -0.412598  ,  0.08740239, -0.68723605, -0.29251551, 0.36521658, -0.25330565,  0.03919754],
                  [ 0.97960327, -0.41368028,  0.26308195,  0.94303171, -0.92383705, 0.28187289,  0.35914219, -0.46526478],
                  [ 0.2583081 ,  0.97956892,  0.31049517, -0.68557195, -0.68612885, -0.9054485, -0.70507179,  0.11431403],
                  [-0.7674351 ,  0.69421738, -0.8007104 ,  0.93470719,  0.61132148, 0.54328029,  0.00919623, -0.34544161]])

    # predefined weights and biases to ease your testing
    w = np.array([[0.77805922, 0.67805674, 0.18799035, 0.93644034, 0.87466635, 0.66450703],
                  [0.86038224, 0.21901606, 0.87774923, 0.21039304, 0.76061141, 0.37033866],
                  [0.49032109, 0.71247207, 0.61826719, 0.37348737, 0.4197679 , 0.70488014],
                  [0.37720786, 0.39471295, 0.68555261, 0.48458372, 0.29309447, 0.01436672],
                  [0.68969515, 0.10709357, 0.02608303, 0.35893371, 0.53729841, 0.53873035],
                  [0.31109099, 0.99274133, 0.78935902, 0.77859174, 0.02639908, 0.17466261],
                  [0.30502676, 0.07085277, 0.03068556, 0.4183926 , 0.07385148, 0.99708494],
                  [0.87156768, 0.47651573, 0.76058837, 0.1566234 , 0.95023629, 0.78754312]])
    b = np.array([-0.41458132, -0.5132465, 0.13485534, 0.31234684, -0.1248132, 0.34524132])

    N, D = X.shape

    # forward pass
    linear_test = linear_forward(X, w, b)
    print("linear_test")
    print(linear_test)

    """
    linear_test:
    array([[ 0.17053049 -0.49028621  1.24210294  0.16607946  0.5155312   0.96254506]
           [-0.66000863 -0.08221805 -0.23124341  0.55491303 -0.21842568  0.3191322 ]
           [-0.36904111  0.60467294  0.62060448  1.66314573 -0.10032968  0.58519508]
           [-0.34748938 -1.14078261  0.09789411 -0.6842616   0.43958586 -0.0521021 ]
           [-0.16206085 -0.64226044  0.92800334  0.48579765 -0.31495605 -0.2972579 ]])
    """

    relu_test = relu_forward(linear_test)
    print("relu_test")
    print(relu_test)

    """
    relu_test:
    array([[0.17053049 0.         1.24210294 0.16607946 0.5155312  0.96254506]
          [0.         0.         0.         0.55491303 0.         0.3191322 ]
          [0.         0.60467294 0.62060448 1.66314573 0.         0.58519508]
          [0.         0.         0.09789411 0.         0.43958586 0.        ]
          [0.         0.         0.92800334 0.48579765 0.         0.        ]])
    """

    softmax_test = softmax_forward(relu_test)
    print("softmax_test")
    print(softmax_test)

    """
    softmax:
    array([[0.10662601 0.08990891 0.31134448 0.10615247 0.15055496 0.23541316]
           [0.14049437 0.14049437 0.14049437 0.24471163 0.14049437 0.19331088]
           [0.07835807 0.14344646 0.14575008 0.41340786 0.07835807 0.14067947]
           [0.15026499 0.15026499 0.16571914 0.15026499 0.23322092 0.15026499]
           [0.12262529 0.12262529 0.31017499 0.19932386 0.12262529 0.12262529]])
    """

    assert np.all(np.abs(np.sum(softmax_test, axis=1) - 1.0) < 1e-10), "Rows of softmax output need to sum to 1!"

    # backward pass
    softmax_back_test = softmax_backward(relu_test)
    print(softmax_back_test.__repr__())

    """
    softmax_back_test:
    """
    sm = np.array([[[ 0.09525691, -0.00958663, -0.03319742, -0.01131862, -0.01605308, -0.02510117],
                    [-0.00958663,  0.0818253 , -0.02799264, -0.00954405, -0.01353623, -0.02116574],
                    [-0.03319742, -0.02799264,  0.2144091 , -0.03304999, -0.04687446, -0.07329459],
                    [-0.01131862, -0.00954405, -0.03304999,  0.09488413, -0.01598178, -0.02498969],
                    [-0.01605308, -0.01353623, -0.04687446, -0.01598178, 0.12788817, -0.03544262],
                    [-0.02510117, -0.02116574, -0.07329459, -0.02498969, -0.03544262,  0.1799938 ]],

                   [[ 0.12075571, -0.01973867, -0.01973867, -0.03438061, -0.01973867, -0.02715909],
                    [-0.01973867,  0.12075571, -0.01973867, -0.03438061, -0.01973867, -0.02715909],
                    [-0.01973867, -0.01973867,  0.12075571, -0.03438061, -0.01973867, -0.02715909],
                    [-0.03438061, -0.03438061, -0.03438061,  0.18482785, -0.03438061, -0.04730542],
                    [-0.01973867, -0.01973867, -0.01973867, -0.03438061, 0.12075571, -0.02715909],
                    [-0.02715909, -0.02715909, -0.02715909, -0.04730542, -0.02715909,  0.15594178]],

                   [[ 0.07221808, -0.01124019, -0.01142069, -0.03239384, -0.00613999, -0.01102337],
                    [-0.01124019,  0.12286957, -0.02090733, -0.05930189, -0.01124019, -0.02017997],
                    [-0.01142069, -0.02090733,  0.124507  , -0.06025423, -0.01142069, -0.02050404],
                    [-0.03239384, -0.05930189, -0.06025423,  0.2425018 , -0.03239384, -0.058158  ],
                    [-0.00613999, -0.01124019, -0.01142069, -0.03239384, 0.07221808, -0.01102337],
                    [-0.01102337, -0.02017997, -0.02050404, -0.058158  , -0.01102337,  0.12088875]],

                   [[ 0.12768542, -0.02257957, -0.02490178, -0.02257957, -0.03504494, -0.02257957],
                    [-0.02257957,  0.12768542, -0.02490178, -0.02257957, -0.03504494, -0.02257957],
                    [-0.02490178, -0.02490178,  0.13825631, -0.02490178, -0.03864917, -0.02490178],
                    [-0.02257957, -0.02257957, -0.02490178,  0.12768542, -0.03504494, -0.02257957],
                    [-0.03504494, -0.03504494, -0.03864917, -0.03504494, 0.17882892, -0.03504494],
                    [-0.02257957, -0.02257957, -0.02490178, -0.02257957, -0.03504494,  0.12768542]],

                   [[ 0.10758833, -0.01503696, -0.0380353 , -0.02444215, -0.01503696, -0.01503696],
                    [-0.01503696,  0.10758833, -0.0380353 , -0.02444215, -0.01503696, -0.01503696],
                    [-0.0380353 , -0.0380353 ,  0.21396646, -0.06182527, -0.0380353 , -0.0380353 ],
                    [-0.02444215, -0.02444215, -0.06182527,  0.15959386, -0.02444215, -0.02444215],
                    [-0.01503696, -0.01503696, -0.0380353 , -0.02444215, 0.10758833, -0.01503696],
                    [-0.01503696, -0.01503696, -0.0380353 , -0.02444215, -0.01503696,  0.10758833]]])

    assert np.all(np.abs(sm - softmax_back_test) < 1e-8), "Softmax gradient incorrect!"

    relu_back_test = relu_backward(relu_test)
    print("relu_back_test")
    print(relu_back_test.__repr__())
    """
    relu_back_test:
    """
    rbt = np.array([[1., 0., 1., 1., 1., 1.],
                    [0., 0., 0., 1., 0., 1.],
                    [0., 1., 1., 1., 0., 1.],
                    [0., 0., 1., 0., 1., 0.],
                    [0., 0., 1., 1., 0., 0.]])
    assert np.all(np.abs(rbt - relu_back_test) < 1e-20), "relu gradient incorrect!"

    linear_back_test, bias_back_test = linear_backward(X, w, b)
    print("linear_back_test")
    print(linear_back_test)
    lbt = np.array([[-0.38314889,  0.35891731,  0.09037955,  0.98397352, -0.74292248, -0.5883056,  0.54354937,  0.79001348],
                    [ 0.58758113, -0.412598  ,  0.08740239, -0.68723605, -0.29251551, 0.36521658, -0.25330565,  0.03919754],
                    [ 0.97960327, -0.41368028,  0.26308195,  0.94303171, -0.92383705, 0.28187289,  0.35914219, -0.46526478],
                    [ 0.2583081 ,  0.97956892,  0.31049517, -0.68557195, -0.68612885, -0.9054485, -0.70507179,  0.11431403],
                    [-0.7674351 ,  0.69421738, -0.8007104 ,  0.93470719,  0.61132148, 0.54328029,  0.00919623, -0.34544161]])
    assert np.all(np.abs(lbt - linear_back_test) < 1e-10), "Linear gradient incorrect!"
    print("bias")
    print(bias_back_test)   # should be 1

    print("All tests passed!")
