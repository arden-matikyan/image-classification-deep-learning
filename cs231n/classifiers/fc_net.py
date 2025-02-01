from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


        # print("in dim", input_dim, " hid ", hidden_dims[0])
        # seperating first layer 
        self.params['W1'] = np.random.normal(scale=weight_scale, size=(input_dim, hidden_dims[0]))
        self.params['b1'] = np.zeros(hidden_dims[0])
          
        if self.normalization:
          print("layer or batch")
          self.params['gamma1'] = np.ones(hidden_dims[0])
          self.params['beta1'] = np.zeros(hidden_dims[0])

        # hidden layers 
        for l in range(1, len(hidden_dims)):
          # print(" hid-1 ", hidden_dims[l-1], " hid ", hidden_dims[l])
          self.params[f'W{l+1}'] = np.random.normal(scale = weight_scale, size=(hidden_dims[l-1], hidden_dims[l]))
          self.params[f'b{l+1}'] = np.zeros(hidden_dims[l])

          if self.normalization:
            self.params[f'gamma{l+1}'] = np.ones(hidden_dims[l])
            self.params[f'beta{l+1}'] = np.zeros(hidden_dims[l])


        l = len(hidden_dims)-1
        # last/output layer 
        self.params[f'W{l+2}'] = np.random.normal(scale = weight_scale, size=(hidden_dims[-1], num_classes))
        self.params[f'b{l+2}'] = np.zeros(num_classes)

        '''
        for l in range(0,len(hidden_dims)+1):
          print("pram: ", self.params[f'W{l+1}'].shape)
        print("layers initialized")
        ''' 

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.
        
        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
            scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
            names to gradients of the loss with respect to those parameters.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #  0, 1, ...., : 
        
        cache_dict = {}
        for i in range(self.num_layers):
      
          keys = [f'W{i+1}', f'b{i+1}', f'gamma{i+1}', f'beta{i+1}']   # list of params
          W, b, gamma, beta = (self.params.get(k, None) for k in keys) # get param vals

          # print(" XS: ", X.shape, " W ", W.shape, " i" , i+1)
          # def conv_bn_relu_forward(x, w, b, gamma, beta, bn_param,last):
          # Convenience layer that performs a convolution, a batch normalization, and a ReLU.
          # store caches in cache dict 

          # normalization parameters 
          bn = self.bn_params[i] if gamma is not None else None  
          # dropout paramters 
          do = self.dropout_param if self.use_dropout else None

          last = False
          if i==self.num_layers-1:
            last = True 

          X, cache_dict[i] = bn_affine_relu_forward(X,W,b,gamma,beta, bn, do ,last)

        scores = X
           
        '''
        z1, z1_cache = affine_relu_forward(X, self.params['W1'], self.params['b1'])
        z2, z2_cache = affine_forward(z1, self.params['W2'], self.params['b2'])
        scores = z2
        ''' 

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early.
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully connected net. Store the #
        # ... # 
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss, d_scores = softmax_loss(scores, y)
        reg_weights = 0 
        # calculate squared sum of elements for each weight matrix
        for k, W in self.params.items():
          if 'W' in k:
            reg_weights += np.sum(W**2)
        
        loss += 0.5 * self.reg * reg_weights


        # print(len(cache_dict))
        for i in reversed(range(self.num_layers)):
          # print(i)
          d_scores, dW, db, dgamma, dbeta = bn_affine_relu_backward(d_scores, cache_dict[i])

          grads[f'W{i+1}'] = dW + self.reg * self.params[f'W{i+1}']
          grads[f'b{i+1}'] = db

          if dgamma is not None and i < self.num_layers-1:
                grads[f'gamma{i+1}'] = dgamma
                grads[f'beta{i+1}'] = dbeta


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
