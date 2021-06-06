import numpy as np
from conv import Convolution as conv
from pool import Pooling as pool
import layer
import cupy as cp
from collections import OrderedDict
import pickle

class SimpleConvNet:
    def __init__(self, input_dim=(1, 100, 100),conv_param = {'filter_num':30, 'filter_size':5,'pad':0, 'stride':1},
                hidden_size = 100, output_size=10, weight_init_std=0.01):
        
        #filter param 
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2*filter_pad)//filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size/2)**2) 

        #パラメータをdictionaryに格納
        self.params = {}
        self.params['W1'] = weight_init_std * cp.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        #print("W1="+str(self.params["W1"].shape))

        self.params['b1'] = cp.zeros(filter_num)
        self.params['W2'] = weight_init_std * cp.random.randn(pool_output_size, hidden_size)
        #print("W2="+str(self.params["W2"].shape))

        self.params['b2'] = cp.zeros(hidden_size)
        self.params['W3'] = weight_init_std * cp.random.randn(hidden_size, output_size)
        #print("W3="+str(self.params["W3"].shape))

        self.params['b3'] = cp.zeros(output_size)

        #layerをdeictionaryに格納
        self.layers = OrderedDict()
        self.layers['Conv1'] = conv(self.params['W1'], self.params['b1'], conv_param['stride'], conv_param['pad'])

        self.layers['Relu1'] = layer.ReLU()

        self.layers['Pool1'] = pool(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = layer.Affine(self.params['W2'], self.params['b2'])

        self.layers['Relu2'] = layer.ReLU()
        self.layers['Affine2'] = layer.Affine(self.params['W3'], self.params['b3'])

        self.last_layer = layer.SoftmaxWithLoss()

        
    def predict(self, x):
        
        for layer_value in self.layers.values():
            x = layer_value.forward(x)

        #print("x_out="+str(x.shape))
        return x


    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)


    def gradient(self, x, t):
        #forward
        self.loss(x, t)

        #backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        
        #設定
        grads = {}
        grads['W1'] = self.layers['Conv1'].dW
        grads['b1'] = self.layers['Conv1'].db
        grads['W2'] = self.layers['Affine1'].dw
        grads['b2'] = self.layers['Affine1'].db
        grads['W3'] = self.layers['Affine2'].dw
        grads['b3'] = self.layers['Affine2'].db
        
        return grads

    
    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1 : t = cp.argmax(t,axis=1)
        
        acc = 0.0
        
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx)
            y = cp.argmax(y, axis=1)
            acc += cp.sum(y == tt) 
        print("acc="+str(acc))
        return acc / x.shape[0]

    def save_params(self, file_name="params2.pkl"):  
            params = {}
            for key, val in self.params.items():
                params[key] = cp.asnumpy(val)
            with open(file_name, 'wb') as f:
                pickle.dump(params, f)