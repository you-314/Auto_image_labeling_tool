import numpy as np
import cupy as cp

def softmax(x):
        # c = cp.max(x)
        # exp_x = cp.exp(x-c)
        # sum_exp_x = cp.sum(exp_x)
        # y = exp_x / sum_exp_x

        #print(type(y))
    x = x - np.max(x, axis=-1, keepdims=True)   # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


def cross_entropy_error(y, t):
    ans = -1
    y = cp.asnumpy(y)
    t = cp.asnumpy(t)
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    batch_size = int(batch_size)
    ans = -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
    return ans

class Affine:
    def __init__(self, w, b):
        self.w = w
        self.b = b
        self.x = None
        self.dw = None
        self.db = None
    
    def forward(self, x):
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        out = cp.dot(x, self.w) + self.b

        #print("Affine="+str(out.shape))
        return out
    
    def backward(self, dout):
        dx = cp.dot(dout, self.w.T)
        self.dw = cp.dot(self.x.T, dout)
        self.db = cp.sum(dout, axis=0)
        dx = dx.reshape(*self.original_x_shape)

        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        #print("soft="+str(self.y.shape))
        self.loss = cross_entropy_error(self.y, self.t)
        #print("loss="+str(self.loss))

        return self.loss
    
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 教師データがone-hot-vectorの場合
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[cp.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx

class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        #print("ReLU_OUT="+str(out.shape))
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx

