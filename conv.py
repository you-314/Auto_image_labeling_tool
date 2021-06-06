import sys, os
sys.path.append(os.pardir)
import numpy as np
import cupy as cp
from util import im2col, col2im

class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        # 中間データ（backward時に使用）
        self.x = None   
        self.col = None
        self.col_W = None
        
        # 重み・バイアスパラメータの勾配
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = int(1+(H + 2*self.pad - FH)/self.stride)
        out_w = int(1+(W + 2*self.pad - FW)/self.stride)

        #print(x.shape)
        col = im2col(x, FH, FW, self.stride, self.pad)
        #print("col="+str(col.shape))
        col_W = self.W.reshape(FN, -1).T
        #Sprint("col_W="+str(col_W.shape))
        # print("================col==================")
        # print(col[0], col[1])
        # print("=====================================\n\n")
        # print("================col_W=====================")
        # print(col_W)
        # print("==========================================\n\n")
        out = cp.dot(col, col_W) + self.b

        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out
    
    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = cp.sum(dout, axis=0)
        self.dW = cp.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = cp.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx