import torch
from pytorch_wavelets import DWTForward, DWTInverse
xfm = DWTForward(J=1, wave='db1', mode='zero')
X = torch.randn(10,5,64,64)
Yl, Yh = xfm(X)
Yh=Yh[0].reshape(10,5,32,-1)
print(Yh.shape,Yl.shape)
ifm = DWTInverse(wave='db1', mode='zero')
Y = ifm((Yl))