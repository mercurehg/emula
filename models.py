
import math
import torch
from torch import nn

class CNNet(nn.Module):
    def __init__(self,
                 num_class) -> None:
        super().__init__()
        k_size = 3
        dilation = 1
        pad = (k_size + (k_size-1) * (dilation-1) - 1) // 2

        self.conv1 = nn.Sequential(nn.Conv1d(1, 32, k_size, 1, padding=pad), nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv1d(32, 64, k_size, 1, padding=pad), nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv1d(64, 128, k_size, 1, padding=pad), nn.ReLU(True))
        self.fc = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(128, num_class))

    def forward(self, x): # [N,F]
        x = self.conv1(x.unsqueeze(1))
        x = self.conv2(x)
        x = self.conv3(x)
        return self.fc(x)


class LSTMNet(nn.Module):
    def __init__(self,
                 num_class) -> None:
        super().__init__()

        self.lstm1 = nn.LSTM(1, 32, batch_first=True)
        self.lstm2 = nn.LSTM(32, 64, batch_first=True)
        self.lstm3 = nn.LSTM(64, 128, batch_first=True)
        self.fc = nn.Linear(128, num_class)

    def forward(self, x):
        x, h0 = self.lstm1(x.unsqueeze(2))
        x, h0 = self.lstm2(x)
        x, h0 = self.lstm3(x)
        return self.fc(x.mean(1))

class Ensemble_MuLANet(nn.Module):
    def __init__(self, 
                 feats, 
                 use_atten, 
                 atten_kerns,
                 num_class,
                 concat='') -> None:
        super().__init__()
        
        assert concat != ''
        self.concat = concat

        self.ensemble = nn.ModuleList()

        for e in range(len(feats)):
            layers = []
            chs = [feats[e], 64, 128, 256]
            for c in range(1,len(chs)):
                if use_atten == False:
                    l = nn.Sequential(nn.Linear(chs[c-1], chs[c]),
                                    nn.ReLU())
                else:
                    l = nn.Sequential(nn.Linear(chs[c-1], chs[c]),
                                    nn.ReLU(),
                                    MultiLocalAtten(chs[c], atten_kerns))
                l.apply(weights_init)
                layers.append(l)            

            mlp = nn.Sequential(nn.Flatten(),
                                nn.Linear(256, 128),
                                nn.ReLU(True),
                                nn.Linear(128, 64),
                                nn.ReLU(True)
                                )
            mlp.apply(weights_init)

            layers.append(mlp)
            layers = nn.Sequential(*layers)
            self.ensemble.append(layers)

        if concat in ['sum', 'mean', 'max']:        
            self.clf = nn.Linear(64, num_class)
        elif concat == 'concat':
            self.clf = nn.Linear(64*len(feats), num_class)

        nn.init.normal_(self.clf.weight, 0, math.sqrt(2. / num_class))
    
    def merge(self, x):
        if self.concat == 'sum':
            x = torch.stack(x, dim=-1).sum(dim=-1)
        elif self.concat == 'mean':
            x = torch.stack(x, dim=-1).mean(dim=-1)
        elif self.concat == 'max':
            x = torch.stack(x, dim=-1).max(dim=-1)[0]
        elif self.concat == 'concat':
            x = torch.concat(x, dim=-1)
        return x

    def forward(self, x): # [N,F]
        out = []
        for e in range(len(self.ensemble)):
            out.append(self.ensemble[e](x[e]))
        
        out = self.merge(out)
        out = self.clf(out)
        return out
    
class MuLANet(nn.Module):
    def __init__(self, 
                 feat, 
                 use_atten, 
                 atten_kerns,
                 num_class) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        chs = [feat, 64, 128, 256]
        for c in range(1,len(chs)):
            if use_atten == False:
                l = nn.Sequential(nn.Linear(chs[c-1], chs[c]),
                                nn.ReLU())
            else:
                l = nn.Sequential(nn.Linear(chs[c-1], chs[c]),
                                nn.ReLU(),
                                MultiLocalAtten(chs[c], atten_kerns))
            l.apply(weights_init)
            self.layers.append(l)            

        self.mlp = nn.Sequential(nn.Flatten(),
                                nn.Linear(256, 128),
                                nn.ReLU(True),
                                nn.Linear(128, 64),
                                nn.ReLU(True),
                                nn.Linear(64, num_class)
                                )
        #self.clf = nn.Linear(64, num_class)
        self.mlp.apply(weights_init)
        #nn.init.normal_(self.clf.weight, 0, math.sqrt(2. / num_class))
    
    def forward(self, x): # [N,F]
        N, F = x.shape
        for l in range(len(self.layers)):
            x = self.layers[l](x)
        x = self.mlp(x)
        #x = self.clf(x)
        return x

### Final manuscript
    
class MultiLocalAtten(nn.Module):
    def  __init__(self, in_feat, k_sizes:list) -> None:
        super().__init__()

        assert len(k_sizes) != 0

        self.locals = nn.ModuleList()
        for k in k_sizes:
            self.locals.append(LocalAtten1D(k))

        self.conv = nn.Sequential(
                    nn.Conv1d(in_feat, in_feat, kernel_size=len(k_sizes), stride=1, bias=False))
                    #nn.ReLU(True))

    def forward(self, x): #[N,F]
        out = []
        for l in range(len(self.locals)):
            out.append(self.locals[l](x))
        out = torch.stack(out, dim=2)
        out = self.conv(out).squeeze(2)
        return out

class LocalAtten1D(nn.Module):
    def __init__(self, 
                 k_size) -> None:
        super().__init__()

        dilation = 1
        pad = (k_size + (k_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv1d(1, 1, k_size, 1, padding=pad, bias=False)
        self.conv.apply(weights_init)

    def forward(self, x): #[N,F]
        atten = torch.sigmoid(self.conv(x.unsqueeze(1)).squeeze(1))
        return x * atten
    

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0)