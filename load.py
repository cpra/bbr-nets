
'''
Load a trained multi-view classification and bounding box regression network.
Author: Christopher Pramerdorfer
License: zlib
'''

import torch
import torch.nn as nn
from torch.autograd import Variable

import os
import math

# init network

def bbr_combine_avg(sbb, fbb, tbb):
    '''
    Combine per-view regression results.
    Returns the result as [x0, y0, z0, x1, y1, z1].
    '''

    x0 = (fbb[:, 1] + tbb[:, 1]) / 2.0
    x1 = (fbb[:, 3] + tbb[:, 3]) / 2.0

    y0 = (sbb[:, 1] + tbb[:, 0]) / 2.0
    y1 = (sbb[:, 3] + tbb[:, 2]) / 2.0

    z0 = (sbb[:, 0] + fbb[:, 0]) / 2.0
    z1 = (sbb[:, 2] + fbb[:, 2]) / 2.0

    return torch.stack((x0, y0, z0, x1, y1, z1), dim=1)

class SingleViewNet(nn.Module):
    '''
    Frontend network for a single view.
    '''

    def __init__(self, cfg):
        super(SingleViewNet, self).__init__()

        dim = 1
        layers = []

        for v in cfg:
            ltype = v['type']

            if ltype == 'P':
                size = v['size']
                stride = v['stride']
                layers += [nn.MaxPool2d(kernel_size=size, stride=stride)]
            elif ltype == 'C':
                filters = v['filters']
                size = v['size']
                pad = v['pad']
                layers += [nn.Conv2d(dim, filters, kernel_size=size, padding=pad), nn.BatchNorm2d(filters), nn.ReLU(inplace=True)]
                dim = filters
            elif ltype == 'D':
                p = v['p']
                view_layers += [nn.Dropout(p)]
            else:
                raise RuntimeError('Invalid layer type "{}"'.format(ltype))

        self._net = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        return self._net(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class MultiViewBBRNet(nn.Module):
    '''
    Multi-view classification and bounding box regression network.
    '''

    def __init__(self, input_size, num_classes, cfg, num_fc, drop_fc):
        super(MultiViewBBRNet, self).__init__()

        # occupancy frontends

        self._scnn = SingleViewNet(cfg)
        self._fcnn = SingleViewNet(cfg)
        self._tcnn = SingleViewNet(cfg)

        # occlusion frontends (pool to match size of occupancy frontend outputs)

        self._slpool = nn.Sequential(*[nn.MaxPool2d(kernel_size=c['size'], stride=c['stride']) for c in cfg if c['type'] == 'P'])
        self._flpool = nn.Sequential(*[nn.MaxPool2d(kernel_size=c['size'], stride=c['stride']) for c in cfg if c['type'] == 'P'])
        self._tlpool = nn.Sequential(*[nn.MaxPool2d(kernel_size=c['size'], stride=c['stride']) for c in cfg if c['type'] == 'P'])

        # mid-sections

        nuin = self._scnn(Variable(torch.zeros(1, 1, input_size, input_size))).numel()
        nlin = self._slpool(Variable(torch.zeros(1, 1, input_size, input_size))).numel()
        nin = nuin + nlin

        self._smid = nn.Sequential(
            nn.Linear(nin, num_fc),
            nn.ReLU(True),
            nn.Dropout(drop_fc)
        )

        self._fmid = nn.Sequential(
            nn.Linear(nin, num_fc),
            nn.ReLU(True),
            nn.Dropout(drop_fc)
        )

        self._tmid = nn.Sequential(
            nn.Linear(nin, num_fc),
            nn.ReLU(True),
            nn.Dropout(drop_fc)
        )

        # classifiers

        self._sclf = nn.Linear(num_fc, num_classes)
        self._fclf = nn.Linear(num_fc, num_classes)
        self._tclf = nn.Linear(num_fc, num_classes)

        # regressors

        self._sbb = nn.Linear(num_fc, 4)
        self._fbb = nn.Linear(num_fc, 4)
        self._tbb = nn.Linear(num_fc, 4)

    def forward(self, uside, ufront, utop, lside, lfront, ltop):
        # occupancy frontend

        xs = self._scnn(uside)
        xf = self._fcnn(ufront)
        xt = self._tcnn(utop)

        # occlusion frontend

        ls = self._slpool(lside)
        lf = self._flpool(lfront)
        lt = self._tlpool(ltop)

        # concat

        xs = torch.cat([xs, ls], dim=1)
        xf = torch.cat([xf, lf], dim=1)
        xt = torch.cat([xt, lt], dim=1)

        # mid-section

        xs = xs.view(xs.size(0), -1)
        xf = xf.view(xf.size(0), -1)
        xt = xt.view(xt.size(0), -1)

        xs = self._smid(xs)
        xf = self._fmid(xf)
        xt = self._tmid(xt)

        # backend

        sclf = self._sclf(xs)
        fclf = self._fclf(xf)
        tclf = self._tclf(xt)

        clf = torch.stack((sclf, fclf, tclf), dim=2)
        clf = torch.mean(clf, dim=2)

        sbb = self._sbb(xs)
        fbb = self._fbb(xf)
        tbb = self._tbb(xt)

        return clf, sbb, fbb, tbb

class BBCombinationNet(nn.Module):
    '''
    Wrapper around any other multi-view net that combines regression results via `bbr_combine_avg`.
    '''

    def __init__(self, net):
        super(BBCombinationNet, self).__init__()

        self._net = net

    def forward(self, uside, ufront, utop, lside=None, lfront=None, ltop=None):
        occ = lside is not None and lfront is not None and ltop is not None

        if(occ):
            cids, sbb, fbb, tbb = self._net(uside, ufront, utop, lside, lfront, ltop)
        else:
            cids, sbb, fbb, tbb = self._net(uside, ufront, utop)

        return cids, bbr_combine_avg(sbb, fbb, tbb)

net = MultiViewBBRNet(
    50,
    2,
    [
        { 'type': 'C', 'size': 3, 'pad': 1, 'filters': 64 },
        { 'type': 'C', 'size': 3, 'pad': 1, 'filters': 64 },
        { 'type': 'P', 'size': 2, 'stride': 2 },
        { 'type': 'C', 'size': 3, 'pad': 1, 'filters': 96 },
        { 'type': 'C', 'size': 3, 'pad': 1, 'filters': 96 },
        { 'type': 'P', 'size': 2, 'stride': 2 },
        { 'type': 'C', 'size': 3, 'pad': 1, 'filters': 128 },
        { 'type': 'C', 'size': 3, 'pad': 1, 'filters': 128 },
        { 'type': 'P', 'size': 2, 'stride': 2 }
    ],
    512,
    0.25
)

# load parameters

assert(os.path.exists('params.pth'))

state = torch.load('params.pth')
net.load_state_dict(state)
net.eval()
