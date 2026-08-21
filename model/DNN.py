import logging
from torch.utils.data import Subset

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from opacus import PrivacyEngine

from model.CNN_MNIST import CNN_MNIST
from model.CNN_UTKFace import CNN_UTKFace, CNN_UTKFace_dropout
from model.CNN_bsq import CNN_UTKFace_bsq, CNN_UTKFace_bsq_dropout, CNN_MNIST_bsq
from model.Simple_CNN_bsq import Simple_CNN_Tanh_bsq
from model.MLP_bsq import MLP_bsq, MLP_VIB_bsq
from model.ResNet import resnet18, resnet50, resnet18_dp
from model.ResNet_dropout import resnet18_drop
from model.VGG import vgg11_bn
from model.resnet_bsq import resnet_bsq, resnet_bsq_dropout
from model.mlp_5layer_tanh import MLP_5Layer_Tanh
from model.mlp_5layer_tanh_dropout import MLP_5Layer_Tanh_Dropout
from model.mlp_5layer_bsq import MLP_5Layer_Tanh_bsq
from model.MobileNetV2 import mobilenetv2
from model.MobileNetV2_bsq import MobileNetV2_bsq
from model.resnet_ATnT import resnet18_ATnT
from opacus.validators import ModuleValidator


class DNN(nn.Module):
    def __init__(self, args=None):
        super(DNN, self).__init__()

        self.logger = logging.getLogger("DNN")
        self.args = args
        self.device = args['device']
        self.in_chan=3
        if args['dataset_name']=='tinyimagenet':
            self.imagenet = True
        else:
            self.imagenet = False
        if args['dataset_name']=='cifar100':
            self.num_classes = 100
        elif args['dataset_name']=='tinyimagenet':
            self.num_classes = 200
        elif args['dataset_name']=='UTKFace' or args['dataset_name']=='celebA':  #predict the gender (male, female) of a person
            self.num_classes = 2
        elif args['dataset_name']=='vggface2':
            self.num_classes = 8631

        elif args['dataset_name']=='mnist' or args['dataset_name']=='fmnist':
            self.in_chan=1
            self.num_classes = 10
        elif args['dataset_name'] == 'ATnT':
            self.in_chan=1
            self.num_classes = 40
        elif args['dataset_name'] == 'binary_random':
            self.in_chan=1
            self.num_classes = 2
        elif args['dataset_name'] == 'texas':
            self.num_classes = 100
        elif args['dataset_name'] == 'location':
            self.num_classes = 8
        else:
            self.num_classes = 10
        self.model = self.determine_net(args['net_name'])

    def determine_net(self, net_name, pretrained=False):
        self.logger.debug("determin_net for  %s" % net_name)
        self.in_dim = {
            "location": 168,
            "texas": 6169,
            "adult": 14,
            "accident": 29,
            "stl10": 96*96*3,
            "cifar10": 32*32*3,
            "cifar100": 32 * 32 * 3,
            "svhn": 32 * 32 * 3,
            "mnist": 28*28*1,
            "fmnist": 28*28*1,
            "cinic10": 32 * 32 * 3,
            "tinyimagenet": 224 * 224 * 3,
            'UTKFace': 224 * 224 * 3,
            'celebA': 224 * 224 * 3,
            'fairface': 224 * 224 * 3,
            'ATnT': 112 * 92,
            'binary_random': 32 * 32,
            'vggface2': 224 * 224 * 3,
        }

        in_dim = self.in_dim[self.args['dataset_name']]
        out_dim = self.num_classes
        imagenet=self.imagenet
        in_chan=self.in_chan
        if net_name == "mlp" or net_name == 'mlp_dp':
            if self.args['quantization']=='bsq' :
                return MLP_bsq(in_dim, out_dim,Nbits=self.args['w_bit'])
            else:
                return MLP(in_dim, out_dim)
        if net_name in ("mlp_5layer", "mlp_5layer_dp", "mlp_5layer_adversarial"):
            if self.args['quantization'] == 'bsq':
                return MLP_5Layer_Tanh_bsq(in_dim, out_dim, Nbits=self.args['w_bit'])
            return MLP_5Layer_Tanh(in_dim, out_dim)
        if net_name.startswith("mlp_5layer_drop"):
            if net_name.startswith("mlp_5layer_drop010"):
                dropout_rate = 0.1
            elif net_name.startswith("mlp_5layer_drop015"):
                dropout_rate = 0.15
            elif net_name.startswith("mlp_5layer_drop020"):
                dropout_rate = 0.2
            else:
                raise ValueError(f"Unknown mlp_5layer dropout net_name: {net_name}")
            if self.args['quantization'] == 'bsq':
                raise NotImplementedError("bsq is not supported for mlp_5layer dropout defense")
            return MLP_5Layer_Tanh_Dropout(in_dim, out_dim, dropout_rate=dropout_rate)

        if net_name == "mlp_dropout":
            if self.args['quantization']=='bsq':
                return MLP_bsq(in_dim, out_dim,Nbits=self.args['w_bit'])
            else:
                return MLP_dropout(in_dim, out_dim,dropout_rate = 0.95)
        if net_name=='mlp_vib':
            if self.args['quantization']=='bsq':
                return MLP_VIB_bsq(in_dim, out_dim, Nbits=self.args['w_bit'])
            else:
                return MLP_VIB(in_dim, out_dim)
        elif net_name == "logistic":
            return LRTorchNet(in_dim=in_dim, out_dim=out_dim)
        elif net_name == "simple_cnn":
            if self.args['quantization'] == 'bsq':
                # Restore original behavior: do not force bin init by default.
                bin_init = bool(self.args.get('bsq_bin_init', False))
                return Simple_CNN_Tanh_bsq(
                    num_classes=out_dim,
                    in_channels=in_chan,
                    Nbits=self.args['w_bit'],
                    bin=bin_init,
                )
            return Simple_CNN_Tanh(num_classes=out_dim)
        elif net_name == "CNN_MNIST" or net_name == "CNN_MNIST_dp" or net_name=='CNN_MNIST_adversarial' or net_name=='CNN_MNIST_memorization' or net_name=='CNN_MNIST_HSIC':
            if self.args['quantization'] == 'bsq':
                return CNN_MNIST_bsq(num_classes=out_dim, Nbits=self.args['w_bit'], latent_dim=64)
            else:
                return CNN_MNIST(num_classes=out_dim, latent_dim=64)
        elif net_name == "simple_cnn_dropout":
            return Simple_CNN_Tanh_dropout(num_classes=out_dim)
        elif net_name == "resnet18" or net_name =='resnet18_adversarial':
            if self.args['quantization']=='qdrop' or self.args['quantization']=='hawq':
                model = models.resnet18(weights=None)
                model.fc = nn.Linear(model.fc.in_features, out_dim)
                return model

            elif self.args['quantization']=='bsq':
                model = resnet_bsq(num_classes=out_dim, depth=20, block_name='BasicBlock', Nbits=self.args['w_bit'],
                                   act_bit=self.args['a_bit'], bin=False)
                return model
            else:
                return resnet18(num_classes=out_dim, in_channels=in_chan)
        elif net_name == "mobilenetv2":
            if self.args['quantization'] == 'bsq':
                return MobileNetV2_bsq(
                    num_classes=out_dim,
                    in_channels=in_chan,
                    Nbits=self.args['w_bit'],
                    bin=False,
                )
            if self.args['quantization'] in ('qdrop', 'hawq'):
                model = models.mobilenet_v2(weights=None)
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, out_dim)
                return model
            return mobilenetv2(num_classes=out_dim, in_channels=in_chan)
        elif net_name in (
            "resnet18_drop010",
            "resnet18_drop015",
            "resnet18_drop020",
            "resnet18_drop010_full",
            "resnet18_drop015_full",
            "resnet18_drop020_full",
        ):
            if net_name.startswith("resnet18_drop010"):
                bsq_drop = 0.1
                conv_fc_drop = 0.1
            elif net_name.startswith("resnet18_drop015"):
                bsq_drop = 0.15
                conv_fc_drop = 0.15
            elif net_name.startswith("resnet18_drop020"):
                bsq_drop = 0.2
                conv_fc_drop = 0.2

            if self.args['quantization']=='bsq':
                model = resnet_bsq_dropout(num_classes=out_dim, depth=20, block_name='BasicBlock', Nbits=self.args['w_bit'],
                                   act_bit=self.args['a_bit'], bin=False, dropout_rate=bsq_drop)
                return model
            else:
                model = models.resnet18(weights=None)
                dropout_rate = conv_fc_drop
                model.conv1 = nn.Sequential(
                    model.conv1,
                    nn.Dropout2d(dropout_rate)
                )
                model.maxpool = nn.Sequential(
                    model.maxpool,
                    nn.Dropout2d(dropout_rate),
                )

                for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
                    layer = getattr(model, layer_name)
                    for block in layer:
                        block.conv1 = nn.Sequential(
                            block.conv1,
                            nn.Dropout2d(dropout_rate)
                        )
                        block.conv2 = nn.Sequential(
                            block.conv2,
                            nn.Dropout2d(dropout_rate)
                        )
                        if block.downsample is not None:
                            ds = block.downsample
                            if isinstance(ds, nn.Sequential) and len(ds) > 0:
                                parts = list(ds.children())
                                block.downsample = nn.Sequential(
                                    parts[0],
                                    nn.Dropout2d(dropout_rate),
                                    *parts[1:],
                                )
                            else:
                                block.downsample = nn.Sequential(
                                    ds,
                                    nn.Dropout2d(dropout_rate),
                                )
                    setattr(
                        model,
                        layer_name,
                        nn.Sequential(layer, nn.Dropout2d(dropout_rate)),
                    )

                model.avgpool = nn.Sequential(
                    model.avgpool,
                    nn.Dropout2d(dropout_rate),
                )

                in_features = model.fc.in_features
                model.fc = nn.Sequential(
                    nn.Dropout(dropout_rate),
                    nn.Linear(in_features, out_dim),
                    nn.Dropout(dropout_rate)
                )

                return model
        elif net_name == "resnet18_ATnT":
            return resnet18_ATnT(num_classes=out_dim)

        elif net_name == "resnet18_dp":
            if self.args['quantization']=='qdrop' or self.args['quantization']=='hawq':
                model = models.resnet18(weights=None)
                model.fc = nn.Linear(model.fc.in_features, out_dim)
                replace_bn_with_gn(model, num_groups=32)

                return model
            elif self.args['quantization']=='bsq':
                model = resnet_bsq(num_classes=out_dim, depth=20, block_name='BasicBlock', Nbits=self.args['w_bit'],
                                   act_bit=self.args['a_bit'], bin=False)
                return model
            else:
                return resnet18(num_classes=out_dim, in_channels=in_chan)
        elif net_name == "resnet50":
            return resnet50(num_classes=out_dim)
        elif net_name == "densenet":
            return models.densenet121(num_classes=out_dim)
        elif net_name == "vgg":
            return vgg11_bn(3, out_dim)
        elif net_name == "CNN_UTKFace" or net_name == "CNN_UTKFace_dp" or net_name=='CNN_UTKFace_adversarial':
            if self.args['quantization'] == 'bsq':
                return CNN_UTKFace_bsq(Nbits=self.args['w_bit'])
            else:
                return CNN_UTKFace()


        elif net_name =='CNN_UTKFace_dropout' :
            if self.args['quantization'] == 'bsq':
                return CNN_UTKFace_bsq_dropout(Nbits=self.args['w_bit'])
            else:
                return CNN_UTKFace_dropout()
        elif net_name == "LeNet":
            return LeNet(num_classes=out_dim,in_channels=in_chan)
        elif net_name == "DT":
            return DecisionTreeClassifier()
        elif net_name == "RF":
            return RandomForestClassifier()
        else:
            raise ValueError(f"Unknown network: {net_name}")

    def train_model(self, train_loader, test_loader, label_key=None):
        self.model = self.model.to(self.device)

        if self.args['optim'] == "SGD":
            optimizer = optim.SGD(self.model.parameters(), lr=self.args['lr'], momentum=0.9, weight_decay=1e-4)
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=self.args['lr'], weight_decay=1e-4)

        criterion = nn.CrossEntropyLoss()

        test_acc = 0.0
        for epoch in range(self.args['num_epochs']):
            self.model.train()
            losses = []

            for data, target in tqdm(train_loader):

                data = data.to(self.device)

                if isinstance(target, dict):
                    if label_key is None:
                        raise ValueError("label_key must be specified for multi-label datasets.")
                    target = target[label_key]
                target = target.to(torch.int64).to(self.device)

                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())


            train_acc = self.test_model_acc(train_loader,label_key)
           # test_acc = self.test_model_acc(test_loader,label_key)

           # test_acc = self.test_model_acc(test_loader)
            print(f' epoch:{epoch} | train acc:{round(train_acc, 4)}')



    def load_model(self, save_name):
        self.model.load_state_dict(torch.load(save_name))



    def predict_proba(self, test_case):
        self.model.eval()
        self.model = self.model.to(self.device)
        with torch.no_grad():
            feature = torch.unsqueeze(test_case.to(torch.float32), 0).to(self.device)
            logits = self.model(feature)
            posterior = F.softmax(logits, dim=1)
            if torch.isnan(posterior).any() or torch.isinf(posterior).any():
                print("Warning: NaN/Inf detected in posterior after softmax. Applying nan_to_num fix.")
                posterior = torch.nan_to_num(posterior, nan=0.0, posinf=1.0, neginf=0.0)  # Probabilities are [0,1]

            return posterior.detach().cpu().numpy()


    def test_model_acc(self, test_loader,label_key=None):
        self.model.eval()

        self.model = self.model.to(self.device)
        correct = 0

        with torch.no_grad():

            for data, target  in test_loader:
                data = data.to(self.device)
                if isinstance(target, dict):
                    if label_key is None:
                        raise ValueError("label_key must be specified for multi-label datasets.")
                    target = target[label_key]

                target = target.to(torch.int64).to(self.device)

                outputs = self.model(data).to(self.device)

                pred = outputs.argmax(dim=1, keepdim=True)

                correct += pred.eq(target.view_as(pred)).sum().item()

            return correct / len(test_loader.dataset)

    def logits(self,target_sample):
        if isinstance(target_sample, Subset):
            with torch.no_grad():
                feature=target_sample[0][0]
                label=target_sample[0][1]
                feature =torch.unsqueeze(feature.to(torch.float32), 0).to(self.device)
                label=torch.unsqueeze(torch.tensor(label, dtype=torch.long),0).to(self.device)
                logits = self.model(feature).to(self.device)
                logits=logits[0].detach().cpu().numpy()
                probs=np.exp(logits)/np.sum(np.exp(logits))
               # confidence=probs[label]
                confidence=np.max(probs)
                # confidence = np.clip(confidence, 1e-10, 1 - 1e-10)
                # confidence=np.log(confidence/(1-confidence))
        else:
            ValueError("This is not a Subset.")
        return confidence

    def forward(self,x):
        x=self.model(x)
        return x

    def get_mu_std(self, x):
        """Get mu and std from VIB model if available"""
        if hasattr(self.model, 'get_mu_std'):
            return self.model.get_mu_std(x)
        else:
            raise AttributeError(f"Model {type(self.model).__name__} does not have get_mu_std method")

    def get_last_hidden_layer(self, test_case):  #for UTKFace
        self.model.eval()
        self.model = self.model.to(self.device)
        test_case=test_case.to(self.device)
        with torch.no_grad():
            feature = torch.unsqueeze(test_case.to(torch.float32), 0).to(self.device)
            output = self.model.get_last_hidden_layer(feature)
            return output

    #if run bsq need to open this
    # def get_gradients(self, feature,label):
    #     self.model = self.model.to(self.device)
    #     criterion = nn.CrossEntropyLoss()
    #
    #     output = self.model(feature)
    #     loss = criterion(output, label)
    #     grads = torch.autograd.grad(loss, self.model.parameters())
    #     grads = [g.detach().clone() for g in grads]
    #     return grads
    #
    def pruning(self, threshold, drop=True):
        if self.model is not None and hasattr(self.model, 'pruning'):
            return self.model.pruning(threshold=threshold, drop=drop)
        else:
            raise AttributeError("model is not set or has no 'pruning' method.")
    #
    def total_param(self):
        if self.model is not None and hasattr(self.model, 'total_param'):
            return self.model.total_param()
        else:
            raise AttributeError("model is not set or has no 'total_param' method.")
    #
    def total_bit(self):
        if self.model is not None and hasattr(self.model, 'total_bit'):
            return self.model.total_bit()
        else:
            raise AttributeError("model is not set or has no 'total_bit' method.")

    def get_Nbits(self):
        if self.model is not None and hasattr(self.model, 'get_Nbits'):
            return self.model.get_Nbits()
        else:
            raise AttributeError("model is not set or has no 'get_Nbits' method.")

    def set_Nbits(self, Nbit_dict):
        if self.model is not None and hasattr(self.model, 'set_Nbits'):
            self.model.set_Nbits(Nbit_dict)
        else:
            raise AttributeError("model is not set or has no 'set_Nbits' method.")

    def to_bin(self):
        if self.model is not None and hasattr(self.model, 'to_bin'):
            self.model.to_bin()
        else:
            raise AttributeError("model is not set or has no 'to_bin' method.")
    #
    # def quant(self, maxbit=8):
    #     if self.model is not None and hasattr(self.model, 'quant'):
    #         self.model.quant(maxbit=maxbit)
    #     else:
    #         raise AttributeError("model is not set or has no 'quant' method.")
    #
    # def print_stat(self):
    #     if self.model is not None and hasattr(self.model, 'print_stat'):
    #         self.model.print_stat()
    #     else:
    #         raise AttributeError("model is not set or has no 'print_stat' method.")

class SimpleCNN(nn.Module):
    def __init__(self, in_dim=3, out_dim=10):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d( in_dim, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, out_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x



def standardize(x, bn_stats):
    if bn_stats is None:
        return x

    bn_mean, bn_var = bn_stats

    view = [1] * len(x.shape)
    view[1] = -1
    x = (x - bn_mean.view(view)) / torch.sqrt(bn_var.view(view) + 1e-5)

    # if variance is too low, just ignore
    x *= (bn_var.view(view) != 0).float()
    return x

class Simple_CNN_Tanh(nn.Module):
    def __init__(self,num_classes=10, in_channels=3, input_norm=None,**kwargs):
        super(Simple_CNN_Tanh, self).__init__()
        self.in_channels = in_channels
        self.features = None
        self.classifier = None
        self.norm = None
        self.num_classes=num_classes

        self.build(input_norm, **kwargs)

    def build(self, input_norm=None, num_groups=None,
              bn_stats=None, size=None):

        if self.in_channels == 3:
            if size == "small":
                cfg = [16, 16, 'M', 32, 32, 'M', 64, 'M']
            else:
                cfg = [32, 32, 'M', 64, 64, 'M', 128, 128, 'M']

            self.norm = nn.Identity()
        else:
            if size == "small":
                cfg = [16, 16, 'M', 32, 32]
            else:
                cfg = [64, 'M', 64]
            if input_norm is None:
                self.norm = nn.Identity()
            elif input_norm == "GroupNorm":
                self.norm = nn.GroupNorm(num_groups, self.in_channels, affine=False)
            else:
                self.norm = lambda x: standardize(x, bn_stats)

        layers = []
        act = nn.Tanh
       # act = nn.ReLU

        c = self.in_channels
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(c, v, kernel_size=3, stride=1, padding=1)

                layers += [conv2d, act()]
                c = v

        self.features = nn.Sequential(*layers)

        if self.in_channels == 3:
            hidden = 128
            self.classifier = nn.Sequential(nn.Linear(c * 4 * 4, hidden), act(), nn.Linear(hidden, self.num_classes))
        else:
            self.classifier = nn.Linear(c * 4 * 4, self.num_classes)

    def forward(self, x):
        if self.in_channels != 3:
            x = self.norm(x.view(-1, self.in_channels, 8, 8))
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Simple_CNN_Tanh_dropout(nn.Module):
    def __init__(self, num_classes=10, in_channels=3, input_norm=None, drop_p=0.95, **kwargs):
        super(Simple_CNN_Tanh_dropout, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.drop_p = drop_p
        self.build(input_norm, **kwargs)


    def build(self, input_norm=None, num_groups=None, bn_stats=None, size=None):
        c = self.in_channels

        if self.in_channels == 3:
            if size == "small":
                cfg = [16, 16, 'M', 32, 32, 'M', 64, 'M']
            else:
                cfg = [32, 32, 'M', 64, 64, 'M', 128, 128, 'M']

            self.norm = nn.Identity()
        else:
            if size == "small":
                cfg = [16, 16, 'M', 32, 32]
            else:
                cfg = [64, 'M', 64]
            if input_norm is None:
                self.norm = nn.Identity()
            elif input_norm == "GroupNorm":
                self.norm = nn.GroupNorm(num_groups, self.in_channels, affine=False)
            else:
                self.norm = lambda x: standardize(x, bn_stats)

        layers = []
        act = nn.Tanh

        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                if self.drop_p > 0:
                    layers += [nn.Dropout2d(self.drop_p)]
            else:
                conv2d = nn.Conv2d(c, v, kernel_size=3, stride=1, padding=1)
                layers += [conv2d, act()]
                c = v

        self.features = nn.Sequential(*layers)

        if self.in_channels == 3:
            hidden = 128
            self.classifier = nn.Sequential(
                nn.Linear(c * 4 * 4, hidden),
                act(),
                nn.Dropout(self.drop_p),
                nn.Linear(hidden, self.num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(self.drop_p),
                nn.Linear(c * 4 * 4, self.num_classes)
            )

    def forward(self, x):
        if self.in_channels != 3:
            x = self.norm(x.view(-1, self.in_channels, 8, 8))
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x




class LRTorchNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LRTorchNet, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs

class LeNet(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super(LeNet, self).__init__()
        self.in_channels = in_channels
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
        )

        if in_channels == 1:
            fc_input_dim = 12 * 7 * 7
        elif in_channels == 3:
            fc_input_dim = 12 * 8 * 8
        else:
            raise ValueError(f"Unsupported in_channels: {in_channels}. Please specify 1 or 3.")

        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, num_classes)
        )

    def forward(self, x):
        out = self.body(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

# class MLP(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super().__init__()
#
#         self.input_fc = nn.Linear(input_dim, 3000)
#         self.output_fc = nn.Linear(3000, output_dim)
#
#     def forward(self, x):
#
#         batch_size = x.shape[0]
#         x = x.view(batch_size, -1)
#      #   h = torch.sigmoid(self.input_fc(x))
#         h = torch.relu(self.input_fc(x))
#
#         output = self.output_fc(h)
#
#         return output

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.input_fc = nn.Linear(input_dim, 3000)

        self.hidden_fc1 = nn.Linear(3000, 1000)

        self.hidden_fc2 = nn.Linear(1000, 512)
        self.hidden_fc3= nn.Linear(512, 64)

        self.output_fc = nn.Linear(64, output_dim)

    def forward(self, x):
        batch_size = x.shape[0]

        x = x.view(batch_size, -1)

        h = F.relu(self.input_fc(x))

        h = F.relu(self.hidden_fc1(h))
        h = F.relu(self.hidden_fc2(h))
        h = F.relu(self.hidden_fc3(h))

        output = self.output_fc(h)

        return output



# for defense to MInv by Variational Information Bottleneck,
class MLP_VIB(nn.Module):
    def __init__(self, input_dim, output_dim, latent_dim=64):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, 3000)
        self.hidden_fc1 = nn.Linear(3000, 1000)
        self.hidden_fc2 = nn.Linear(1000, 512)
        self.hidden_fc3 = nn.Linear(512, latent_dim * 2)
        self.classifier = nn.Linear(latent_dim, output_dim)
        self.output_fc = self.classifier
        self.latent_dim = latent_dim

    def encode(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        h = F.relu(self.input_fc(x))
        h = F.relu(self.hidden_fc1(h))
        h = F.relu(self.hidden_fc2(h))
        h = self.hidden_fc3(h)  # [batch, latent_dim*2]
        mu, std = h[:, :self.latent_dim], h[:, self.latent_dim:]
        std = F.softplus(std - 5, beta=1)
        return mu, std

    def forward(self, x):
        mu, std = self.encode(x)
        eps = torch.randn_like(std)
        z = mu + std * eps
        output = self.classifier(z)
        return output

    def get_mu_std(self, x):
        return self.encode(x)



class MLP_dropout(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.1):
        super().__init__()

        self.input_fc = nn.Linear(input_dim, 3000)
        self.input_dropout = nn.Dropout(dropout_rate)

        self.hidden_fc1 = nn.Linear(3000, 1000)
        self.hidden_dropout1 = nn.Dropout(dropout_rate)

        self.hidden_fc2 = nn.Linear(1000, 512)
        self.hidden_dropout2 = nn.Dropout(dropout_rate)
        
        self.hidden_fc3= nn.Linear(512, 64)
        self.hidden_dropout3 = nn.Dropout(dropout_rate)

        self.output_fc = nn.Linear(64, output_dim)
        self.output_dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        batch_size = x.shape[0]

        x = x.view(batch_size, -1)

        h = F.relu(self.input_fc(x))
        h = self.input_dropout(h)

        h = F.relu(self.hidden_fc1(h))
        h = self.hidden_dropout1(h)
        
        h = F.relu(self.hidden_fc2(h))
        h = self.hidden_dropout2(h)
        
        h = F.relu(self.hidden_fc3(h))
        h = self.hidden_dropout3(h)

        output = self.output_fc(h)
        output = self.output_dropout(output)

        return output

def replace_bn_with_gn(model, num_groups=32):
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            num_channels = module.num_features
            setattr(model, name, nn.GroupNorm(num_groups=min(num_groups, num_channels), num_channels=num_channels))
        else:
            replace_bn_with_gn(module, num_groups=num_groups)
