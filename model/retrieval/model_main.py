from timm.models import create_model
import torch
import torch.nn.functional as F
import torch.nn as nn
import os
from utils.tools import fix_parameter, print_param
from model.retrieval.slots import ScouterAttention
from model.retrieval.position_encode import build_position_encoding


class Identical(nn.Module):
    def __init__(self):
        super(Identical, self).__init__()

    def forward(self, x):
        return x


def load_backbone(args):
    bone = create_model(args.base_model, pretrained=True,
                        num_classes=args.num_classes)

    bone.global_pool = Identical()
    bone.fc = Identical()
    return bone


class MainModel(nn.Module):
    def __init__(self, args, vis=False):
        super(MainModel, self).__init__()
        self.args = args
        self.pre_train = args.pre_train
        if "18" not in args.base_model:
            self.num_features = 2048
        else:
            self.num_features = 512
        self.feature_size = args.feature_size
        self.drop_rate = 0.0
        hidden_dim = 128
        num_concepts = args.num_cpt
        num_classes = args.num_classes
        self.back_bone = load_backbone(args)
        self.activation = nn.Tanh()
        self.vis = vis

        if not self.pre_train:
            self.conv1x1 = nn.Conv2d(self.num_features, hidden_dim, kernel_size=(1, 1), stride=(1, 1))
            self.norm = nn.BatchNorm2d(hidden_dim)
            self.position_emb = build_position_encoding('sine', hidden_dim=hidden_dim)
            self.slots = ScouterAttention(args, hidden_dim, num_concepts, vis=self.vis)
            self.scale = 1
            self.cls = torch.nn.Linear(num_concepts, num_classes, bias=False)
        else:
            self.fc = nn.Linear(self.num_features, args.num_classes)
            self.drop_rate = 0

    def forward(self, x, weight=None, things=None, check=None):
        x = self.back_bone(x)
        # x = x.view(x.size(0), self.num_features, self.feature_size, self.feature_size)
        if not self.pre_train:
            x = self.conv1x1(x)
            x = self.norm(x)
            x = torch.relu(x)
            pe = self.position_emb(x)
            x_pe = x + pe

            b, n, r, c = x.shape
            x = x.reshape((b, n, -1)).permute((0, 2, 1))
            x_pe = x_pe.reshape((b, n, -1)).permute((0, 2, 1))
            if self.vis:
                updates, attn, normed = self.slots(x_pe, x, weight, things, check)
            else:
                updates, attn = self.slots(x_pe, x, weight, things, check)

            if self.args.cpt_activation == "att":
                cpt_activation = attn
            else:
                cpt_activation = updates
            attn_cls = self.scale * torch.sum(cpt_activation, dim=-1)
            cpt = self.activation(attn_cls)
            if not self.args.fusion_loader:
                cls = self.cls(cpt)

            if self.vis:
                return (cpt - 0.5) * 2, cls, attn, updates, normed
            else:
                if self.args.fusion_loader:
                    return cpt
                else:
                    return (cpt - 0.5) * 2, cls, attn, updates
        else:
            x = F.adaptive_max_pool2d(x, 1).squeeze(-1).squeeze(-1)
            if self.drop_rate > 0:
                x = F.dropout(x, p=self.drop_rate, training=self.training)
            x = self.fc(x)
            return x


class ModelFusion(nn.Module):
    def __init__(self, args):
        super(ModelFusion, self).__init__()
        num_concepts = args.num_cpt
        num_classes = args.num_classes
        self.args = args
        args.pre_train = False
        model_base = MainModel(args)
        checkpoint = torch.load(os.path.join(args.output_dir,
                                             f"{args.base_model}_cls{args.num_classes}_cpt{args.num_cpt}_" +
                                             f"{'use_slot_' + args.cpt_activation if not args.pre_train else 'no_slot'}.pt"),
                                map_location="cpu")
        model_base.load_state_dict(checkpoint, strict=True)
        # fix_parameter(model_base, ["PPPP"], mode="open")
        print("load pre-trained model finished, start training")
        model_base.cls = Identical()
        self.cpt_g = model_base

        if self.args.fusion:
            self.fc1 = nn.Sequential(
                torch.nn.Linear(2, 2, bias=True),
                torch.nn.ReLU(),
                torch.nn.Tanh()
            )

            self.cls = torch.nn.Linear(num_concepts + 2, 2, bias=False)
        else:
            self.cls = torch.nn.Linear(num_concepts, 2, bias=False)

    def forward(self, x, csv_):
        cpt = self.cpt_g(x)
        if self.args.fusion:
            csv_0 = self.fc1(csv_)
            cat = torch.cat([cpt, csv_0], dim=-1)
            cls = self.cls(cat)
        else:
            cls = self.cls(cpt)

        return cls


# if __name__ == '__main__':
#     model = MainModel()
#     inp = torch.rand((2, 1, 224, 224))
#     pred, out, att_loss = model(inp)
#     print(pred.shape)
#     print(out.shape)


