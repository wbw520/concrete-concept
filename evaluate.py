import os
import torch
from configs import parser
import torch.nn.functional as F
from model.retrieval.model_main import MainModel
from loaders.loader import loader_generation, MakeListImage
from utils.cal_tools import matrixs


@torch.no_grad()
def evaluation(args, model, loader, device):
    model.eval()
    preds_record = []
    labels_record = []

    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device, dtype=torch.float32), label.to(device, dtype=torch.int64)
        if not args.pre_train:
            cpt, pred, att, update = model(data)
        else:
            pred = F.log_softmax(model(data), dim=-1)

        preds_record.append(pred)
        labels_record.append(label)

    pred = torch.cat(preds_record, dim=0)
    labels = torch.cat(labels_record, dim=0)
    pred = pred.argmax(dim=-1)

    return pred.cpu().detach().numpy(), labels.cpu().detach().numpy()


def main():
    train_loader1, train_loader2, val_loader = loader_generation(args)

    # load model and weights
    model = MainModel(args, vis=False)
    device = torch.device("cuda:0")
    model.to(device)
    checkpoint = torch.load(os.path.join(args.output_dir, f"{args.base_model}_cls{args.num_classes}_" + f"cpt{args.num_cpt if not args.pre_train else ''}_" +
                f"{'use_slot_' + args.cpt_activation if not args.pre_train else 'no_slot'}.pt"),
                            map_location="cuda:0")
    model.load_state_dict(checkpoint, strict=True)

    preds, labels = evaluation(args, model, val_loader, device)
    matrixs(preds, labels, args.base_model + " Per-class Normalized", ["0", "1", "2", "3"])


if __name__ == '__main__':
    args = parser.parse_args()
    args.pre_train = True
    main()