import os
import torch
from termcolor import colored
from configs import parser
from utils.engine_retrieval import train2, test2
from model.retrieval.model_main import ModelFusion
from loaders.loader import loader_generation
from utils.tools import fix_parameter, print_param


os.makedirs('saved_model/', exist_ok=True)


def main():
    model = ModelFusion(args, vis=False)
    device = torch.device(args.device)

    # CUDNN
    torch.backends.cudnn.benchmark = True

    print(colored('trainable parameter name: ', "blue"))
    print_param(model)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    model.to(device)

    train_loader1, train_loader2, val_loader = loader_generation(args)
    best_acc = 0

    for i in range(args.epoch):
        print(colored('Epoch %d/%d' % (i + 1, args.epoch), 'yellow'))
        print(colored('-' * 15, 'yellow'))

        # Adjust lr
        if i == args.lr_drop:
            print("Adjusted learning rate to 1/10")
            optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"] * 0.1
        train2(args, model, device, train_loader1, optimizer, i)

        if i % args.fre == 0:
            acc = test2(args, model, val_loader, device)

            if acc > best_acc:
                print("higher accuracy, saving current model parameter")
                best_acc = acc
                torch.save(model.state_dict(), os.path.join(args.output_dir,
                    f"{'Fusion' if not args.fusion else 'No_fusion'}_{args.base_model}_cls{args.num_classes}_" + f"cpt{args.num_cpt if not args.pre_train else ''}_" +
                    f"{'use_slot_' + args.cpt_activation if not args.pre_train else 'no_slot'}.pt"))


if __name__ == '__main__':
    args = parser.parse_args()
    args.pre_train = False
    args.fusion = True
    main()
