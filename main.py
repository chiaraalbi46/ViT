from comet_ml import Experiment
import os.path
import torch
from get_dataset_dataloaders import get_dataset
import argparse
import pickle
import numpy as np
from vit import ViT

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train ViT")

    parser.add_argument("--epochs", dest="epochs", default=1, help="number of epochs")
    parser.add_argument("--plot_step", dest="plot_step", default=2,
                        help="number of attention weights savings during train")
    parser.add_argument("--batch_size", dest="batch_size", default=250, help="Batch size")
    parser.add_argument("--lr", dest="lr", default=0.001, help="learning rate train")
    parser.add_argument("--weight_decay", dest="weight_decay", default=0., help="weight decay")
    parser.add_argument("--scheduling", dest="scheduling", default=0,
                        help="1 if scheduling lr policy applied, 0 otherwise")
    parser.add_argument("--drop", dest="drop", default=0.0, help="dropout value train")
    parser.add_argument("--val_perc", dest="val_perc", default=0, help="% validation set")

    # rand augmentation
    parser.add_argument("--rand_aug_numops", dest="rand_aug_numops", default=None,
                        help="number of augmentation transformations to apply sequentially")
    parser.add_argument("--rand_aug_magn", dest="rand_aug_magn", default=None,
                        help="magnitude for all the transformations")

    # iperparametri ViT
    parser.add_argument("--img_size", dest="img_size", default=32, help="size of the images (int))")
    parser.add_argument("--patch_size", dest="patch_size", default=8, help="patch dimension (int)")
    parser.add_argument("--embed_dim", dest="embed_dim", default=384, help="embedding dimension")
    parser.add_argument("--hidden_dim", dest="hidden_dim", default=384,
                        help="hidden dimension for MLP in Tranformer Block")
    parser.add_argument("--num_heads", dest="num_heads", default=2, help="number of heads of mha layers")
    parser.add_argument("--num_layers", dest="num_layers", default=7,
                        help="number of layers (blocks) of transformer encoder")
    parser.add_argument("--num_classes", dest="num_classes", default=100, help="number of classes of the dataset")

    parser.add_argument("--device", dest="device", default='0', help="choose GPU")
    parser.add_argument("--name_proj", dest="name_proj", default='VIT', help="define comet ml project folder")
    parser.add_argument("--name_exp", dest="name_exp", default='None', help="define comet ml experiment")
    parser.add_argument("--comments", dest="comments", default=None, help="comments (str) about the experiment")

    # parser.add_argument("--atn_path", dest="atn_path", default=None,
    #                     help="path to the folder where storing the attention weights")
    parser.add_argument("--weights_path", dest="weights_path", default=None,
                        help="path to the folder where storing the model weights")
    parser.add_argument("--dataset", dest="dataset", default='../Vit_vs_mlp_mixer/datasets/cifar100',
                        help="path to the dataset folder (where train, test, meta are located)")  # './cifar100_data'

    args = parser.parse_args()

    # Iperparametri
    batch_size = int(args.batch_size)
    num_epochs = int(args.epochs)
    plot_step = int(args.plot_step)
    num_plots = int(num_epochs / plot_step)
    lr = float(args.lr)
    wd = float(args.weight_decay)
    sched = int(args.scheduling)
    dropout_value = float(args.drop)
    numops = args.rand_aug_numops if args.rand_aug_numops is None else int(args.rand_aug_numops)
    magn = args.rand_aug_magn if args.rand_aug_magn is None else int(args.rand_aug_magn)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device: ", device)

    hyper_params = {
        "img_size": int(args.img_size),
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": lr,
        "weight_decay": wd,
        "scheduling": sched,
        "dropout_value": dropout_value,
        "patch_size": int(args.patch_size),
        "num_layers": int(args.num_layers),
        "embed_dim": int(args.embed_dim),
        "hidden_dim": int(args.hidden_dim),
        "num_heads": int(args.num_heads),
        "rand_aug_numops": numops,
        "rand_aug_magn": magn
    }

    # comet ml integration
    experiment = Experiment(project_name=args.name_proj)
    experiment.set_name(args.name_exp)

    # Definizione modello
    vit = ViT(img_size=int(args.img_size), embed_dim=hyper_params['embed_dim'], num_channels=3,
              num_heads=hyper_params['num_heads'], num_layers=hyper_params['num_layers'],
              num_classes=int(args.num_classes), patch_size=hyper_params['patch_size'],
              hidden_dim=hyper_params['hidden_dim'], dropout_value=hyper_params['dropout_value'])

    experiment.log_parameters(hyper_params)
    experiment.set_model_graph(vit)

    save_weights_path = os.path.join(args.weights_path, args.name_exp)
    if not os.path.exists(save_weights_path):
        os.makedirs(save_weights_path)
    print("save weights: ", save_weights_path)

    # atn_weights_path = os.path.join(args.atn_path, args.name_exp)
    # if not os.path.exists(atn_weights_path):
    #     os.makedirs(atn_weights_path)
    # print("atn weights: ", atn_weights_path)

    # Dataset, dataloaders
    train_loader, validation_loader = get_dataset(ds=args.dataset, hyperparams=hyper_params,
                                                  val_perc=int(args.val_perc))

    # Definizione ottimizzatore
    optimizer = torch.optim.Adam(vit.parameters(), lr=lr, weight_decay=wd)

    if sched == 1:
        print("Scheduling of learning rate applied")
        # Definizione scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=lr, pct_start=0.05,
                                                        total_steps=len(train_loader) * num_epochs,
                                                        anneal_strategy='linear')
        # anche pct_start dovrebbe essere iperparametro ...

    # Definizione loss
    loss = torch.nn.CrossEntropyLoss()

    # Model to GPU
    vit = vit.to(device)

    # Conteggio parametri
    model_parameters = filter(lambda p: p.requires_grad, vit.parameters())  # [x for x in net.parameters()]
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Number of parameters: ", params)

    experiment.log_other('num_parameters', params)
    if args.comments:
        experiment.log_other('comments', args.comments)

    print("Start training loop")
    atn_save = False

    for epoch in range(num_epochs):
        vit.train()  # Sets the module in training mode

        # if epoch == 0 or (epoch + 1) % num_plots == 0:
        #     print("Save the attention weights of the first batch")
        #     atn_save = True

        # batch training
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        for it, train_batch in enumerate(train_loader):
            train_images = train_batch[0].to(device)
            train_labels = train_batch[1].to(device)

            optimizer.zero_grad()

            # prediction
            out, atn_weights_list = vit(train_images)  # (batch_size, num_classes)

            # if it == 0 and atn_save:
            #     print("Saving ")
            #     f = open(os.path.join(atn_weights_path, 'atn_epoch_' + str(epoch) + '_it_0' + '.pckl'), 'wb')
            #     pickle.dump([atn_weights_list], f)
            #     f.close()
            #     atn_save = False

            # compute loss
            train_loss = loss(out, train_labels)  # batch loss
            train_losses.append(train_loss.item())

            # compute accuracy
            train_acc = (out.argmax(dim=-1) == train_labels).float().mean()
            train_accuracies.append(train_acc.item())

            train_loss.backward()

            # update weights
            optimizer.step()

            rl = optimizer.param_groups[0]["lr"]
            # experiment.log_metric('learning_rate_batch_sched', rl, step=it)

            if sched == 1:
                # rl = scheduler.get_last_lr()  # anche optimizer.param_groups[0]["lr"] va bene
                # # experiment.log_metric('learning_rate_batch_sched', rl, step=it)
                scheduler.step()

        if sched == 1:
            experiment.log_metric('learning_rate_epoch', rl, step=epoch + 1)  # the last batch learning rate

        # Validation step
        print()

        vit.eval()  # Sets the module in evaluation mode (validation/test)
        with torch.no_grad():
            for val_it, val_batch in enumerate(validation_loader):
                val_images = val_batch[0].to(device)
                val_labels = val_batch[1].to(device)

                val_out, _ = vit(val_images)

                val_loss = loss(val_out, val_labels)
                val_losses.append(val_loss.item())

                val_acc = (val_out.argmax(dim=-1) == val_labels).float().mean()
                val_accuracies.append(val_acc.item())

        # comet ml
        experiment.log_metric('train_epoch_loss', sum(train_losses) / len(train_losses), step=epoch + 1)
        experiment.log_metric('train_epoch_acc', sum(train_accuracies) / len(train_accuracies), step=epoch + 1)
        experiment.log_metric('valid_epoch_loss', sum(val_losses) / len(val_losses), step=epoch + 1)
        experiment.log_metric('valid_epoch_acc', sum(val_accuracies) / len(val_accuracies), step=epoch + 1)

        # print("End valid test")
        print("Epoch [{}], Train loss: {:.4f}, Validation loss: {:.4f}".format(
            epoch + 1, sum(train_losses) / len(train_losses), sum(val_losses) / len(val_losses)))

        # Save weights
        if epoch % 10 == 0:
            torch.save(vit.state_dict(), save_weights_path + '/weights_' + str(epoch + 1) + '.pth')

    torch.save(vit.state_dict(), save_weights_path + f"final.pth")

    experiment.end()
    print("End training loop")

    # Ex: python main.py --epochs 12 --dataset  C:\Users\chiar\PycharmProjects\ViT\cifar100_data --num_heads 12
    # --hidden_dim 384 --batch_size 128 --num_classes 100 --name_exp cifar100_tanh
    # --comments 'mlp head with tanh (no layer norm)'

