""" Test a model. Fine tuning is also possible. """

from comet_ml import Experiment
import torch
import json
from vit import ViT
from get_dataset_dataloaders import *
import argparse
import numpy as np
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine tuning")

    parser.add_argument("--weight_epoch", dest="weight_epoch", default=151,
                        help="epoch at which to recover model weights")
    parser.add_argument("--exp", dest="exp", default='pretraining_imagenet',
                        help="name of the folder of the experiment (the same on comet)")
    parser.add_argument("--num_classes", dest="num_classes", default=100,
                        help="number of classes of the tested dataset")
    parser.add_argument("--dataset", dest="dataset", default='./cifar100_data',
                        help="path to the dataset folder (where train, test, meta are located) we want to "
                             "test (fine tune) the pretrained model on")

    # comet parameters
    parser.add_argument("--name_proj", dest="name_proj", default='VIT', help="define comet ml project folder")
    parser.add_argument("--name_exp", dest="name_exp", default='None', help="define comet ml experiment")
    parser.add_argument("--comments", dest="comments", default=None, help="comments (str) about the experiment")
    # parser.add_argument("--atn_path", dest="atn_path", default=None,
    #                     help="path to the folder where storing the attention weights")

    # parametri fine tuning
    parser.add_argument("--epochs", dest="epochs", default=1, help="number of epochs")
    # parser.add_argument("--plot_step", dest="plot_step", default=2,
    #                     help="number of attention weights savings during train")
    parser.add_argument("--batch_size", dest="batch_size", default=250, help="Batch size")
    parser.add_argument("--lr", dest="lr", default=0.001, help="learning rate train")
    parser.add_argument("--weight_decay", dest="weight_decay", default=0.1, help="weight decay")
    parser.add_argument("--scheduling", dest="scheduling", default=0,
                        help="1 if scheduling lr policy applied, 0 otherwise")

    args = parser.parse_args()

    # path al file contenente i pesi del modello
    # assumi che il dizionario con gli iperparametri venga salvato nella cartella in cui vengono salvati i pesi
    # dell'esperim (weights_path/nome_exp/...)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    weights_path = './model_weights/' + args.exp
    print("wei path: ", weights_path)
    print(weights_path + '/weights_' + str(args.weight_epoch) + '.pth')

    with open(weights_path + '/hyperparams.json') as json_file:
        hyper_params = json.load(json_file)

    save_weights_path = './model_weights/' + args.name_exp
    if not os.path.exists(save_weights_path):
        os.makedirs(save_weights_path)
    print("save weights: ", save_weights_path)

    batch_size = int(args.batch_size)
    num_epochs = int(args.epochs)
    lr = float(args.lr)
    wd = float(args.weight_decay)
    sched = int(args.scheduling)

    hyper_params_ft = {
        "img_size": hyper_params['img_size'],
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": lr,
        "weight_decay": wd,
        "scheduling": sched,
        "patch_size": hyper_params['patch_size'],
        "num_layers": hyper_params['num_layers'],
        "embed_dim": hyper_params['embed_dim'],
        "hidden_dim": hyper_params['hidden_dim'],
        "num_heads": hyper_params['num_heads'],
        "rand_aug_numops": hyper_params['rand_aug_numops'],  # todo: mettere come parametri in input
        "rand_aug_magn": hyper_params['rand_aug_magn']
    }

    # Definizione modello
    model = ViT(img_size=hyper_params['img_size'], embed_dim=hyper_params['embed_dim'], num_channels=3,
                num_heads=hyper_params['num_heads'], num_layers=hyper_params['num_layers'],
                num_classes=hyper_params['num_classes'], patch_size=hyper_params['patch_size'],
                hidden_dim=hyper_params['hidden_dim'], dropout_value=0.0)  # no dropout on fine tuning

    # print(model)

    if device == 'cpu':
        model.load_state_dict(torch.load(weights_path + '/weights_' + str(args.weight_epoch) + '.pth',
                                         map_location='cpu'))
    else:
        model.load_state_dict(torch.load(weights_path + '/weights_' + str(args.weight_epoch) + '.pth'))

    model.eval()

    train_loader, validation_loader = get_dataset_fine_tuning(ds=args.dataset, hyperparams=hyper_params_ft)
    num_in_features = model.embed_dim

    # add new learnable linear layer
    model.mlp_head = torch.nn.Linear(num_in_features, int(args.num_classes))
    model.to(device)
    # print(model)

    # training loop

    # comet ml integration
    experiment = Experiment(project_name=args.name_proj)
    experiment.set_name(args.name_exp)
    experiment.log_parameters(hyper_params_ft)  # gli iperparameteri di finetuning
    # experiment.set_model_graph(model)

    # Definizione ottimizzatore
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    if sched == 1:
        print("Scheduling of learning rate applied")
        # Definizione scheduler
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=lr, pct_start=0.05,
        #                                                 total_steps=len(train_loader) * num_epochs,
        #                                                 anneal_strategy='linear')

        # cosine for fine tuning
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=lr, pct_start=0.05,
                                                        total_steps=len(train_loader) * num_epochs,
                                                        anneal_strategy='cosine')
        # anche pct_start dovrebbe essere iperparametro ...

    # Definizione loss
    loss = torch.nn.CrossEntropyLoss()

    # Conteggio parametri
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())  # [x for x in net.parameters()]
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Number of parameters: ", params)

    experiment.log_other('num_parameters', params)
    if args.comments:
        experiment.log_other('comments', args.comments)

    print("Start training (fine tuning) loop")
    for epoch in range(num_epochs):
        model.train()  # Sets the module in training mode

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
            out, atn_weights_list = model(train_images)  # (batch_size, num_classes)

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

            if sched == 1:
                # rl = scheduler.get_last_lr()  # anche optimizer.param_groups[0]["lr"] va bene
                # # experiment.log_metric('learning_rate_batch_sched', rl, step=it)
                scheduler.step()

        experiment.log_metric('learning_rate_epoch', rl, step=epoch + 1)  # the last batch learning rate

        # Validation step
        print()

        model.eval()  # Sets the module in evaluation mode (validation/test)
        with torch.no_grad():
            for val_it, val_batch in enumerate(validation_loader):
                val_images = val_batch[0].to(device)
                val_labels = val_batch[1].to(device)

                val_out, _ = model(val_images)

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
            torch.save(model.state_dict(), save_weights_path + '/weights_' + str(epoch + 1) + '.pth')

    torch.save(model.state_dict(), save_weights_path + f"final.pth")

    experiment.end()
    print("End training loop")

    # Ex: python fine_tune.py --weight_epoch 151 --dataset C:\Users\chiar\PycharmProjects\ViT\cifar100_data --num_classes 100
    # --exp pretraining_imagenet --comments 'fine tuning on CIFAR 100' --name_exp ft_cifar100 --batch_size 250
    # --lr 0.001 --weight_decay 0.0001 --epochs 200
