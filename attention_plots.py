""" Attention weights plot: Heads visualization and Attention rollout"""


def function():  # eventualmente per storico ... prende in ingresso i pckl, so l'iterazione, so l'epoca ...
    import torch
    import torchvision.transforms as transforms
    from torch.utils.data import random_split
    import pickle
    import torchvision
    import matplotlib.pyplot as plt
    import numpy as np

    # with open("./atn_epoch_0_it_5.pckl", "rb") as f:
    #     atn_list = pickle.load(f)
    #
    # atn_list = atn_list[0]  # errore già fixato
    # # 2 layers, 2 heads, 128 immagini, matrice degli atn weights 17*17
    #
    # # plot an attention matrix
    # atn_list_l0 = atn_list[0]  # first layer
    # # atn_list_l0 shape is: batch_size, heads, n_tokens, n_tokens
    # atn_matrix = atn_list_l0[2, 0, :, :]  # terza immagine, prima head
    # # plt.imshow(atn_matrix.detach().cpu().numpy())
    # # plt.show()

    # PLOT heads visualization # funziona
    # se voglio estrarre epoca e iterazione dal nome
    filename = './atn_epoch_0_it_5.pckl'
    f = open(filename, 'rb')
    atn_list = pickle.load(f)
    atn = atn_list[0]  # con i nuovi toglierlo
    spl = filename.split('/')[-1]  # atn_epoch_0_it_5.pckl
    pkl = spl.split('_')
    ita = spl.split('_')[-1]  # 5.pckl
    it = int(ita.split('.')[0])
    epoch = pkl[2]

    batch_index = 2
    batch_size = 128
    transform = transforms.Compose(
        [transforms.ToTensor()])

    dataset = torchvision.datasets.CIFAR100(root='./cifar100_data', train=True,
                                            download=False, transform=transform)

    val_size = round((30 / 100) * len(dataset))  # primo termine il valore % che vogliamo come validattion set
    train_size = len(dataset) - val_size

    train_set, validation_set = random_split(dataset, [train_size, val_size],
                                             generator=torch.Generator().manual_seed(1))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False)

    # labels
    with open("./cifar100_data/cifar-100-python/meta", "rb") as f:
        retrieved_data = pickle.load(f)

    fine_labels = retrieved_data['fine_label_names']

    atn_l0 = atn[0]  # layer 0 - batch size, heads, n_token, n_tokens
    bi = atn_l0[batch_index]  # heads, n_token, n_tokens

    # plt.imshow(bi[1].detach().cpu().numpy())  # una head
    # plt.show()

    with torch.no_grad():
        for val_it, val_batch in enumerate(validation_loader):
            val_images = val_batch[0]
            val_labels = val_batch[1]

            if val_it == it:

                # # Visualize attention matrix - heads
                fig = plt.figure(figsize=(16, 8))
                fig.suptitle("Visualization of Attention", fontsize=24)
                # fig.add_axes()

                chosen_image = val_images[batch_index]  # c, h, w
                chosen_image = chosen_image.permute(1, 2, 0)  # h, w, c
                img = np.asarray(chosen_image)
                ax = fig.add_subplot(2, 4, 1)
                ax.imshow(img)
                for i in range(atn_l0.size(1)):  # heads
                    attn_heatmap = bi[i, 0, 1:].reshape((4, 4)).detach().cpu().numpy()  # cls token
                    ax = fig.add_subplot(2, 4, i + 2)
                    ax.imshow(attn_heatmap)

                plt.show()
                break


# https://github.com/jacobgil/vit-explain/blob/main/vit_rollout.py
def rollout(attentions, batch_index, discard_ratio=0.0, head_fusion='mean'):
    import numpy as np
    import torch

    """ Attention rollout implementation.

    Parameters
    ----------
    attentions : list
            List of attentions weights. The dimension of attentions is the number of layers of the model under exam

    batch_index : int
            Index of the image in the batch tested

    discard_ratio : float
            % of weights to drop if filter is 1
    
    head_fusion : str
            Strategy of fusion for the heads. 'mean', 'min', 'max' are the possibilities
    """

    result = torch.eye(attentions[0].size(-1))  # n_tokens, n_tokens
    with torch.no_grad():
        for attention in attentions:  # ciclo sui layers
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise "Attention head fusion type Not supported"

            if discard_ratio > 0.0:
                print("Drop the lowest attentions")
                # Drop the lowest attentions, but don't drop the class token
                flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)  # b, n_tokens * n_tokens
                # prendo il 90% dei valori più piccoli di flat
                _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)
                # mi assicuro di non avere 0 (cls_token) tra gli indici che vogliamo togliere
                indices = indices[indices != 0]
                flat[batch_index, indices] = 0  # NB: si modifica anche attention_heads_fused (perchè ho usato view)

            I = torch.eye(attention_heads_fused.size(-1))
            a = attention_heads_fused.cpu() + I
            a = a / a.sum(dim=-1).unsqueeze(-1)

            result = torch.matmul(a, result)  # b, n_tokens, n_tokens

    # Look at the total attention between the class token, and the image patches
    mask = result[batch_index, 0, 1:]
    width = int(mask.size(-1) ** 0.5)
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    return mask


def heads_visualization(atn_wei_list, batch_index, chosen_image, save_path):
    """ Heads visualization for a single image.

    Parameters
    ----------
    atn_wei_list : list
            List of attentions weights. The dimension of attentions is the number of layers of the model under exam

    batch_index : int
            Index of the image in the batch tested

    chosen_image: Tensor
            Image on which plot attention weights (images[batch_index])

    save_path : str
            Path to the location you want to save the plot (it ends with the weight_epoch at which
            the model is recovered)
    """

    # with open("./cifar100_data/cifar-100-python/meta", "rb") as f:
    #     retrieved_data = pickle.load(f)
    # fine_labels = retrieved_data['fine_label_names']
    # lab = [fine_labels[labels[j]] for j in range(batch_size)]
    # imshow(torchvision.utils.make_grid(images), labels=lab)

    name = save_path + '_rollout_vis_' + str(batch_index) + '.png'

    # check if the file exists
    if not os.path.exists(name):
        print("heads saving")
        fig = plt.figure(figsize=(16, 8))
        fig.suptitle("Attention heads visualization", fontsize=24)
        chosen_image = chosen_image.permute(1, 2, 0)  # h, w, c
        img = np.asarray(chosen_image)
        ax = fig.add_subplot(2, 5, 1)  # 2 righe, 5 colonne
        ax.imshow(img)
        atn_l0 = atn_wei_list[0]  # layer 0
        bi = atn_l0[batch_index]

        # work on this
        # fig = plt.figure(figsize=(10, 10))
        # plt.imshow(bi[1].detach().cpu().numpy())  # una head
        # plt.show()

        for i in range(atn_l0.size(1)):  # heads
            attn_heatmap = bi[i, 0, 1:].reshape((8, 8)).detach().cpu().numpy()  # cls token
            ax = fig.add_subplot(2, 5, i + 2)
            ax.imshow(attn_heatmap)

        # save fig
        plt.savefig(name)
        # plt.show()


def rollout_visualization(atn_wei_list, batch_index, chosen_image, save_path, discard_ratio=0.9,
                          head_fusion='min'):
    import cv2
    import matplotlib.pyplot as plt

    """ Rollout visualization for a single image.

    Parameters
    ----------
    atn_wei_list : list
            List of attention weights (len(atn_wei_list) = number of layers of the model)
            
    batch_index : int
            Index of the image in the batch tested
    
    chosen_image: Tensor
            Image on which plot attention weights (images[batch_index]) 
    
    save_path : str
            Path to the location you want to save the plot (it ends with the weight_epoch at which 
            the model is recovered)

    discard_ratio : float
            % of weights to drop if filter is 1

    head_fusion : str
            Strategy of fusion for the heads. 'mean', 'min', 'max' are the possibilities
    """

    # with open("./cifar100_data/cifar-100-python/meta", "rb") as f:
    #     retrieved_data = pickle.load(f)
    # fine_labels = retrieved_data['fine_label_names']

    name = save_path + '_hf_' + head_fusion + '_heads_vis_' + str(batch_index) + '.png'
    if filter == 1:
        name = save_path + '_hf_' + head_fusion + '_dr' + str(discard_ratio) + '_heads_vis_' + str(batch_index) + '.png'

    # check if the file exists
    if not os.path.exists(name):
        print("rollout saving")
        mask = rollout(atn_wei_list, batch_index, discard_ratio=discard_ratio, head_fusion=head_fusion)
        # chosen_image = images[batch_index]  # c, h, w
        chosen_image = chosen_image.permute(1, 2, 0)  # h, w, c
        tup = (chosen_image.shape[0], chosen_image.shape[1])
        maski = cv2.resize(mask, tup)[..., np.newaxis]  # aggiungo dim canali  h, w, c  / mask.max()
        resulti = (maski * chosen_image.cpu().numpy())  # .astype("uint8")

        # lab = fine_labels[labels[batch_index]]
        # ims = torch.stack([chosen_image, torch.from_numpy(resulti)])
        # imshow(torchvision.utils.make_grid(ims.permute(0, 3, 1, 2)), 'attention rollout map')

        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(16, 16))
        ax1.set_title('Original')
        ax2.set_title('Attention Map')
        ax3.set_title('Mask resized')
        _ = ax1.imshow(chosen_image)
        _ = ax2.imshow(resulti)
        _ = ax3.imshow(maski)

        # save fig
        plt.savefig(name)
        # plt.show()


if __name__ == '__main__':
    # function()
    from utils import *
    from get_dataset_dataloaders import get_dataset_fine_tuning
    import json
    import matplotlib.pyplot as plt
    import pickle
    import numpy as np
    import os
    import argparse

    parser = argparse.ArgumentParser(description="Attention weights plot")

    parser.add_argument("--model_weights_path", dest="model_weights_path",
                        default='./model_weights/pretraining_imagenet'
                        , help="path to the folder where are stored the weights of the model in exam (no final slash)")
    parser.add_argument("--weight_epoch", dest="weight_epoch", default=1,
                        help="epoch at which we recover the model weights")
    parser.add_argument("--batch_index", dest="batch_index", default=0,
                        help="index of the image of the batch to test")
    parser.add_argument("--atn_plot_path", dest="atn_plot_path", default='./plots/atn_weights'
                        , help="path to the folder where the atn weights images are stored (no final slash)")
    parser.add_argument("--batch_size", dest="batch_size", default=4, help="size of the batch for test the model")
    parser.add_argument("--num_classes", dest="num_classes", default=100, help="number of classes of the dataset "
                        "(we need this info only for the fine tuning model)")
    parser.add_argument("--dataset", dest="dataset", default='./cifar100_data', help="dataset to test the model")
    parser.add_argument("--ft", dest="ft", default=0, help="1 for fine tuned model, 0 for pretrained model")

    # params for rollout
    parser.add_argument("--head_fusion", dest="head_fusion", default='mean', help="head fusion strategy "
                                                                                  "for attention rollout")
    parser.add_argument("--discard_ratio", dest="discard_ratio", default=0.9, help="% of atn weights to drop (float)")

    args = parser.parse_args()

    wp = args.model_weights_path
    we = int(args.weight_epoch)
    batch_index = int(args.batch_index)
    batch_size = int(args.batch_size)
    dataset = args.dataset

    if int(args.ft) == 1:
        print("fine tuned model")
        model, device = get_fine_tuned_model(weights_path=wp, weight_epoch=we, num_classes=int(args.num_classes))
    else:
        print("pretrained model")
        model, device = get_pretrained_model(weights_path=wp, weight_epoch=we)

    model.eval()

    with open(wp + '/hyperparams.json') as json_file:
        hyper_params = json.load(json_file)

    hyper_params['batch_size'] = batch_size
    train_loader, validation_loader = get_dataset_fine_tuning(ds=dataset, hyperparams=hyper_params)
    # mi serve questa perchè devo avere le immagini 64x64

    # get some random training images
    dataiter = iter(validation_loader)  # su train loader c'è rand aug ...
    images, labels = dataiter.next()

    out, atn_wei_list = model(images.to(device))
    # acc = (out.argmax(dim=-1) == labels.to(device)).float().mean()

    name_exp = wp.split('/')[-1]  # nome cartella dell'esperimento (da cui ho preso i pesi per il modello)
    atn_plots_path = os.path.join(args.atn_plot_path, name_exp)
    if not os.path.exists(atn_plots_path):
        os.makedirs(atn_plots_path)
    print("atn plot path: ", atn_plots_path)

    # Heads visualization
    heads_visualization(atn_wei_list, batch_index=batch_index, chosen_image=images[batch_index],
                        save_path=atn_plots_path + '/we_' + str(we))
    # heads_visualization(atn_wei_list, batch_index=2, save_path=atn_plots_path)
    # heads_visualization(atn_wei_list, batch_index=3, save_path=atn_plots_path)

    # Attention rollout
    rollout_visualization(atn_wei_list=atn_wei_list, batch_index=batch_index, chosen_image=images[batch_index],
                          save_path=atn_plots_path + '/we' + str(we), discard_ratio=float(args.discard_ratio),
                          head_fusion=args.head_fusion)

    # Ex: python attention_plots.py --we 91 --head_fusion min




# # ATTENTION ROLLOUT
# from utils import rollout
# import cv2
# import matplotlib.pyplot as plt
#
# batch_index = 0
# with open("./cifar100_data/cifar-100-python/meta", "rb") as f:
#     retrieved_data = pickle.load(f)
# fine_labels = retrieved_data['fine_label_names']
#
# # metodo 1
# mask = rollout(atn_wei_list, batch_index, discard_ratio=0.2, head_fusion='min')
# chosen_image = images[batch_index]  # c, h, w
# chosen_image = chosen_image.permute(1, 2, 0)  # h, w, c
# tup = (chosen_image.shape[0], chosen_image.shape[1])
# maski = cv2.resize(mask, tup)[..., np.newaxis]  # aggiungo dim canali  h, w, c  / mask.max()
# resulti = (maski * chosen_image.cpu().numpy())  # .astype("uint8")
# lab = fine_labels[labels[batch_index]]
#
# ims = torch.stack([chosen_image, torch.from_numpy(resulti)])
# # imshow(torchvision.utils.make_grid(ims.permute(0, 3, 1, 2)), 'attention rollout map')
# fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(16, 16))
# ax1.set_title('Original')
# ax2.set_title('Attention Map')
# ax3.set_title('Mask resized')
# _ = ax1.imshow(chosen_image)
# _ = ax2.imshow(resulti)
# _ = ax3.imshow(maski)
# plt.show()

# END attention rollout


# # HEADS VISUALIZATION
# with open("./cifar100_data/cifar-100-python/meta", "rb") as f:
#     retrieved_data = pickle.load(f)
# fine_labels = retrieved_data['fine_label_names']
# lab = [fine_labels[labels[j]] for j in range(batch_size)]
# # imshow(torchvision.utils.make_grid(images), labels=lab)
#
# batch_index = 0
# fig = plt.figure(figsize=(16, 8))
# fig.suptitle("Visualization of Attention", fontsize=24)
# im = images[batch_index]
# chosen_image = im.permute(1, 2, 0)  # h, w, c
# img = np.asarray(chosen_image)
# ax = fig.add_subplot(2, 5, 1)  # 2 righe, 5 colonne
# ax.imshow(img)
# atn_l0 = atn_wei_list[0]  # layer 0
# bi = atn_l0[batch_index]
#
# # work on this
# # fig = plt.figure(figsize=(10, 10))
# # plt.imshow(bi[1].detach().cpu().numpy())  # una head
# # plt.show()
#
# for i in range(atn_l0.size(1)):  # heads
#     attn_heatmap = bi[i, 0, 1:].reshape((8, 8)).detach().cpu().numpy()  # cls token
#     ax = fig.add_subplot(2, 5, i + 2)
#     ax.imshow(attn_heatmap)
#
# plt.show()


# visualizzazione proposta in https://github.com/jacobgil/vit-explain/blob/main/vit_rollout.py (non mi piace)
# from utils import show_mask_on_image
#
# np_img = np.array(chosen_image)[:, :, ::-1]
# mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
# mask = show_mask_on_image(np_img, mask)
# cv2.imshow("Input Image", np_img)
# cv2.imshow("masked", mask)
# # cv2.imwrite("input.png", np_img)
# # cv2.imwrite(name, mask)
# cv2.waitKey(-1)


# mi dà una maschera tutta nera ...
# # metodo 2 - funziona solo per batch = 1 https://github.com/jeonsworld/ViT-pytorch/blob/main/visualize_attention_map.ipynb -

# def show_mask_on_image(img, mask):
#     import numpy as np
#     import cv2
#     img = np.float32(img) / 255
#     heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
#     heatmap = np.float32(heatmap) / 255
#     cam = heatmap + np.float32(img)
#     cam = cam / np.max(cam)
#
#     return np.uint8(255 * cam)

# import matplotlib.pyplot as plt
# import cv2
#
# batch_index = 2
# chosen_image = images[batch_index]  # c, h, w
# chosen_image = chosen_image.permute(1, 2, 0)  # h, w, c
# _, atn_wei_list_im = model(images[batch_index].unsqueeze(0).to(device))
# att_mat = torch.stack(atn_wei_list_im).squeeze(1)  # layers, heads, n_tokens, n_tokens
#
# # Average the attention weights across all heads.
# att_mat = torch.mean(att_mat, dim=1)  # n_layers, n_tokens, n_tokens
#
# # To account for residual connections, we add an identity matrix to the
# # attention matrix and re-normalize the weights.
# residual_att = torch.eye(att_mat.size(1))  # n_tokens, n_tokens
# aug_att_mat = att_mat + residual_att.to(device)
# aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)
#
# # Recursively multiply the weight matrices
# joint_attentions = torch.zeros(aug_att_mat.size()).to(device)
# joint_attentions[0] = aug_att_mat[0]
#
# for n in range(1, aug_att_mat.size(0)):
#     joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])
#
# # Attention from the output token to the input space.
# v = joint_attentions[-1]  # mappa ultimo layer
# grid_size = int(np.sqrt(aug_att_mat.size(-1)))
# mask = v[0, 1:].reshape(grid_size, grid_size).cpu().detach().numpy()  # cls_token - altri token (ma non cls) similarità
# tup = (chosen_image.shape[0], chosen_image.shape[1])
# mask = cv2.resize(mask / mask.max(), tup)[..., np.newaxis]  # aggiungo dim canali
# result = (mask * chosen_image.cpu().detach().numpy()).astype("uint8")
#
# fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
# ax1.set_title('Original')
# ax2.set_title('Attention Map')
# _ = ax1.imshow(chosen_image)
# _ = ax2.imshow(result)
# plt.show()

# embedding filters (non mi funziona)
# fig = plt.figure(figsize=(4, 4))
# f = filter[0:28]  # 28, 3, 8, 8
# col = 7
# rows = f.shape[0] // col + 1
# from torch.nn.functional import normalize
# import torchvision.transforms as transforms
# tr = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# for i in range(f.shape[0]):
#     el = f[i]  # .permute(1, 2, 0)  # patch_size, patch_size, c
#     ax = fig.add_subplot(4, 7, i+1)
#     ax.axes.get_xaxis().set_visible(False)
#     ax.axes.get_yaxis().set_visible(False)
#
#     el = el / torch.norm(el)  # normalize(el, p=1.0)
#
#     # el = tr(el)  # -1, 1
#     # image = ((el * 0.5) + 0.5)  # 0, 1
#     ax.imshow(el.permute(1, 2, 0).numpy())
# plt.axis('off')
# plt.show()