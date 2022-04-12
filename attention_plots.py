
# todo: conviene salvare il batch delle immagini di cui salvo gli attention weights. nel training il problema è che per
#  una stessa iterazione fissata non ritroverò lo stesso batch... quindi non posso tracciare l'andamento (cioè vedere
#  come cambiano in pesi nel corso delle epoche su una stessa immagine) ... questa cosa la posso fare quando testo il
#  modello / quando faccio validation perchè in quel caso trovo lo stesso batch (shuffle=False)

# todo: mettere argparse o comunque gestirla meglio ... quando ho cifar10

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
def rollout(attentions, batch_index, filter=0, discard_ratio=0.9, head_fusion='mean'):
    import numpy as np
    import torch

    """ Attention rollout implementation.

    Parameters
    ----------
    attentions : list
            List of attentions weights. The dimension of attentions is the number of layers of the model under exam

    batch_index : int
            Index of the image in the batch tested

    filter : int
            1 if you want to drop the lowest attention weights, 0 otherwise

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

            if filter == 1:
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


def heads_visualization(atn_wei_list, batch_index, save_path):
    """ Heads visualization for a single image.

    Parameters
    ----------
    atn_wei_list : list
            List of attentions weights. The dimension of attentions is the number of layers of the model under exam

    batch_index : int
            Index of the image in the batch tested

    save_path : str
            Path to the location you want to save the plot
    """

    # with open("./cifar100_data/cifar-100-python/meta", "rb") as f:
    #     retrieved_data = pickle.load(f)
    # fine_labels = retrieved_data['fine_label_names']
    # lab = [fine_labels[labels[j]] for j in range(batch_size)]
    # imshow(torchvision.utils.make_grid(images), labels=lab)

    fig = plt.figure(figsize=(16, 8))
    fig.suptitle("Attention heads visualization", fontsize=24)
    im = images[batch_index]
    chosen_image = im.permute(1, 2, 0)  # h, w, c
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
    name = save_path + '/rollout_vis_' + str(batch_index) + '.png'
    plt.savefig(name)

    # plt.show()


def rollout_visualization(batch_index, save_path, filter=0, discard_ratio=0.9, head_fusion='min'):
    import cv2
    import matplotlib.pyplot as plt

    """ Rollout visualization for a single image.

    Parameters
    ----------
    batch_index : int
            Index of the image in the batch tested
    
    save_path : str
            Path to the location you want to save the plot

    filter : int
            1 if you want to drop the lowest attention weights, 0 otherwise

    discard_ratio : float
            % of weights to drop if filter is 1

    head_fusion : str
            Strategy of fusion for the heads. 'mean', 'min', 'max' are the possibilities
    """

    # with open("./cifar100_data/cifar-100-python/meta", "rb") as f:
    #     retrieved_data = pickle.load(f)
    # fine_labels = retrieved_data['fine_label_names']

    mask = rollout(atn_wei_list, batch_index, filter=filter, discard_ratio=discard_ratio, head_fusion=head_fusion)
    chosen_image = images[batch_index]  # c, h, w
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
    name = save_path + '/heads_vis_' + str(batch_index) + '.png'  # potrei aggiungere l'epoca ...
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

    # # pretrained model
    # wp = './model_weights/pretraining_imagenet'  # no slash
    # we = 261

    # fine tuned model
    wp = './fine_tuning_vit_cifar100'  # weights_path
    we = 71  # weight_epoch

    dataset = './cifar100_data'
    with open('./fine_tuning_vit_cifar100/hyperparams.json') as json_file:
        hyper_params = json.load(json_file)
    batch_size = 4
    num_classes = 100

    # model, device = get_pretrained_model(weights_path=wp, weight_epoch=we, fine_tuning=0)
    model, device = get_fine_tuned_model(weights_path=wp, weight_epoch=we, num_classes=num_classes)
    model.eval()

    hyper_params['batch_size'] = batch_size
    train_loader, validation_loader = get_dataset_fine_tuning(ds=dataset, hyperparams=hyper_params)
    # mi serve questa perchè devo avere le immagini 64x64

    # get some random training images
    dataiter = iter(validation_loader)  # su train loader c'è rand aug ...
    images, labels = dataiter.next()

    out, atn_wei_list = model(images.to(device))
    # acc = (out.argmax(dim=-1) == labels.to(device)).float().mean()

    name_exp = wp.split('/')[-1]  # nome cartella dell'esperimento (da cui ho preso i pesi per il modello)
    atn_plots_path = os.path.join('./atn_plots', name_exp)
    if not os.path.exists(atn_plots_path):
        os.makedirs(atn_plots_path)
    print("atn plots path: ", atn_plots_path)

    # Heads visualization
    heads_visualization(atn_wei_list, batch_index=0, save_path=atn_plots_path)
    # heads_visualization(atn_wei_list, batch_index=2, save_path=atn_plots_path)
    # heads_visualization(atn_wei_list, batch_index=3, save_path=atn_plots_path)

    # Attention rollout
    rollout_visualization(batch_index=0, save_path=atn_plots_path, filter=0, discard_ratio=0.9, head_fusion='min')























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










