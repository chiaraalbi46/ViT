""" Attention weights plot: Heads visualization and Attention rollout"""


# The implementations are a combination of these links:
# https://github.com/jacobgil/vit-explain/blob/main/vit_rollout.py
# https://github.com/jacobgil/vit-explain/blob/main/vit_explain.py
# https://github.com/jeonsworld/ViT-pytorch/blob/main/visualize_attention_map.ipynb
# https://colab.research.google.com/github/hirotomusiker/schwert_colab_data_storage/blob/master/notebook/Vision_Transformer_Tutorial.ipynb#scrollTo=txIf-L2Fwoxy


def rollout(atn_wei_list_im, discard_ratio=0.0, head_fusion='mean'):
    import torch

    """ Attention rollout implementation.

    Parameters
    ----------
    attentions : list
            List of attentions weights. The dimension of attentions is the number of layers of the model under exam

    discard_ratio : float
            % of weights to drop

    head_fusion : str
            Strategy of fusion for the heads. 'mean', 'min', 'max' are the possibilities
    """

    att_mat = torch.stack(atn_wei_list_im).squeeze(1)  # layers, heads, n_tokens, n_tokens

    # ADDED
    if head_fusion == "mean":
        # Average the attention weights across all heads. (head fusione)
        att_mat = torch.mean(att_mat, dim=1)  # n_layers, n_tokens, n_tokens
    elif head_fusion == "max":
        att_mat = att_mat.max(axis=1)[0]
    elif head_fusion == "min":
        att_mat = att_mat.min(axis=1)[0]
    else:
        raise "Attention head fusion type Not supported"
    #

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1))  # n_tokens, n_tokens
    aug_att_mat = att_mat + residual_att.to(device)
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size()).to(device)  # layers, n_tokens n_tokens
    # ADDED
    if discard_ratio > 0.0:
        # drop lowest attentions
        l0 = aug_att_mat[0].reshape(-1)  # flatten 65*65
        # prendo il 90% dei valori più piccoli di flat
        _, indices = l0.topk(int(l0.size(-1) * discard_ratio), -1, False)
        # mi assicuro di non avere 0 (cls_token) tra gli indici che vogliamo togliere
        indices = indices[indices != 0]
        l0[indices] = 0
        aug_att_mat[0] = l0.reshape(aug_att_mat[0].size(0), aug_att_mat[0].size(0))  # primo layer
    #
    residual_att = torch.eye(att_mat.size(1))
    aug_att_mat[0] = aug_att_mat[0] + residual_att.to(device)
    aug_att_mat[0] = aug_att_mat[0] / aug_att_mat[0].sum(dim=-1).unsqueeze(-1)
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):  # ciclo layers
        # ADDED
        if discard_ratio > 0.0:
            # drop lowest attentions
            li = aug_att_mat[n].reshape(-1)  # flatten 65*65
            # prendo il 90% dei valori più piccoli di flat
            _, indices = li.topk(int(li.size(-1) * discard_ratio), -1, False)
            # mi assicuro di non avere 0 (cls_token) tra gli indici che vogliamo togliere
            indices = indices[indices != 0]
            li[indices] = 0
            aug_att_mat[n] = li.reshape(aug_att_mat[n].size(0), aug_att_mat[n].size(0))  # n_tokens, n_tokens
        #
        residual_att = torch.eye(att_mat.size(1))
        aug_att_mat[n] = aug_att_mat[n] + residual_att.to(device)
        aug_att_mat[n] = aug_att_mat[n] / aug_att_mat[n].sum(dim=-1).unsqueeze(-1)
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])

    last_layer = joint_attentions[-1]  # mappa ultimo layer
    grid_size = int(np.sqrt(aug_att_mat.size(-1)))  # radice quadrata di n_tokens
    mask = last_layer[0, 1:].reshape(grid_size, grid_size).cpu().detach().numpy()
    # cls_token - altri token (ma non cls)
    return mask


def rollout_visualization(atn_wei_list, batch_index, chosen_image, save_path, discard_ratio=0.9, head_fusion='mean'):
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
            % of weights to drop

    head_fusion : str
            Strategy of fusion for the heads. 'mean', 'min', 'max' are the possibilities
    """

    # with open("./cifar100_data/cifar-100-python/meta", "rb") as f:
    #     retrieved_data = pickle.load(f)
    # fine_labels = retrieved_data['fine_label_names']

    name = save_path + '_hf_' + head_fusion + '_rollout_vis_' + str(batch_index) + '.png'
    if discard_ratio > 0.0:
        name = save_path + '_hf_' + head_fusion + '_dr' + str(discard_ratio) + '_rollout_vis_' + str(
            batch_index) + '.png'

    # check if the file exists
    if not os.path.exists(name):
        print("rollout saving")
        mask = rollout(atn_wei_list, discard_ratio=discard_ratio, head_fusion=head_fusion)
        chosen_image = chosen_image.permute(1, 2, 0)  # h, w, c
        tup = (chosen_image.shape[0], chosen_image.shape[1])
        mask = cv2.resize(mask, tup)[..., np.newaxis] / mask.max()  # aggiungo dim canali  h, w, c
        result = (mask * chosen_image.cpu().numpy())  # .astype("uint8")

        # lab = fine_labels[labels[batch_index]]
        # ims = torch.stack([chosen_image, torch.from_numpy(result)])
        # imshow(torchvision.utils.make_grid(ims.permute(0, 3, 1, 2)), 'attention rollout map')

        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(16, 16))
        ax1.set_title('Original')
        ax2.set_title('Attention Map')
        ax3.set_title('Mask resized')
        _ = ax1.imshow(chosen_image)
        _ = ax2.imshow(result)
        _ = ax3.imshow(mask)

        # save fig
        plt.savefig(name)
        # plt.show()


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

    name = save_path + '_heads_vis_' + str(batch_index) + '.png'

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

        for i in range(atn_l0.size(1)):  # heads
            attn_heatmap = bi[i, 0, 1:].reshape((8, 8)).detach().cpu().numpy()  # cls token
            ax = fig.add_subplot(2, 5, i + 2)
            ax.imshow(attn_heatmap)

        # save fig
        plt.savefig(name)
        # plt.show()


if __name__ == '__main__':
    from utils import *
    from get_dataset_dataloaders import *
    import json
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import argparse

    parser = argparse.ArgumentParser(description="Attention weights plot")

    parser.add_argument("--model_weights_path", dest="model_weights_path",
                        default='./model_weights/pretraining_imagenet'
                        , help="path to the folder where are stored the weights of the model in exam (no final slash)")
    parser.add_argument("--weight_epoch", dest="weight_epoch", default=261,
                        help="epoch at which we recover the model weights")
    parser.add_argument("--weight_epoch_ft", dest="weight_epoch_ft", default=71,
                        help="epoch at which we recover the model weights")
    parser.add_argument("--batch_index", dest="batch_index", default=0,
                        help="index of the image of the batch to test")
    parser.add_argument("--atn_plot_path", dest="atn_plot_path", default='./plots/atn_weights'
                        , help="path to the folder where the atn weights images are stored (no final slash)")
    parser.add_argument("--batch_size", dest="batch_size", default=4, help="size of the batch for test the model")
    parser.add_argument("--num_classes", dest="num_classes", default=100, help="number of classes of the dataset "
                                                                               "(we need this info only for the fine "
                                                                               "tuning model)")
    parser.add_argument("--dataset", dest="dataset", default='./cifar100_data', help="dataset to test the model")
    parser.add_argument("--ft", dest="ft", default=0, help="1 for fine tuned model, 0 for pretrained model")

    # params for rollout
    parser.add_argument("--head_fusion", dest="head_fusion", default='mean', help="head fusion strategy "
                                                                                  "for attention rollout")
    parser.add_argument("--discard_ratio", dest="discard_ratio", default=0.0, help="% of atn weights to drop (float)")

    args = parser.parse_args()

    wp = args.model_weights_path
    we = int(args.weight_epoch)
    we_ft = int(args.weight_epoch_ft)
    batch_index = int(args.batch_index)
    batch_size = int(args.batch_size)
    dataset = args.dataset

    if int(args.ft) == 1:
        print("fine tuned model")
        model, device = get_fine_tuned_model(weights_path=wp, weight_epoch=we_ft, num_classes=int(args.num_classes))
    else:
        print("pretrained model")
        model, device = get_pretrained_model(weights_path=wp, weight_epoch=we)

    model.eval()

    with open(wp + '/hyperparams.json') as json_file:
        hyper_params = json.load(json_file)

    hyper_params['batch_size'] = batch_size
    train_loader, validation_loader = get_dataset_fine_tuning(ds=dataset, hyperparams=hyper_params)
    # to get 64x64 images

    # get some random training images
    dataiter = iter(validation_loader)  # rand aug is on train ...
    images, labels = dataiter.next()
    # import torchvision
    # imshow(torchvision.utils.make_grid(images), labels)

    out, atn_wei_list = model(images.to(device))
    acc = (out.argmax(dim=-1) == labels.to(device)).float().mean()

    chosen_image = images[batch_index]  # c, h, w
    chosen_image = chosen_image.permute(1, 2, 0)  # h, w, c
    _, atn_wei_list_im = model(images[batch_index].unsqueeze(0).to(device))

    name_exp = wp.split('/')[-1]  # folder of experiment (from which the weights of the model have been recovered)
    atn_plots_path = os.path.join(args.atn_plot_path, name_exp)
    if not os.path.exists(atn_plots_path):
        os.makedirs(atn_plots_path)
    print("atn plot path: ", atn_plots_path)

    # Heads visualization
    heads_visualization(atn_wei_list, batch_index=batch_index, chosen_image=images[batch_index],
                        save_path=atn_plots_path + '/we_' + str(we))

    # Attention rollout
    rollout_visualization(atn_wei_list=atn_wei_list_im, batch_index=batch_index, chosen_image=images[batch_index],
                          save_path=atn_plots_path + '/we' + str(we), discard_ratio=float(args.discard_ratio),
                          head_fusion=args.head_fusion)

    # Ex: python attention_plots.py --we 91
