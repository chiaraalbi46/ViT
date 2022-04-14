"""  Linear and positional embedding weights plot """


def transform_pca(filter, embed_dim, patch_size):
    from sklearn.decomposition import PCA
    import torch
    pca = PCA()

    filter = torch.reshape(filter, (embed_dim, -1))
    filter = torch.transpose(filter, 0, 1)
    filter = pca.fit_transform(filter)
    filter = torch.from_numpy(filter)
    filter = torch.transpose(filter, 0, 1)
    filter = torch.reshape(filter, (embed_dim, 3, patch_size, patch_size))

    return filter


def filter_visualization(tensor, num_comp, name, save_path, nrow=7, padding=1):
    import torchvision
    save_path = save_path + '_lin_embedding_weights.png'

    # check if the file exists
    if not os.path.exists(save_path):

        # Visualizzo solo le prime num_comp
        filter = tensor[0:num_comp]

        rows = filter.shape[0] // nrow + 1
        grid = torchvision.utils.make_grid(filter, nrow=nrow, normalize=True, padding=padding)  # c, h, w
        print(grid.shape)
        plt.figure(figsize=(nrow, rows))
        plt.title('Linear embedding weights \n(first 28 principal components)\n' + name)
        plt.imshow(grid.numpy().transpose((1, 2, 0)))  # h, w, c
        # save fig
        plt.savefig(save_path)
        # plt.show()


def positional_embed_visualization(pos_embed, name, save_path):
    import torch.nn.functional as F

    save_path = save_path + '_pos_embedding_weights.png'

    # check if the file exists
    if not os.path.exists(save_path):
        fig = plt.figure(figsize=(8, 8))
        plt.title("Visualization of position embedding similarities \n" + name, fontsize=18)

        for i in range(1, pos_embed.shape[0]):  # non considero il cls token
            # One cell shows cos similarity between an embedding and all the other embeddings.
            sim = F.cosine_similarity(pos_embed[i:i + 1], pos_embed[1:], dim=1)  # n_patches
            sim = sim.reshape((8, 8)).detach().cpu().numpy()  # una patch
            ax = fig.add_subplot(8, 8, i)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            ax.imshow(sim)
        # save fig
        plt.savefig(save_path)
        # plt.show()


if __name__ == '__main__':
    # function()
    from utils import *
    import json
    import matplotlib.pyplot as plt
    import os
    import argparse

    parser = argparse.ArgumentParser(description="Linear and positional embedding weights plot")

    parser.add_argument("--model_weights_path", dest="model_weights_path", default='./model_weights/pretraining_imagenet'
                        , help="path to the folder where are stored the weights of the model in exam (no final slash)")
    parser.add_argument("--weight_epoch", dest="weight_epoch", default=1,
                        help="epoch at which we recover the model weights")
    parser.add_argument("--num_comp", dest="num_comp", default=28,
                        help="number of principal components to visualize for linear embedding weights plot")
    parser.add_argument("--le_weights_path", dest="le_weights_path", default='./plots/linear_embedding_weights'
                        , help="path to the folder where the lin embedding weights images are stored (no final slash)")
    parser.add_argument("--pe_weights_path", dest="pe_weights_path", default='./plots/positional_embedding_weights'
                        , help="path to the folder where the positional embedding weights images are stored "
                               "(no final slash)")

    args = parser.parse_args()

    wp = args.model_weights_path
    we = int(args.weight_epoch)
    folder = wp.split('/')[-1]
    lin_emb_path = os.path.join(args.le_weights_path, folder)
    pos_emb_path = os.path.join(args.pe_weights_path, folder)

    if not os.path.exists(lin_emb_path):
        os.makedirs(lin_emb_path)

    if not os.path.exists(pos_emb_path):
        os.makedirs(pos_emb_path)

    model, _ = get_pretrained_model(weights_path=wp, weight_epoch=we)
    with open(wp + '/hyperparams.json') as json_file:
        hyper_params = json.load(json_file)

    # Visualize linear embedding weights
    le_weights = model.patch_embedding.linear_embedding.weight.cpu().detach()  # (embed_dim, patch_size^2 * in_chans)
    filter = transform_pca(le_weights, embed_dim=hyper_params['embed_dim'], patch_size=hyper_params['patch_size'])
    filter_visualization(tensor=filter, num_comp=int(args.num_comp), name='Epoch ' + str(we),
                         save_path=lin_emb_path + '/we_' + str(we))

    # Visualize position embedding similarities.
    pe_weights = model.position_embedding.position_embedding.cpu().detach()  # (n_token, embed_dim)
    positional_embed_visualization(pos_embed=pe_weights, name='Epoch ' + str(we),
                                   save_path=pos_emb_path + '/we_' + str(we))