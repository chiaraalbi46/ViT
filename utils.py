""" Some functions that help to interpretate the steps """


def imshow(img, labels):
    import matplotlib.pyplot as plt
    import numpy as np
    # img = img / 2 + 0.5  # unnormalize (se metto transform.Normalize)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # (w, h, c)
    plt.title(labels)
    plt.show()


def try_vit(plot=False):
    import torchvision.transforms as transforms
    import torchvision
    import pickle

    batch_size = 4

    transform = transforms.Compose(
        [transforms.ToTensor()])  # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    # with rand augmentation
    # transform = transforms.Compose([
    #     transforms.RandAugment(num_ops=2, magnitude=10),
    #     transforms.ToTensor()])

    trainset = torchvision.datasets.CIFAR100(root='./cifar100_data', train=True, download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    with open("./cifar100_data/cifar-100-python/meta", "rb") as f:
        retrieved_data = pickle.load(f)

    # print("Fine Label Names:", retrieved_data['fine_label_names'])
    # print("Coarse Label Names:", retrieved_data['coarse_label_names'])
    fine_labels = retrieved_data['fine_label_names']
    # todo: modificare codice classe cifar10/100 per aggiungere la coarse label

    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    if plot:
        # # show images
        lab = [fine_labels[labels[j]] for j in range(batch_size)]
        imshow(torchvision.utils.make_grid(images), labels=lab)
        # print labels
        print(' '.join(f'{fine_labels[labels[j]]:5s}' for j in range(batch_size)))

        # patch_embed = PatchEmbedding(img_size=(32, 32), in_chans=3, patch_size=8, embed_dim=60)
        # le = patch_embed(images, plot=True)  # batch_size, n_tokens, embed_dim

        # torchvision.utils.make_grid(sample_batched[0]) resituisce un tensore (3, 36, 138)
        # dove 36 = 32 + 2 + 2 (width di un'immagine più i margini )
        # 138 = 32 * 4 + 10 (height di un'immagine * batch_size più i margini sempre di 2px per un tot di 10px)

    vit = ViT(img_size=32, embed_dim=60, num_channels=3, num_heads=2, num_layers=2, num_classes=100,
              patch_size=8, hidden_dim=60)
    out, atn_weights_list = vit(images)
    return out, atn_weights_list


def plot_embed_patches(embedded_patches, n_images=None):
    import torchvision
    import matplotlib.pyplot as plt

    """ Plot patches for the dataset images.

        Parameters
        ----------
        embedded_patches : torch.tensor
                Embedded patches

        n_images : int
                Number of the images whose patches we want to plot

    """
    b, n = embedded_patches.shape[0], embedded_patches.shape[1]  # batch size, number of patches

    if n_images is not None:
        # faccio vedere l'embedding solo di n_images e non di tutte le immagini (b)
        b = n_images

    fig, ax = plt.subplots(b, 1, figsize=(14, 3))
    fig.suptitle("Embedded patches")

    if b == 1:
        img_grid = torchvision.utils.make_grid(embedded_patches[0], nrow=n, normalize=True, pad_value=0.9)
        img_grid = img_grid.permute(1, 2, 0)
        ax.imshow(img_grid)
        ax.axis('off')
    else:
        for i in range(b):  # number of images
            img_grid = torchvision.utils.make_grid(embedded_patches[i], nrow=n, normalize=True, pad_value=0.9)
            img_grid = img_grid.permute(1, 2, 0)
            ax[i].imshow(img_grid)
            ax[i].axis('off')

    plt.show()
    plt.close()


if __name__ == '__main__':
    from vit import ViT
    from embeddings import PatchEmbedding
    import cv2
    import torch

    # test values
    b = 1
    c = 3
    img_size = (200, 200)  # h, w
    patch_size = 50
    embed_dim = 50

    image = cv2.imread("./imgs/prova_patch.jpg")  # 200, 200, 3
    ima = torch.from_numpy(image)
    ima = torch.unsqueeze(ima, 0)
    # cv2.imshow("Test Image", image)
    # cv2.waitKey(0)

    im = ima.permute(0, 3, 1, 2).float()
    # opencv return a BGR image - to get an RGB image we extract the channels and stack them in the 'right' order
    b_ch = im[:, 0, :, :]  # blue channel
    g_ch = im[:, 1, :, :]  # green channel
    r_ch = im[:, 2, :, :]  # red channel

    im_rgb = torch.stack([r_ch, g_ch, b_ch], dim=1)

    # patch_embed = PatchEmbedding(img_size=img_size, in_chans=c, patch_size=patch_size, embed_dim=embed_dim)
    # le = patch_embed(im_rgb, plot=True)  # 1, 16, 50

    o, atn_wl = try_vit(plot=True)
    # try_vit(plot=True)



