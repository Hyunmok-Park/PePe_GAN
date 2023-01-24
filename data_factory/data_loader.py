import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def ImageLoader(data_path='dataset', batch_size=4, img_size=36):
    dataset = dset.ImageFolder(root=data_path,
                               transform=transforms.Compose([
                                   transforms.Resize(img_size),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize(0, 1),
                               ]))
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size)
    return data_loader