import glob
import imageio
import os.path as osp
from torch.utils.data import Dataset


class SceneryDataset(Dataset):
    dataset_dir = 'scenery'

    def __init__(self,
                 data_dir: str = 'data') -> None:
        """
            data_dir:
        """     
        super().__init__()

        self.dataset_dir = osp.join(data_dir, self.dataset_dir)
        self.imgA_paths = glob.glob(f"{self.dataset_dir}/scenery_photo/*.jpg")
        self.imgB_paths = glob.glob(f"{self.dataset_dir}/scenery_cartoon/*/*.jpg")

    def __len__(self):
        return max(len(self.imgA_paths), len(self.imgB_paths))

    def __getitem__(self, index):
        imgA_path = self.imgA_paths[index % len(self.imgA_paths)]
        imgB_path = self.imgB_paths[index % len(self.imgB_paths)]

        imageA, imageB = imageio.v2.imread(imgA_path), imageio.v2.imread(imgB_path)
        
        return imageA, imageB
    
if __name__ == "__main__":
    dataset = SceneryDataset(data_dir='data')
    print(len(dataset)) #14615
    imageA, imageB = dataset[0]
    print(imageA.shape, imageB.shape)
    
    
    import matplotlib.pyplot as plt
    plt.subplot(1, 2, 1)
    plt.imshow(imageA)
    plt.title('Photo Image', fontweight='bold')
    plt.subplot(1, 2, 2)
    plt.imshow(imageB)
    plt.title('Cartoon Image', fontweight='bold')
    plt.show()