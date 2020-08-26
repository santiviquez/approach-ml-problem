import torch
import numpy as np
from PIL import Image
from PIL import ImageFile

# load images without and ending bit
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ClassicationDataset:
    """
    A general classication dataset class that can be used for all
    kind of image classification problems.
    """

    def __init__(
        self,
        image_paths,
        targets,
        resize=None,
        augmentations=None
    ):
        """
        Args:
            image_paths: list of path to image
            targets: numpy array
            resize: tuple, eg (256, 256), resizes image if not None
            augmentations: albumentation augmentations
        """
        self.image_paths = image_paths
        self.targets = targets
        self.resize = resize
        self.augmentations = augmentations
    

    def __len__(self):
        """
        Return the total number of samples in the dataset
        """
        return len(self.image_paths)
    

    def __getitem__(self, item):
        """
        For a given item index, return everything we need
        to train a given model
        """
        # use PIL to open the image
        image = Image.open(self.image_paths[item])
        # convert image to RGB, we need single channel images
        image = image.convert("RGB")
        # grab correct targets
        targets = self.targets[item]

        # resize if need it
        if self.resize is not None:
            image = image.resize(
                (self.resize[1], self.resize[0]),
                resample=Image.BILINEAR
            )
        
        # convet image to numpy array
        image = np.array(image)

        # if we have albymentation augmentations
        # add them to the image
        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented["image"]
        
        # pytorch expects CHW insted of HWC
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        # return tensors of image and targets
        return {
            "image": torch.tensor(image, dtype=torch.float),
            "targets": torch.tensor(targets, dtype=torch.long)
        }