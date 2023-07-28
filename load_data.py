from torch.utils.data import Dataset, DataLoader


def load_data(directory_path,patch_size,image_size):
  '''
  Takes: the path of a directory with images, patch_size,image_size
  Returns :a Torch DataLoader object with images in format
          which can be directly passed to a VIT style encoder
  '''
