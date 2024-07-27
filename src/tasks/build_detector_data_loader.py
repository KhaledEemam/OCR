from torch.utils.data import DataLoader
from .build_detector_dataset import CustomDetectorDataset

class getDetectorLoader :
    def __init__(self,images,targets ,batch_size, num_workers , shuffle , pin_memory ,max_height,max_width) :
        self.dataset = CustomDetectorDataset(images,targets,max_height,max_width)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle

    def get_loader(self) :
        return DataLoader(self.dataset , batch_size = self.batch_size , shuffle = self.shuffle , num_workers = self.num_workers , pin_memory = self.pin_memory)