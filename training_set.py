from dataset import Dataset

class Training_set(Dataset):

    def __init__(self, dataset: Dataset):
        super().__init__(dataset.dataset, dataset.target, dataset.split)
        self.dataset = self._training_set
        self.target = self._training_target
