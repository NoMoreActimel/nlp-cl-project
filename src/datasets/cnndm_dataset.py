from src.datasets import HuggingFaceDataset

class CNNDMHuggingFaceDataset(HuggingFaceDataset):
    def __init__(
            self,
            path,
            name=None,
            streaming=False,
            split=None,
            data_files=None,
            max_samples=None,
            shuffle=None,
            shuffle_seed=None,
            max_length=1024,
            **kwargs
            ):
        super().__init__(path, name, streaming, split, data_files, max_samples, shuffle, shuffle_seed)

        self._preprocess()
        self._filter_max_length(max_length)

    def _filter_max_length(self, max_length):
        dataset_size = len(self.dataset)

        def _keep(sample):
            return len(self.tokenizer(sample["input"], truncation=False)["input_ids"]) <= max_length
        
        self.dataset = self.dataset.filter(_keep)

        print(f' \
            Filtered dataset by total input_ids max_length="{max_length}", \
            size reduced from {dataset_size} to {len(self.dataset)} samples! \
        ')
    
    def _preprocess(self):
        items = []

        for sample in self.dataset:
            input = sample["article"]
            target = sample["summary"]
            items.append({"input": input, "target": target})
        
        self.dataset = items
