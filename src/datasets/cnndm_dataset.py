from src.datasets import HuggingFaceDataset
from transformers import AutoTokenizer

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
            model_name="google/t5-base",
            filter_max_length=True,
            **kwargs
            ):
        super().__init__(path, name, streaming, split, data_files, max_samples, shuffle, shuffle_seed)

        self._preprocess()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if filter_max_length:
            self.max_length = max_length
            self._filter_max_length(max_length)

    def _filter_max_length(self, max_length):
        dataset_size = len(self.dataset)

        def _keep(sample):
            toks = self.tokenizer(sample["input"], truncation=False)
            return len(toks["input_ids"]) <= max_length
        
        if hasattr(self.dataset, "filter"):
            self.dataset = self.dataset.filter(_keep)
        else:
            self.dataset = [s for s in self.dataset if _keep(s)]

        print(f' \
            Filtered dataset by total input_ids max_length="{max_length}", \
            size reduced from {dataset_size} to {len(self.dataset)} samples! \
        ')
    
    def _preprocess(self):
        items = []

        for sample in self.dataset:
            input = sample["article"]
            target = sample["highlights"]
            items.append({"input": input, "target": target})
        
        self.dataset = items
