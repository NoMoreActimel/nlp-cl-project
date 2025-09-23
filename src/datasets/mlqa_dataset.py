import re
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoTokenizer

from src.datasets import HuggingFaceDataset

class MLQAHuggingFaceDataset(TorchDataset):
    def __init__( 
            self,
            name="mlqa.en.en",
            huggingface_split="test", # No train available, we split test manually
            streaming=False,
            shuffle=True,
            shuffle_seed=52, # always provide shuffle_seed, otherwise train_test_split will give different splits
            split="train",
            split_train_val_test=True,
            split_random_state=42,
            val_size=0.1,
            test_size=0.1,
            model_type="enc-dec",
            filter_max_length=True,
            max_length=1024,
            model_name="mt5-base", # for tokenization in case of max-length filtering
            **kwargs
    ):
        super().__init__()
        self.dataset = HuggingFaceDataset(
            path="facebook/mlqa",
            name=name,
            streaming=streaming,
            split=huggingface_split,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed
        )

        self.lang = name.split('.')[-1]

        self.model_type = model_type
        self._preprocess()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if filter_max_length:
            self.max_length = max_length
            self.filter_max_length(max_length)

        self.split = split
        self.split_train_val_test = split_train_val_test
        self.split_random_state = split_random_state
        self.val_size = val_size
        self.test_size = test_size
        self._train_test_split()
        self.calc_max_lengths()

    def _pad_punctuation(self, text):
        text = re.sub(r'([^\w\s])', r' \1 ', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    def _string_join(self, lst):
        return re.sub(r'\s+', ' ', ' '.join(lst))

    def calc_max_lengths(self):
        max_ques_len = 0
        max_ans_len = 0
        for i, item in enumerate(self.dataset):
            ques_ids = self.tokenizer(item["text"], truncation=False)["input_ids"]
            ans_ids = self.tokenizer(item["answer"], truncation=False)["input_ids"]
            if i % 1000 == 0:
                print(f"QUESTION: {item['text']}")
                print(f"ANSWER: {item['answer']}")
            max_ques_len = max(len(ques_ids), max_ques_len)
            max_ans_len = max(len(ans_ids), max_ans_len)

        print(f"\nDATASET LANGUAGE: {self.lang}")
        print(f"DATASET LENGTH: {len(self.dataset)}")
        print(f"MAX QUESTION LENGTH IN MLQA DATASET: {max_ques_len}")
        print(f"MAX ANSWER LENGTH IN MLQA DATASET: {max_ans_len}\n")

    def _train_test_split(self):
        train, val, test, train_plus_val = self.dataset, None, None, self.dataset
        if self.test_size > 0.0:
            train_plus_val, test = train_test_split(
                self.dataset,
                test_size=self.test_size,
                random_state=self.split_random_state
            )
        rel_val_size = self.val_size / (1.0 - self.test_size)
        if self.val_size > 0.0:
            train, val = train_test_split(
                train_plus_val,
                test_size=rel_val_size,
                random_state=self.split_random_state
            )
        
        if self.split == "train":
            self.dataset = train
        elif self.split == "val":
            self.dataset = val
        else:
            self.dataset = test
        
    def filter_max_length(self, max_length):
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
            context = self._pad_punctuation(sample["context"])
            question = self._pad_punctuation(sample["question"])
            inputs = self._string_join(['question:', question, 'context:', context, 'answer:'])
            labels = [self._pad_punctuation(label) for label in sample["answers"]["text"]]

            item = {"text": inputs, "answer": labels, "lang": self.lang}
            if self.model_type == "enc-dec":
                item["input"] = inputs
                item["target"] = labels[0]
            else:
                full_text = inputs + labels[0]
                item["input"] = full_text
                item["target"] = full_text
            items.append(item)
            
        self.dataset = items

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
