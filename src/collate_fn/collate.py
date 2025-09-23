import logging
import numpy as np
import torch

from torch.nn.utils.rnn import pad_sequence

from src.collate_fn.t5_mlm_data_collator import FlaxDataCollatorForT5MLM, compute_input_and_target_lengths, group_texts

logger = logging.getLogger(__name__)

class CollateClass:
    def __init__(self, tokenizer, max_length, mlm_items=False, mlm_probability=0.15, mean_span_length=3, decoder_start_token_id=0, gpt2_lm_mode=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_items = mlm_items
        self.mlm_probability = mlm_probability
        self.mean_span_length = mean_span_length
        self.decoder_start_token_id = decoder_start_token_id
        self.gpt2_lm_mode = gpt2_lm_mode
    
    def _generate_spans_per_item(self, input_ids):
        """
        Generate random spans for masking.
        input_ids must be 1-d
        """

        seq_length = (input_ids != self.tokenizer.pad_token_id).sum()
        num_tokens_to_mask = int(seq_length * self.mlm_probability)

        # Use a geometric distribution to decide span lengths
        spans = []
        while num_tokens_to_mask > 0:
            span_len = min(np.random.geometric(p=1 / self.mean_span_length), num_tokens_to_mask)
            start_idx = np.random.randint(0, seq_length - span_len)
            spans.append((start_idx, start_idx + span_len))
            num_tokens_to_mask -= span_len
        
        return spans
    
    def _apply_span_masking_per_item(self, input_ids, spans):
        """
        Replace spans in input_ids with sentinel tokens and generate target sequence.
        input-ids must be 1-d
        """
        sentinel_token_id = self.tokenizer.convert_tokens_to_ids("<extra_id_0>")
        input_ids_masked = input_ids.clone()
        
        for start, end in spans:
            input_ids_masked[start:end] = sentinel_token_id
            sentinel_token_id += 1

        return input_ids_masked

    def _item_input(self, item):
        if isinstance(item, dict):
            if "input" in item:
                return item["input"]
            return item["text"]
        return item[0]

    def _item_target(self, item):
        if isinstance(item, dict):
            if "target" in item:
                return item["target"]
            return item["text"]
        return item[1]
    
    def _update_batch_with_dataset_items(self, batch, dataset_items):
        batch.update({k: [item[k] for item in dataset_items] for k in dataset_items[0]})
        return batch

    def _legacy_mlm_call(self, dataset_items):
        inputs = [self._item_input(item) for item in dataset_items]

        input_encodings = self.tokenizer.batch_encode_plus(
            [sentence for sentence in inputs],
            padding="longest",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        input_ids, attention_mask = input_encodings.input_ids, input_encodings.attention_mask

        input_ids_masked = []
        for i, item in enumerate(input_ids):
            spans = self._generate_spans_per_item(item)
            item_masked = self._apply_span_masking_per_item(item, spans)
            input_ids_masked.append(item_masked)

        input_ids_masked = pad_sequence(
            input_ids_masked,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )

        batch = {
            "input_ids": input_ids_masked,
            "attention_mask": attention_mask,
            "labels": input_ids,
            "decoder_attention_mask": None
        }
        batch = self._update_batch_with_dataset_items(batch, dataset_items)
        return batch

    def _mlm_call(self, dataset_items):
        inputs = [self._item_input(item) for item in dataset_items]
        input_encodings = self.tokenizer(
            [sentence for sentence in inputs],
            return_attention_mask=False
        )
        input_ids = input_encodings.input_ids
        
        expanded_inputs_length, targets_length = compute_input_and_target_lengths(
            inputs_length=self.max_length,
            noise_density=self.mlm_probability,
            mean_noise_span_length=self.mean_span_length
        )

        input_ids = group_texts(input_ids, expanded_inputs_length)
        batch = {"input_ids": np.array(input_ids)}
        
        mlm_data_collator = FlaxDataCollatorForT5MLM(
            tokenizer=self.tokenizer,
            noise_density=self.mlm_probability,
            mean_noise_span_length=self.mean_span_length,
            input_length=self.max_length,
            target_length=targets_length,
            pad_token_id=self.tokenizer.pad_token_id,
            decoder_start_token_id=self.decoder_start_token_id
        )

        batch = mlm_data_collator(batch)

        # def pad_numpy_to_max_length(ids, max_length):
        #     pad_size = max_length - ids.shape[-1]
        #     pad_id = self.tokenizer.pad_token_id
        #     return np.pad(ids, ((0, 0), (0, pad_size)), mode="constant", constant_values=pad_id)

        # if batch["input_ids"].shape[-1] < mlm_data_collator.input_length:
        #     batch["input_ids"] = pad_numpy_to_max_length(batch["input_ids"], mlm_data_collator.input_length)

        batch_torch = {}
        for k, v in batch.items():
            if isinstance(v, np.ndarray):
                batch_torch[k] = torch.from_numpy(v)
            else:
                batch_torch[k] = v

        batch_torch["attention_mask"] = (batch_torch["input_ids"] != self.tokenizer.pad_token_id).long()
        batch_torch["decoder_attention_mask"] = (batch_torch["labels"] != self.tokenizer.pad_token_id).long()

        batch_torch = self._update_batch_with_dataset_items(batch_torch, dataset_items)
        return batch_torch

    def _gpt2_lm_call(self, dataset_items):
        """Handle datasets that return {'text': ...} format for GPT2 language modeling"""
        texts = [self._item_input(item) for item in dataset_items]
        
        # For language modeling, we use the same text as both input and target
        # The GPT2 model will handle the shifting internally
        text_encodings = self.tokenizer.batch_encode_plus(
            texts,
            padding="longest",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        input_ids, attention_mask = text_encodings.input_ids, text_encodings.attention_mask
        
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),  # Same as input for language modeling
            "decoder_attention_mask": attention_mask.clone()
        }
        batch = self._update_batch_with_dataset_items(batch, dataset_items)
        return batch

    def __call__(self, dataset_items):
        if self.mlm_items:
            return self._mlm_call(dataset_items)
        
        if self.gpt2_lm_mode:
            return self._gpt2_lm_call(dataset_items)

        inputs = [self._item_input(item) for item in dataset_items]
        targets = [self._item_target(item) for item in dataset_items]

        input_encodings = self.tokenizer.batch_encode_plus(
            [sentence for sentence in inputs],
            padding="longest",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        input_ids, attention_mask = input_encodings.input_ids, input_encodings.attention_mask

        target_encodings = self.tokenizer.batch_encode_plus(
            [sentence for sentence in targets],
            padding="longest",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        labels, decoder_attention_mask = target_encodings.input_ids, target_encodings.attention_mask
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "decoder_attention_mask": decoder_attention_mask
        }
        batch = self._update_batch_with_dataset_items(batch, dataset_items)
        return batch
