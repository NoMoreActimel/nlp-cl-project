# Adapted from: https://github.com/qcwthu/Lifelong-Fewshot-Language-Learning/blob/cf7d17ce7de6a707d929d0542b3d5e639569855f/Summarization/model.py

import os
import pdb
import sys
import torch
import torch.nn as nn

from torch.nn.functional import kl_div
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class T5forSummarization(nn.Module):
    def __init__(self, model_name, cache_dir, max_length, max_new_tokens=None, output_hidden_states=True):
        super(T5forSummarization, self).__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.decoder_start_token_id_use = self.model.config.decoder_start_token_id
        self.output_hidden_states = output_hidden_states

    def _step(
            self, input_ids, attention_mask=None, decoder_input_ids=None, labels=None, decoder_attention_mask=None, **kwargs
    ):
        # embeddings = self.model.encoder.embed_tokens(input_ids)
        return self.model(
            # inputs_embeds=embeddings,
            input_ids=input_ids,
            attention_mask=attention_mask,
            # decoder_input_ids=decoder_input_ids,
            # decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            output_attentions=True,
            output_hidden_states=True
        )

    def _step_pre(
            self, input_ids, attention_mask=None, decoder_input_ids=None, labels=None, decoder_attention_mask=None
    ):
        # embeddings = self.model.encoder.embed_tokens(input_ids)
        return self.model(
            # inputs_embeds=embeddings,
            input_ids=input_ids,
            attention_mask=attention_mask,
            # decoder_input_ids=decoder_input_ids,
            # decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            output_attentions=True,
            output_hidden_states=True
        )

    def forward(self, batch):
        outputs = self._step(**batch)
        return outputs.loss

    def _generative_step(self, batch):
        # embedding = self.model.encoder.embed_tokens(batch["input_ids"])
        # decoder_input_ids = (
        #     torch.ones(
        #         (batch["input_ids"].shape[0], 1),
        #         dtype=torch.long, device=batch["input_ids"].device
        #     ) * self.decoder_start_token_id_use
        # )
        generated_ids = self.model.generate(
            # inputs_embeds=embedding,
            # decoder_input_ids=decoder_input_ids,
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=True,
            # max_length=self.max_length,
            max_new_tokens=self.max_new_tokens,
            num_beams=2,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True
        )

        preds = self.ids_to_clean_text(generated_ids)
        target = self.ids_to_clean_text(batch["labels"]) #[torch.where(batch["labels"] != -100)])
        input = self.ids_to_clean_text(batch["input_ids"]) #[torch.where(batch["labels"] != -100)])
        return input, target, preds

    def _generative_samples(self, batch):
        # embedding = self.model.encoder.embed_tokens(batch["input_ids"])
        # decoder_input_ids = (
        #     torch.ones(
        #         (batch["input_ids"].shape[0], 1),
        #         dtype=torch.long, device=batch["input_ids"].device
        #     ) * self.decoder_start_token_id_use
        # )

        generated_ids = self.model.generate(
            # inputs_embeds=embedding,
            # decoder_input_ids=decoder_input_ids,
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=True,
            # max_length=self.max_length,
            max_new_tokens=self.max_new_tokens,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True,
            do_sample=True,
            top_k = 64,
            num_return_sequences=3
        )

        preds = self.ids_to_clean_text(generated_ids)
        target = self.ids_to_clean_text(batch["labels"])
        input = self.ids_to_clean_text(batch["input_ids"])
        return input, target, preds


    def ids_to_clean_text(self, generated_ids):
        generated_ids = generated_ids.clone()
        generated_ids[generated_ids == -100] = self.tokenizer.pad_token_id

        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return self.lmap(str.strip, gen_text)

    def lmap(self, f, x):
        """list(map(f, x))"""
        return list(map(f, x))
