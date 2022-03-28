#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from transformers import BartForConditionalGeneration, BartTokenizer

model_path = sys.argv[1]

tokenizer = BartTokenizer.from_pretrained(model_path)
model = BartForConditionalGeneration.from_pretrained(model_path, return_dict=True)
model.eval()

texts = [
    "Lula served as president from 2003-2010.",
    "Tickets can be purchased at the gate or online.",
    "There will not even be any record when it's been removed from your credit profile. Many feel like settling debt cheats creditors out of their owed payments, but this is not the goal. The approach is created for individuals who are not able to repay what they owe.",
    "And they beckoned unto their partners, which were in the other ship, that they should come and help them. And they came, and filled both ships, so they began to sink.",
]


inputs = tokenizer(texts, padding=True, return_tensors="pt")

# Generate Summary
summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=256)

outputs = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

for i, o in zip(texts, outputs):
    print('----')
    print('input:', i)
    print('output:', o)