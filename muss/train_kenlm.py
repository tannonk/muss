#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

from transformers import PreTrainedTokenizerFast
from muss.kenlm_helpers import train_kenlm_language_model, get_spm_tokenizer, get_kenlm_log_prob

ap = argparse.ArgumentParser()
ap.add_argument('--input_data_paths', type=str, nargs='*', default=['/scratch/tkew/wiki_dumps/dewiki/dewiki.txt'])
ap.add_argument('--model_dir', type=str, required=True, default='/scratch/tkew/muss/resources/models/language_models/kenlm_dewiki')
ap.add_argument('--do_train', action='store_true')
ap.add_argument('--inspect_tok', action='store_true')
ap.add_argument('--inspect_lm', action='store_true')
args = ap.parse_args()

de_texts = [
    """Dem französischen Militär gelang es, militärisch die Oberhand zu behalten.""",
    """Kriegsverluste und Menschenrechtsverletzungen inklusive Folter machten die Auseinandersetzung in Frankreich jedoch so unpopulär, dass sie von politischer Seite beendet wurde und zur Unabhängigkeit Algeriens führte.""",
    """Der EU-Verordnung zufolge ist dafür die Bundesbank zuständig.""",
    """Zuvor hatte er die Legalisierung der sogenannten aktiven Sterbehilfe gefordert.Möglich wurde die Überwachung per Kiez-Kameras erst durch eine im vergangenen Herbst beschlossene Novelle des Hamburger Polizeigesetzes, die die gesetzlichen Grundlagen für die Video-Überwachung lieferte.""",
    """Neben namhaften Marken aus der Autopflege wie Meguiars, Surf City Garage, Dodo Juice und Menzerna bietet der Lupus Autopflege Shop unter anderem Artikel der Kategorien Lackpflege zu der auch die.""",
    """Ich heiße Simone.. Ich hoffe, meine Bilder gefallen Dir und Du bist neugierig, wie der Rest meines Körpers aussieht. ;) Ich chatte gerne, schaue zu und zeige auch mein Body... ;)"""
]

en_texts = [
    """In contrast, (plain) depth-first search, which explores the node branch as far as possible before backtracking and expanding other nodes""",
    """This article needs additional citations for verification. Please help improve this article by adding citations to reliable sources. Unsourced material may be challenged and removed.""",
    """You can adjust your preferences including your right to object where legitimate interest is used, or withdraw your consent to certain Cookies at any time.""",
    """We can't help but want more hot sex stories, honeymoon sex, sex with a stranger or even pool sex – we're extremely curious to learn what people get up to."""
]

if not isinstance(args.input_data_paths, list):
    raise TypeError('input_data_paths should be a list of paths!')

if args.do_train:
    if (Path(args.model_dir) / 'kenlm_model.arpa').exists():
        raise RuntimeError(f'a .arpa file already exists in the model directory')
    train_kenlm_language_model(args.input_data_paths, args.model_dir)
    print("done!")

# import pdb;pdb.set_trace()

texts = de_texts if 'dewiki' in args.model_dir else en_texts

if args.inspect_tok:
    tokenizer = get_spm_tokenizer(Path(args.model_dir))
    for text in texts:
        print(tokenizer.encode(text).tokens)

if args.inspect_lm:
    for text in texts:
        slope = -0.6
        print(text, '--->', get_kenlm_log_prob(text, Path(args.model_dir)))
        print(text, '--->', get_kenlm_log_prob(text, Path(args.model_dir)) / len(text) < slope)