## Fork of Multilingual Unsupervised Sentence Simplification

Original repository can be found at https://github.com/facebookresearch/muss.

Most of the changes in this fork are for experiments relating to paraphrastic sentence simplification in EN and DE.

We've extended API-access and some functionality for our purposes (see `custom_scripts`).
- `prepare_newsela_data_for_param_optim_and_eval.sh`
- `param_search.py`
  - prepares and runs new parameter search on Newsela sentence aligned data sets
- `compute_label_scores_on_corpus.py`
  - applies preprocessor scores to a set of complex-simple sentence pairs (NOTE: this should be updated to the original preprocessor functions rather than self-implemented scorers)
- can run inference on a file with hard-coded control token parameters, making evaluation of different control token values easier.

Other changes were made to the mining pipeline and supporting scripts in order to run on our infrastructure.
- `scripts/mine_sequences.py`
  - changes made according to s3it infrastructure
  - added support for new language: German
- `muss/train_kenlm.py`
  - training new LMs for German paraphrase mining
- `muss/kenlm_helpers.py`
  - essentially the same as the original `muss/kenlm.py` but renamed to avoid import issues with kenlm


---


<!-- # Multilingual Unsupervised Sentence Simplification

Code and pretrained models to reproduce experiments in "MUSS: Multilingual Unsupervised Sentence Simplification by Mining Paraphrases".

## Prerequisites

Linux with python 3.6 or above (not compatible with python 3.9 yet).

## Installing

```
git clone git@github.com:facebookresearch/muss.git
cd muss/
pip install -e .  # Install package
python -m spacy download en_core_web_md fr_core_news_md es_core_news_md  # Install required spacy models
```

## How to use
Some scripts might still contain a few bugs, if you notice anything wrong, feel free to open an issue or submit a Pull Request.

### Simplify sentences from a file using pretrained models
```python
# English
python scripts/simplify.py scripts/examples.en --model-name muss_en_wikilarge_mined
# French
python scripts/simplify.py scripts/examples.fr --model-name muss_fr_mined
# Spanish
python scripts/simplify.py scripts/examples.es --model-name muss_es_mined
```

Pretrained models should be downloaded automatically, but you can also find them here:  
[muss_en_wikilarge_mined](https://dl.fbaipublicfiles.com/muss/muss_en_wikilarge_mined.tar.gz)  
[muss_en_mined](https://dl.fbaipublicfiles.com/muss/muss_en_mined.tar.gz)  
[muss_fr_mined](https://dl.fbaipublicfiles.com/muss/muss_fr_mined.tar.gz)  
[muss_es_mined](https://dl.fbaipublicfiles.com/muss/muss_es_mined.tar.gz)  

### Mine the data
```python
python scripts/mine_sequences.py
```

### Train the model
```python
python scripts/train_model.py
```

### Evaluate simplifications
Please head over to [EASSE](https://github.com/feralvam/easse/) for Sentence Simplification evaluation.


## License

The MUSS license is CC-BY-NC. See the [LICENSE](LICENSE) file for more details.

## Authors

* **Louis Martin** ([louismartincs@gmail.com](mailto:louismartincs@gmail.com))

 -->

## Citation

If you use MUSS in your research, please cite [MUSS: Multilingual Unsupervised Sentence Simplification by Mining Paraphrases](https://arxiv.org/abs/2005.00352)

```
@article{martin2021muss,
  title={MUSS: Multilingual Unsupervised Sentence Simplification by Mining Paraphrases},
  author={Martin, Louis and Fan, Angela and de la Clergerie, {\'E}ric and Bordes, Antoine and Sagot, Beno{\^\i}t},
  journal={arXiv preprint arXiv:2005.00352},
  year={2021}
}
```
