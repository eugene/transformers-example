## :robot: Example transformer implementation

This is a simple transformer implementation based on [Transformers from scratch](http://peterbloem.nl/blog/transformers) suitable for generation and embeddings extraction. As the [dataset](https://www.kaggle.com/rtatman/english-word-frequency) we used most commonly-used single words on the English language web (Trained in character-by-character fashion). 

### :train: Training 
To train a model simply run:
```
python main.py
```
On training completion a file `words.model.pth` will be created - it will include training progress
model parameters and other stuff ready to be explored by the `notebook.ipynb` jupyter notebook (see
below).

### :thought_balloon: Evaluation, embeddings and generation
To explore the results of the training see `notebook.ipynb` jupyter notebook

### :left_right_arrow: Future directions
While this conditional transformer model architecture is very promising for generation,
better embeddings can be extracted with [BERT](https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270) (biderectional transformer).
