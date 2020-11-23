## :robot: Example tranformer implementation

This is a simple transformer implementation based on [Transformers from scratch](http://peterbloem.nl/blog/transformers) suitable for generation and embeddings extraction. As the [dataset](https://www.kaggle.com/rtatman/english-word-frequency) we used most commonly-used single words on the English language web (Trained in character-by-character fashion). 

### Training 
To train a model simply run:
```
python main.py
```

### Evaluation, embeddings and generation
To explore the results of the training see `notebook.ipynb` jupyter notebook

### Future directions
While this conditional transformer model architecture is very promising for generation,
better embeddings can be extracted with [BERT](https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270).