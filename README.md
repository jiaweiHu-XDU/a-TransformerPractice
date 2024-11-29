# Machine Translation with Transformer: Full Process and Implementation

## Set Up Environment
I used the almost latest libraries, and all versions information could be seen in `environment.yml`.

Just use the following command to prepare your environment.
```sh
conda env create -f environment.yml
```

## Download the Dataset
Training the translator, I used the open [Multi30K Dataset](https://github.com/multi30k/dataset) to train and evaluate model.

Extract all files of `task1` and put them under the directory `data/multi30k`.

## Training Transformer Model
I provide the training setup file to train model from scratch. Just use the command to start:
```sh
python main.py --epochs 1000 > output/output.log
```

## Analysis and Exploration
For convenience of analysing the transformer pipline, I also provided a jupyter notebook which travels the whole processing from tokenizing texts to generate a target text.

However, this notebook is just used for analysis rather than training a model, so its accuracy doesn't reflect the performance of a final model in a production setting.

## Reference
+ [https://github.com/hyunwoongko/transformer](https://github.com/hyunwoongko/transformer)
+ [Transformer: Concept and code from scratch](https://mina-ghashami.github.io/posts/2023-01-10-transformer/)