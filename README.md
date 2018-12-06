# Dual Question-Answer Generator for Machine Reading Comprehension

Answer Generation Implementation - https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf
![Alt text](/../master/screenshots/architecture.png?raw=true "R-NET")

Question Generation Implementation - http://www.aclweb.org/anthology/N18-2090

The dataset used for this task is Stanford Question Answering Dataset (https://rajpurkar.github.io/SQuAD-explorer/). 
Pretrained GloVe embeddings are used for both words (https://nlp.stanford.edu/projects/glove/) and 
characters (https://github.com/minimaxir/char-embeddings/blob/master/glove.840B.300d-char.txt).

## Requirements
  * Python2.7
  * NumPy
  * tqdm
  * spacy
  * tensorflow-gpu==1.12
  *	graphviz
  * pythonrogue

# Downloads and Setup
Once you clone this repo, run the following lines from bash **just once** to process the dataset (SQuAD).
```shell
$ pipenv install
$ bash setup.sh
$ pipenv shell
$ python process.py --reduce_glove True --process True
```

### Data format for Question Generation

```
[{"text1":"IBM is headquartered in Armonk , NY .", 
 {"text2":"Where is IBM located ?", 
 {"text3":"Armonk , NY"
}]
```

where "text1" and "annotation1" correspond to the text and rich annotations for the passage. Similarly, "text2" and "text3" correspond to the question and answer parts, respectively. 


# Training the Answer Generation Model

```shell
$ python model.py
```

# Training the Question Generation Model

```
python src/NP2P_trainer.py --config_path config.json
```
where config.json is a JSON file containing all hyperparameters.
We attach a sample [config](./config.json) file along with our repository.

## To test or debug Answer Generation model 
After training, change mode="train" to debug or test from params.py file and run the model.

## For Evaluation purpose of Question Generation Model
```
python NP2P_beam_decoder.py --model_prefix xxx --in_path yyy --out_path zzz --mode beam
```

### Demo Purpose
Run the script
python inference.py
