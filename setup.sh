mkdir data
wget -P data https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
wget -P data https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
rm glove.840B.300d.zip

FASTTEXT_DIR=~/data/fasttext
mkdir -p $FASTTEXT_DIR
wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki-news-300d-1M.vec.zip -O $FASTTEXT_DIR/wiki-news-300d-1M.vec.zip
unzip $FASTTEXT_DIR/wiki-news-300d-1M.vec.zip -d $FASTTEXT_DIR
rm wiki-news-300d-1M.vec.zip