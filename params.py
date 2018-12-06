#all the parameters used for Answer Generation model
import os
import tensorflow as tf
flags = tf.flags
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class Params():

    # data
    data_size = -1 # -1 to use all data
    num_epochs = 10
    train_prop = 0.9 # Not implemented atm
    data_dir = "./data/"
    train_dir = data_dir + "trainset/"
    dev_dir = data_dir + "devset/"
    logdir = "./train/train"
    glove_dir = "./glove.840B.300d.txt" # Glove file name (If you want to use your own glove, replace the file name here)
    glove_char = "./glove.840B.300d.char.txt" # Character Glove file name
    coreNLP_dir = "./stanford-corenlp-full-2017-06-09" # Directory to Stanford coreNLP tool
    fasttext_file = data_dir + "fasttext/"
    word2idx_file = data_dir + "word2idx.json"
    char2idx_file = data_dir + "char2idx.json"
    # Data dir
    target_dir = "indices.txt"
    q_word_dir = "words_questions.txt"
    q_chars_dir = "chars_questions.txt"
    p_word_dir = "words_context.txt"
    p_chars_dir = "chars_context.txt"

    train_record_file = data_dir + "train.tfrecords"
    dev_record_file = data_dir + "dev.tfrecords"
    test_record_file = data_dir + "test.tfrecords"

    word_emb_file = data_dir + "word_emb.json"
    char_emb_file = data_dir + "char_emb.json"
    train_eval_file = data_dir + "train_eval.json"
    dev_eval_file = data_dir +  "dev_eval.json"
    test_eval_file = data_dir + "test_eval.json"
    dev_meta = data_dir + "dev_meta.json"
    test_meta = data_dir + "test_meta.json"
    log_dir = "log/event"
    save_dir = "log/model"
    answer_dir = "log/answer"
    answer_file = answer_dir + "answer.json"




    # Training
	# NOTE: To use demo, put batch_size == 1
    mode = "train" # case-insensitive options: ["train", "test", "debug"]
    dropout = 0.2 # dropout probability, if None, don't use dropout
    zoneout = None # zoneout probability, if None, don't use zoneout
    optimizer = "adam" # Options: ["adadelta", "adam", "gradientdescent", "adagrad"]
    batch_size = 64 if mode is not "test" else 100# Size of the mini-batch for training
    save_steps = 50 # Save the model at every 50 steps
    clip = True # clip gradient norm
    norm = 5.0 # global norm
    # NOTE: Change the hyperparameters of your learning algorithm here
    opt_arg = {'adadelta':{'learning_rate':1, 'rho': 0.95, 'epsilon':1e-6},
                'adam':{'learning_rate':1e-3, 'beta1':0.9, 'beta2':0.999, 'epsilon':1e-8},
                'gradientdescent':{'learning_rate':1},
                'adagrad':{'learning_rate':1}}

    # Architecture
    SRU = True # Use SRU cell, if False, use standard GRU cell
    max_p_len = 300 # Maximum number of words in each passage context
    max_q_len = 30 # Maximum number of words in each question context
    max_char_len = 16 # Maximum number of characters in a word
    vocab_size = 91605 # Number of vocabs in glove.840B.300d.txt + 1 for an unknown token
    char_vocab_size = 95 # Number of characters in glove.840B.300d.char.txt + 1 for an unknown character
    emb_size = 300 # Embeddings size for words
    char_emb_size = 8 # Embeddings size for characters
    attn_size = 75 # RNN cell and attention module size
    num_layers = 3 # Number of layers at question-passage matching
    bias = True # Use bias term in attention
    pretrained_char = False
    glove_word_size = int(2.2e6)
    test_para_limit = 1000
    test_ques_limit = 100
    ques_limit = 100
    para_limit = 400
    char_limit = 16
    grad_clip = 5.0

    capacity = 15000
    num_threads = 4
    use_cudnn = True
    is_bucket = False
    bucket_range = [40, 361, 40]

    batch_size = 64
    num_steps = 60000
    checkpoint = 10
    period = 100
    val_num_batches = 150
    init_lr =0.5
    keep_prob = 0.7
    ptr_keep_prob = 0.7
    grad_clip = 5.0
    hidden = 75
    char_hidden = 100
    patience = 3
