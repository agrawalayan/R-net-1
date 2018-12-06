import tensorflow as tf
import spacy
import os
import numpy as np
import ujson as json
#import src/NP2P_beam_decoder as qg
import sys
sys.path.insert(0, 'src')
import NP2P_beam_decoder as qg
#from src import NP2P_beam_decoder as qg
from params import Params
from data_load import cudnn_gru, native_gru, dot_attention, summ, ptr_net
from process import word_tokenize, convert_idx
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from flask import Flask, request
app = Flask(__name__)

@app.route("/")
def home():
    with open('demo.html', 'r') as fl:
        html = fl.read()
        return html





class InfModel(object):
    # Used to zero elements in the probability matrix that correspond to answer
    # spans that are longer than the number of tokens specified here.
    max_answer_tokens = 15

    def __init__(self, word_mat, char_mat):
        self.c = tf.placeholder(tf.int32, [1, None])
        self.q = tf.placeholder(tf.int32, [1, None])
        self.ch = tf.placeholder(tf.int32, [1, None, Params.char_limit])
        self.qh = tf.placeholder(tf.int32, [1, None, Params.char_limit])
        self.tokens_in_context = tf.placeholder(tf.int64)

        self.word_mat = tf.get_variable("word_mat", initializer=tf.constant(
            word_mat, dtype=tf.float32), trainable=False)
        self.char_mat = tf.get_variable(
            "char_mat", initializer=tf.constant(char_mat, dtype=tf.float32))

        self.c_mask = tf.cast(self.c, tf.bool)
        self.q_mask = tf.cast(self.q, tf.bool)
        self.c_len = tf.reduce_sum(tf.cast(self.c_mask, tf.int32), axis=1)
        self.q_len = tf.reduce_sum(tf.cast(self.q_mask, tf.int32), axis=1)

        self.c_maxlen = tf.reduce_max(self.c_len)
        self.q_maxlen = tf.reduce_max(self.q_len)

        self.ch_len = tf.reshape(tf.reduce_sum(
            tf.cast(tf.cast(self.ch, tf.bool), tf.int32), axis=2), [-1])
        self.qh_len = tf.reshape(tf.reduce_sum(
            tf.cast(tf.cast(self.qh, tf.bool), tf.int32), axis=2), [-1])

        self.ready()

    def ready(self):
        N, PL, QL, CL, d, dc, dg = \
            1, self.c_maxlen, self.q_maxlen, Params.char_limit, Params.hidden, Params.char_emb_size, \
            Params.char_hidden
        gru = cudnn_gru if Params.use_cudnn else native_gru

        with tf.variable_scope("emb"):
            with tf.variable_scope("char"):
                ch_emb = tf.reshape(tf.nn.embedding_lookup(
                    self.char_mat, self.ch), [N * PL, CL, dc])
                qh_emb = tf.reshape(tf.nn.embedding_lookup(
                    self.char_mat, self.qh), [N * QL, CL, dc])
                cell_fw = tf.contrib.rnn.GRUCell(dg)
                cell_bw = tf.contrib.rnn.GRUCell(dg)
                _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, ch_emb, self.ch_len, dtype=tf.float32)
                ch_emb = tf.concat([state_fw, state_bw], axis=1)
                _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, qh_emb, self.qh_len, dtype=tf.float32)
                qh_emb = tf.concat([state_fw, state_bw], axis=1)
                qh_emb = tf.reshape(qh_emb, [N, QL, 2 * dg])
                ch_emb = tf.reshape(ch_emb, [N, PL, 2 * dg])

            with tf.name_scope("word"):
                c_emb = tf.nn.embedding_lookup(self.word_mat, self.c)
                q_emb = tf.nn.embedding_lookup(self.word_mat, self.q)

            c_emb = tf.concat([c_emb, ch_emb], axis=2)
            q_emb = tf.concat([q_emb, qh_emb], axis=2)

        with tf.variable_scope("encoding"):
            rnn = gru(num_layers=3, num_units=d, batch_size=N,
                      input_size=c_emb.get_shape().as_list()[-1])
            c = rnn(c_emb, seq_len=self.c_len)
            q = rnn(q_emb, seq_len=self.q_len)

        with tf.variable_scope("attention"):
            qc_att = dot_attention(c, q, mask=self.q_mask, hidden=d)
            rnn = gru(num_layers=1, num_units=d, batch_size=N,
                      input_size=qc_att.get_shape().as_list()[-1])
            att = rnn(qc_att, seq_len=self.c_len)

        with tf.variable_scope("match"):
            self_att = dot_attention(att, att, mask=self.c_mask, hidden=d)
            rnn = gru(num_layers=1, num_units=d, batch_size=N,
                      input_size=self_att.get_shape().as_list()[-1])
            match = rnn(self_att, seq_len=self.c_len)

        with tf.variable_scope("pointer"):
            init = summ(q[:, :, -2 * d:], d, mask=self.q_mask)
            pointer = ptr_net(batch=N, hidden=init.get_shape().as_list()[-1])
            logits1, logits2 = pointer(init, match, d, self.c_mask)

        with tf.variable_scope("predict"):
            outer = tf.matmul(tf.expand_dims(tf.nn.softmax(logits1), axis=2),
                              tf.expand_dims(tf.nn.softmax(logits2), axis=1))
            outer = tf.cond(
                self.tokens_in_context < self.max_answer_tokens,
                lambda: tf.matrix_band_part(outer, 0, -1),
                lambda: tf.matrix_band_part(outer, 0, self.max_answer_tokens))
            self.yp1 = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
            self.yp2 = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)


class Inference(object):

    def __init__(self):
        with open(Params.word_emb_file, "r") as fh:
            self.word_mat = np.array(json.load(fh), dtype=np.float32)
	    #self.word_mat = np.delete(self.word_mat1, slice(91588,91689), axis = 0)
	    print(self.word_mat.shape)
        with open(Params.char_emb_file, "r") as fh:
            self.char_mat = np.array(json.load(fh), dtype=np.float32)
        with open(Params.word2idx_file, "r") as fh:
            self.word2idx_dict = json.load(fh)
        with open(Params.char2idx_file, "r") as fh:
            self.char2idx_dict = json.load(fh)
        #print("hiiiii")
        self.model = InfModel(self.word_mat, self.char_mat)
        #print("helllllooooo")
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        saver = tf.train.Saver()
        #print("-------")
        saver.restore(self.sess, tf.train.latest_checkpoint(Params.save_dir))
        #saver.restore(self.sess, tf.train.import_meta_graph(Params.save_dir, clear_devices=True))
        #print("endeddddd")

    def response(self, context, question):
        sess = self.sess
        model = self.model
        span, context_idxs, ques_idxs, context_char_idxs, ques_char_idxs = \
            self.prepro(context, question)
        yp1, yp2 = \
            sess.run(
                [model.yp1, model.yp2],
                feed_dict={
                    model.c: context_idxs, model.q: ques_idxs,
                    model.ch: context_char_idxs, model.qh: ques_char_idxs,
                    model.tokens_in_context: len(span)})
        start_idx = span[yp1[0]][0]
        end_idx = span[yp2[0]][1]
        return context[start_idx: end_idx]

    def prepro(self, context, question):
        context = context.replace("''", '" ').replace("``", '" ')
        context_tokens = word_tokenize(context)
        context_chars = [list(token) for token in context_tokens]
        spans = convert_idx(context, context_tokens)
        ques = question.replace("''", '" ').replace("``", '" ')
        ques_tokens = word_tokenize(ques)
        ques_chars = [list(token) for token in ques_tokens]

        context_idxs = np.zeros([1, len(context_tokens)], dtype=np.int32)
        context_char_idxs = np.zeros(
            [1, len(context_tokens), Params.char_limit], dtype=np.int32)
        ques_idxs = np.zeros([1, len(ques_tokens)], dtype=np.int32)
        ques_char_idxs = np.zeros(
            [1, len(ques_tokens), Params.char_limit], dtype=np.int32)

        def _get_word(word):
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in self.word2idx_dict:
                    return self.word2idx_dict[each]
            return 1

        def _get_char(char):
            if char in self.char2idx_dict:
                return self.char2idx_dict[char]
            return 1

        for i, token in enumerate(context_tokens):
            context_idxs[0, i] = _get_word(token)

        for i, token in enumerate(ques_tokens):
            ques_idxs[0, i] = _get_word(token)

        for i, token in enumerate(context_chars):
            for j, char in enumerate(token):
                if j == Params.char_limit:
                    break
                context_char_idxs[0, i, j] = _get_char(char)

        for i, token in enumerate(ques_chars):
            for j, char in enumerate(token):
                if j == Params.char_limit:
                    break
                ques_char_idxs[0, i, j] = _get_char(char)
        return spans, context_idxs, ques_idxs, context_char_idxs, ques_char_idxs



@app.route('/answer',methods = ['POST'])
def answer():
    passage = request.json['passage']
    question = request.json['question']
    print(question)
    #os.system("module load tensorflow/python2.7/1.5.0")
    
    response = infer.response(passage, question)
    #response = ""
    print("Answer: {}".format(response))
    # if not passage or not question:
    #     exit()
    #global query, response
    #query = (passage, question)
    #while not response:
    #    sleep(0.1)
    print("received response: {}".format(response))
    Final_response = {"answer": response}
    RESPONSE = json.dumps(Final_response)
    response = []
    return RESPONSE

@app.route('/question',methods = ['POST'])
def question():
    passage = request.json['passage']
    answer = request.json['answer']
    question = request.json['question']
    print(question)
    #os.system("module swap cudnn  cudnn/9.0v7.3.0.29")
    send_list = []
    file = open('data/test_sent_pre.json','w')
    #text_string = '[{"text3":'+ str(answer) + ',"text1":' + str(passage) + ',"text2":' + str(question) + '}]'
    text_string = [{"text3":passage, "text1":answer, "text2":question}]
    json.dump(text_string, file)
    #send_list.append(test_sent_pre_json)
    #test_sent_pre_json = "["+ test_sent_pre_json + "]"
    #print(test_sent_pre_json)
    
    #file.write(test_sent_pre_json)
    file.close()
    qg.question_gen_run(["/scratch/aka398/R-net-new/logs/NP2P.mpqg_5","/scratch/aka398/R-net-new/data/test_sent_pre.json","/scratch/aka398/R-net-new/pred11.txt","beam"])
    #os.system('python src/NP2P_beam_decoder.py --model_prefix /scratch/aka398/R-net-new/logs/NP2P.mpqg_5 --in_path /scratch/aka398/R-net-new/data/test_sent_pre.json --out_path /scratch/aka398/R-net-new/pred11.txt --mode beam')
    with open('pred11.txt') as pred11:
        questions = pred11.readlines()

    print("Answer: {}".format(questions[-1]))
    print("received response: {}".format(questions[-1]))
    Final_response = {"question": questions[-1]}
    RESPONSE = json.dumps(Final_response)
    response = []
    return RESPONSE


if __name__ == '__main__':
    infer = Inference()
    #os.system("module swap cudnn cudnn/9.0v7.3.0.29")
    #os.system("module list")
    app.run(host = '0.0.0.0',port=8010)




