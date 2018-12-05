# -*- coding: utf-8 -*-
#/usr/bin/python2

from __future__ import print_function

import tensorflow as tf
from tqdm import tqdm
from data_load import get_batch, loading_data,get_record_parser, convert_tokens, evaluate, get_batch_dataset, get_dataset
from data_load import cudnn_gru, native_gru, dot_attention, summ, dropout, ptr_net
from params import Params
from layers import *
from GRU import gated_attention_Wrapper, GRUCell, SRUCell
#from evaluate import *
import numpy as np
import cPickle as pickle
from process import *
from demo import Demo

optimizer_factory = {"adadelta":tf.train.AdadeltaOptimizer,
            "adam":tf.train.AdamOptimizer,
            "gradientdescent":tf.train.GradientDescentOptimizer,
            "adagrad":tf.train.AdagradOptimizer}

class Model(object):
    def __init__(self,batch, word_mat = None, char_mat = None, is_training = True, demo = False):
        # Build the computational graph when initializing
        self.is_training = is_training
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            if demo:
                self.passage_w = tf.placeholder(tf.int32,
                                        [1, Params.max_p_len,],"passage_w")
                self.question_w = tf.placeholder(tf.int32,
                                        [1, Params.max_q_len,],"passage_q")
                self.passage_c = tf.placeholder(tf.int32,
                                        [1, Params.max_p_len,Params.max_char_len],"passage_pc")
                self.question_c = tf.placeholder(tf.int32,
                                        [1, Params.max_q_len,Params.max_char_len],"passage_qc")
                self.passage_w_len_ = tf.placeholder(tf.int32,
                                        [1,1],"passage_w_len_")
                self.question_w_len_ = tf.placeholder(tf.int32,
                                        [1,1],"question_w_len_")
                self.passage_c_len = tf.placeholder(tf.int32,
                                        [1, Params.max_p_len],"passage_c_len")
                self.question_c_len = tf.placeholder(tf.int32,
                                        [1, Params.max_q_len],"question_c_len")
                self.data = (self.passage_w,
                            self.question_w,
                            self.passage_c,
                            self.question_c,
                            self.passage_w_len_,
                            self.question_w_len_,
                            self.passage_c_len,
                            self.question_c_len)
            else:
                self.data, self.num_batch = get_batch(is_training = is_training)
                (self.passage_w,
                self.question_w,
                self.passage_c,
                self.question_c,
                self.passage_w_len_,
                self.question_w_len_,
                self.passage_c_len,
                self.question_c_len,
                self.indices) = self.data
                
            self.passage_w_len = tf.squeeze(self.passage_w_len_, -1)
            self.question_w_len = tf.squeeze(self.question_w_len_, -1)

            self.encode_ids()
            self.params = get_attn_params(Params.attn_size, initializer = tf.contrib.layers.xavier_initializer)
            self.attention_match_rnn()
            self.bidirectional_readout()
            self.pointer_network()
            self.outputs()

        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)
        self.c, self.q, self.ch, self.qh, self.y1, self.y2, self.qa_id = batch.get_next()
        self.is_train = tf.get_variable(
            "is_train", shape=[], dtype=tf.bool, trainable=False)
        self.word_mat = tf.get_variable("word_mat", initializer=tf.constant(
            word_mat, dtype=tf.float32), trainable=False)
        self.char_mat = tf.get_variable(
            "char_mat", initializer=tf.constant(char_mat, dtype=tf.float32))

        self.c_mask = tf.cast(self.c, tf.bool)
        self.q_mask = tf.cast(self.q, tf.bool)
        self.c_len = tf.reduce_sum(tf.cast(self.c_mask, tf.int32), axis=1)
        self.q_len = tf.reduce_sum(tf.cast(self.q_mask, tf.int32), axis=1)

        N, CL = Params.batch_size, Params.char_limit
        self.c_maxlen = tf.reduce_max(self.c_len)
        self.q_maxlen = tf.reduce_max(self.q_len)
        self.c = tf.slice(self.c, [0, 0], [N, self.c_maxlen])
        self.q = tf.slice(self.q, [0, 0], [N, self.q_maxlen])
        self.c_mask = tf.slice(self.c_mask, [0, 0], [N, self.c_maxlen])
        self.q_mask = tf.slice(self.q_mask, [0, 0], [N, self.q_maxlen])
        self.ch = tf.slice(self.ch, [0, 0, 0], [N, self.c_maxlen, CL])
        self.qh = tf.slice(self.qh, [0, 0, 0], [N, self.q_maxlen, CL])
        self.y1 = tf.slice(self.y1, [0, 0], [N, self.c_maxlen])
        self.y2 = tf.slice(self.y2, [0, 0], [N, self.c_maxlen])


        self.ch_len = tf.reshape(tf.reduce_sum(
            tf.cast(tf.cast(self.ch, tf.bool), tf.int32), axis=2), [-1])
        self.qh_len = tf.reshape(tf.reduce_sum(
            tf.cast(tf.cast(self.qh, tf.bool), tf.int32), axis=2), [-1])

        self.ready()

        if is_training:
            self.lr = tf.get_variable(
                "lr", shape=[], dtype=tf.float32, trainable=False)
            self.opt = tf.train.AdadeltaOptimizer(
                learning_rate=self.lr, epsilon=1e-6)
            grads = self.opt.compute_gradients(self.loss)
            gradients, variables = zip(*grads)
            capped_grads, _ = tf.clip_by_global_norm(
                gradients, Params.grad_clip)
            self.train_op = self.opt.apply_gradients(
                zip(capped_grads, variables), global_step=self.global_step)

    def encode_ids(self):
        with tf.device('/cpu:0'):
            self.char_embeddings = tf.Variable(tf.constant(0.0, shape=[Params.char_vocab_size, Params.char_emb_size]),
                                               trainable=True, name="char_embeddings")
            self.word_embeddings = tf.Variable(tf.constant(0.0, shape=[Params.vocab_size, Params.emb_size]),
                                               trainable=False, name="word_embeddings")
            self.word_embeddings_placeholder = tf.placeholder(tf.float32, [Params.vocab_size, Params.emb_size],
                                                              "word_embeddings_placeholder")
            self.emb_assign = tf.assign(self.word_embeddings, self.word_embeddings_placeholder)

        # Embed the question and passage information for word and character tokens
        self.passage_word_encoded, self.passage_char_encoded = encoding(self.passage_w,
                                                                        self.passage_c,
                                                                        word_embeddings=self.word_embeddings,
                                                                        char_embeddings=self.char_embeddings,
                                                                        scope="passage_embeddings")
        self.question_word_encoded, self.question_char_encoded = encoding(self.question_w,
                                                                          self.question_c,
                                                                          word_embeddings=self.word_embeddings,
                                                                          char_embeddings=self.char_embeddings,
                                                                          scope="question_embeddings")

        self.passage_char_encoded = bidirectional_GRU(self.passage_char_encoded,
                                                      self.passage_c_len,
                                                      cell_fn=SRUCell if Params.SRU else GRUCell,
                                                      scope="passage_char_encoding",
                                                      output=1,
                                                      is_training=self.is_training)
        self.question_char_encoded = bidirectional_GRU(self.question_char_encoded,
                                                       self.question_c_len,
                                                       cell_fn=SRUCell if Params.SRU else GRUCell,
                                                       scope="question_char_encoding",
                                                       output=1,
                                                       is_training=self.is_training)
        self.passage_encoding = tf.concat((self.passage_word_encoded, self.passage_char_encoded), axis=2)
        self.question_encoding = tf.concat((self.question_word_encoded, self.question_char_encoded), axis=2)

        # Passage and question encoding
        # cell = [MultiRNNCell([GRUCell(Params.attn_size, is_training = self.is_training) for _ in range(3)]) for _ in range(2)]
        self.passage_encoding = bidirectional_GRU(self.passage_encoding,
                                                  self.passage_w_len,
                                                  cell_fn=SRUCell if Params.SRU else GRUCell,
                                                  layers=Params.num_layers,
                                                  scope="passage_encoding",
                                                  output=0,
                                                  is_training=self.is_training)
        # cell = [MultiRNNCell([GRUCell(Params.attn_size, is_training = self.is_training) for _ in range(3)]) for _ in range(2)]
        self.question_encoding = bidirectional_GRU(self.question_encoding,
                                                   self.question_w_len,
                                                   cell_fn=SRUCell if Params.SRU else GRUCell,
                                                   layers=Params.num_layers,
                                                   scope="question_encoding",
                                                   output=0,
                                                   is_training=self.is_training)

    def ready(self):

        N, PL, QL, CL, d, dc, dg = Params.batch_size, self.c_maxlen, self.q_maxlen, Params.char_limit, Params.hidden, Params.char_emb_size, Params.char_hidden
        with tf.device('/cpu:0'):
            gru = cudnn_gru if Params.use_cudnn else native_gru

            with tf.variable_scope("emb"):
                with tf.variable_scope("char"):
                    ch_emb = tf.reshape(tf.nn.embedding_lookup(
                        self.char_mat, self.ch), [N * PL, CL, dc])
                    qh_emb = tf.reshape(tf.nn.embedding_lookup(
                        self.char_mat, self.qh), [N * QL, CL, dc])
                    ch_emb = dropout(
                        ch_emb, keep_prob=Params.keep_prob, is_train=self.is_train)
                    qh_emb = dropout(
                        qh_emb, keep_prob=Params.keep_prob, is_train=self.is_train)
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
            rnn = gru(num_layers=3, num_units=d, batch_size=N, input_size=c_emb.get_shape(
            ).as_list()[-1], keep_prob=Params.keep_prob, is_train=self.is_train)
            c = rnn(c_emb, seq_len=self.c_len)
            q = rnn(q_emb, seq_len=self.q_len)

        with tf.variable_scope("attention"):
            qc_att = dot_attention(c, q, mask=self.q_mask, hidden=d,
                                   keep_prob=Params.keep_prob, is_train=self.is_train)
            rnn = gru(num_layers=1, num_units=d, batch_size=N, input_size=qc_att.get_shape(
            ).as_list()[-1], keep_prob=Params.keep_prob, is_train=self.is_train)
            att = rnn(qc_att, seq_len=self.c_len)

        with tf.variable_scope("match"):
            self_att = dot_attention(
                att, att, mask=self.c_mask, hidden=d, keep_prob=Params.keep_prob, is_train=self.is_train)
            rnn = gru(num_layers=1, num_units=d, batch_size=N, input_size=self_att.get_shape(
            ).as_list()[-1], keep_prob=Params.keep_prob, is_train=self.is_train)
            match = rnn(self_att, seq_len=self.c_len)

        with tf.variable_scope("pointer"):
            init = summ(q[:, :, -2 * d:], d, mask=self.q_mask,
                        keep_prob=Params.ptr_keep_prob, is_train=self.is_train)
            pointer = ptr_net(batch=N, hidden=init.get_shape().as_list(
            )[-1], keep_prob=Params.ptr_keep_prob, is_train=self.is_train)
            logits1, logits2 = pointer(init, match, d, self.c_mask)

        with tf.variable_scope("predict"):
            outer = tf.matmul(tf.expand_dims(tf.nn.softmax(logits1), axis=2),
                              tf.expand_dims(tf.nn.softmax(logits2), axis=1))
            outer = tf.matrix_band_part(outer, 0, 15)
            self.yp1 = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
            self.yp2 = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=logits1, labels=tf.stop_gradient(self.y1))
            losses2 = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=logits2, labels=tf.stop_gradient(self.y2))
            self.loss = tf.reduce_mean(losses + losses2)

    def attention_match_rnn(self):
        # Apply gated attention recurrent network for both query-passage matching and self matching networks
        with tf.variable_scope("attention_match_rnn"):
            memory = self.question_encoding
            inputs = self.passage_encoding
            scopes = ["question_passage_matching", "self_matching"]
            params = [(([self.params["W_u_Q"],
                         self.params["W_u_P"],
                         self.params["W_v_P"]], self.params["v"]),
                       self.params["W_g"]),
                      (([self.params["W_v_P_2"],
                         self.params["W_v_Phat"]], self.params["v"]),
                       self.params["W_g"])]
            for i in range(2):
                args = {"num_units": Params.attn_size,
                        "memory": memory,
                        "params": params[i],
                        "self_matching": False if i == 0 else True,
                        "memory_len": self.question_w_len if i == 0 else self.passage_w_len,
                        "is_training": self.is_training,
                        "use_SRU": Params.SRU}
                cell = [
                    apply_dropout(gated_attention_Wrapper(**args), size=inputs.shape[-1], is_training=self.is_training)
                    for _ in range(2)]
                inputs = attention_rnn(inputs,
                                       self.passage_w_len,
                                       Params.attn_size,
                                       cell,
                                       scope=scopes[i])
                memory = inputs  # self matching (attention over itself)
            self.self_matching_output = inputs

    def bidirectional_readout(self):
        self.final_bidirectional_outputs = bidirectional_GRU(self.self_matching_output,
                                                             self.passage_w_len,
                                                             cell_fn=SRUCell if Params.SRU else GRUCell,
                                                             # layers = Params.num_layers, # or 1? not specified in the original paper
                                                             scope="bidirectional_readout",
                                                             output=0,
                                                             is_training=self.is_training)

    def pointer_network(self):
        params = (([self.params["W_u_Q"], self.params["W_v_Q"]], self.params["v"]),
                  ([self.params["W_h_P"], self.params["W_h_a"]], self.params["v"]))
        cell = apply_dropout(GRUCell(Params.attn_size * 2), size=self.final_bidirectional_outputs.shape[-1],
                             is_training=self.is_training)
        self.points_logits = pointer_net(self.final_bidirectional_outputs, self.passage_w_len, self.question_encoding,
                                         self.question_w_len, cell, params, scope="pointer_network")

    def outputs(self):
        self.logit_1, self.logit_2 = tf.split(self.points_logits, 2, axis=1)
        self.logit_1 = tf.transpose(self.logit_1, [0, 2, 1])
        self.dp = tf.matmul(self.logit_1, self.logit_2)
        self.dp = tf.matrix_band_part(self.dp, 0, 15)
        self.output_index_1 = tf.argmax(tf.reduce_max(self.dp, axis=2), -1)
        self.output_index_2 = tf.argmax(tf.reduce_max(self.dp, axis=1), -1)
        self.output_index = tf.stack([self.output_index_1, self.output_index_2], axis=1)
        # self.output_index = tf.argmax(self.points_logits, axis = 2)



    def loss_function(self):
        return self.loss

    def get_global_step(self):
        return self.global_step


    def summary(self):
        self.F1 = tf.Variable(tf.constant(0.0, shape=(), dtype = tf.float32),trainable=False, name="F1")
        self.F1_placeholder = tf.placeholder(tf.float32, shape = (), name = "F1_placeholder")
        self.EM = tf.Variable(tf.constant(0.0, shape=(), dtype = tf.float32),trainable=False, name="EM")
        self.EM_placeholder = tf.placeholder(tf.float32, shape = (), name = "EM_placeholder")
        self.dev_loss = tf.Variable(tf.constant(5.0, shape=(), dtype = tf.float32),trainable=False, name="dev_loss")
        self.dev_loss_placeholder = tf.placeholder(tf.float32, shape = (), name = "dev_loss")
        self.metric_assign = tf.group(tf.assign(self.F1, self.F1_placeholder),tf.assign(self.EM, self.EM_placeholder),tf.assign(self.dev_loss, self.dev_loss_placeholder))
        tf.summary.scalar('loss_training', self.loss)
        tf.summary.scalar('loss_dev', self.dev_loss)
        tf.summary.scalar("F1_Score",self.F1)
        tf.summary.scalar("Exact_Match",self.EM)
        tf.summary.scalar('learning_rate', Params.opt_arg[Params.optimizer]['learning_rate'])
        self.merged = tf.summary.merge_all()

def debug():
    print("Under debugging mode...")
    #model = Model(is_training = False)
    #print("Built model")

def test():
    word_mat, char_mat, train_eval_file, eval_file, meta = loading_data(Params.word_emb_file, Params.char_emb_file,
                                                                            Params.train_eval_file,
                                                                            Params.dev_eval_file, Params.dev_meta)
    total = meta["total"]

    print("Loading model...")
    test_batch = get_dataset(Params.test_record_file, get_record_parser(
        Params, is_test=True), Params).make_one_shot_iterator()

    model = Model(test_batch, word_mat, char_mat, is_training=False)

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(Params.save_dir))
        sess.run(tf.assign(model.is_train, tf.constant(False, dtype=tf.bool)))
        losses = []
        answer_dict = {}
        remapped_dict = {}
        for step in tqdm(range(total // Params.batch_size + 1)):
            qa_id, loss, yp1, yp2 = sess.run(
                [model.qa_id, model.loss, model.yp1, model.yp2])
            answer_dict_, remapped_dict_ = convert_tokens(
                eval_file, qa_id.tolist(), yp1.tolist(), yp2.tolist())
            answer_dict.update(answer_dict_)
            remapped_dict.update(remapped_dict_)
            losses.append(loss)
        loss = np.mean(losses)
        metrics = evaluate(eval_file, answer_dict)
        with open(Params.answer_file, "w") as fh:
            json.dump(remapped_dict, fh)
        print("Exact Match: {}, F1: {}".format(
            metrics['exact_match'], metrics['f1']))

def evaluate_batch(model, num_batches, eval_file, sess, data_type, handle, str_handle):
    answer_dict = {}
    losses = []
    for _ in tqdm(range(1, num_batches + 1)):
        qa_id, loss, yp1, yp2, = sess.run(
            [model.qa_id, model.loss, model.yp1, model.yp2], feed_dict={handle: str_handle})
        answer_dict_, _ = convert_tokens(
            eval_file, qa_id.tolist(), yp1.tolist(), yp2.tolist())
        answer_dict.update(answer_dict_)
        losses.append(loss)
    loss = np.mean(losses)
    metrics = evaluate(eval_file, answer_dict)
    metrics["loss"] = loss
    loss_sum = tf.Summary(value=[tf.Summary.Value(
        tag="{}/loss".format(data_type), simple_value=metrics["loss"]), ])
    f1_sum = tf.Summary(value=[tf.Summary.Value(
        tag="{}/f1".format(data_type), simple_value=metrics["f1"]), ])
    em_sum = tf.Summary(value=[tf.Summary.Value(
        tag="{}/em".format(data_type), simple_value=metrics["exact_match"]), ])
    return metrics, [loss_sum, f1_sum, em_sum]


def main():
    word_mat, char_mat, train_eval_file, dev_eval_file, meta  = loading_data(Params.word_emb_file, Params.char_emb_file, Params.train_eval_file, Params.dev_eval_file, Params.dev_meta)
    dev_total = meta["total"]

    print("Building model...")

    parser = get_record_parser(Params)
    train_dataset = get_batch_dataset(Params.train_record_file, parser, Params)
    dev_dataset = get_dataset(Params.dev_record_file, parser, Params)
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, train_dataset.output_types, train_dataset.output_shapes)
    train_iterator = train_dataset.make_one_shot_iterator()
    dev_iterator = dev_dataset.make_one_shot_iterator()

    model = Model(iterator, word_mat, char_mat, is_training = True)

    print("Built model")

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    loss_save = 100.0
    patience = 0
    lr = Params.init_lr

    with tf.Session(config=sess_config) as sess:
        writer = tf.summary.FileWriter(Params.log_dir)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        train_handle = sess.run(train_iterator.string_handle())
        dev_handle = sess.run(dev_iterator.string_handle())
        sess.run(tf.assign(model.is_train, tf.constant(True, dtype=tf.bool)))
        sess.run(tf.assign(model.lr, tf.constant(lr, dtype=tf.float32)))

        for _ in tqdm(range(1, Params.num_steps + 1)):
            global_step = sess.run(model.global_step) + 1
            loss, train_op = sess.run([model.loss, model.train_op], feed_dict={
                handle: train_handle})
            if global_step % Params.period == 0:
                loss_sum = tf.Summary(value=[tf.Summary.Value(
                    tag="model/loss", simple_value=loss), ])
                writer.add_summary(loss_sum, global_step)
            if global_step % Params.checkpoint == 0:
                sess.run(tf.assign(model.is_train,
                                   tf.constant(False, dtype=tf.bool)))
                trainMetrics, summ = evaluate_batch(
                    model, Params.val_num_batches, train_eval_file, sess, "train", handle, train_handle)
                for s in summ:
                    writer.add_summary(s, global_step)

                metrics, summ = evaluate_batch(
                    model, dev_total // Params.batch_size + 1, dev_eval_file, sess, "dev", handle, dev_handle)
                sess.run(tf.assign(model.is_train,
                                   tf.constant(True, dtype=tf.bool)))

                dev_loss = metrics["loss"]
                print("Train Loss: {}, Exact Match: {}, F1: {}".format(
                    trainMetrics['loss'], trainMetrics['exact_match'], trainMetrics['f1']))
                print("Dev Loss: {}, Exact Match: {}, F1: {}".format(
                    metrics['loss'], metrics['exact_match'], metrics['f1']))
                if dev_loss < loss_save:
                    loss_save = dev_loss
                    patience = 0
                else:
                    patience += 1
                if patience >= Params.patience:
                    lr /= 2.0
                    loss_save = dev_loss
                    patience = 0
                sess.run(tf.assign(model.lr, tf.constant(lr, dtype=tf.float32)))
                for s in summ:
                    writer.add_summary(s, global_step)
                writer.flush()
                filename = os.path.join(
                    Params.save_dir, "model_{}.ckpt".format(global_step))
                saver.save(sess, filename)

if __name__ == '__main__':
    if Params.mode.lower() == "debug":
        print("Debugging...")
        debug()
    elif Params.mode.lower() == "test":
        print("Testing on dev set...")
        test()
    #elif Params.mode.lower() == "demo":
    #    print("Run the local host for online demo...")
    #    model = Model(is_training = False, demo = True); print("Built model")
    #    demo_run = Demo(model)
    elif Params.mode.lower() == "train":
        print("Training...")
        main()
    else:
        print("Invalid mode.")
