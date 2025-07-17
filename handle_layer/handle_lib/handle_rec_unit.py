#!/usr/bin/python
# -*- coding:utf-8 -*-

import tensorflow as tf

from handle_layer.handle_lib.handle_base import InputBase
from data.data_utils import index_of_tensor
import numpy as np
def bucketization_fn(x):
    max_int_64 = 9223372036854775800
    x = tf.math.abs(x)
    x = tf.clip_by_value(x, 1, max_int_64)
    x = tf.cast(x, tf.float32)
    x = tf.math.log(x) / 0.301
    x = tf.cast(x, tf.int64)
    return

def create_padding_mark(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, np.newaxis, np.newaxis, :]

def create_look_ahead_mark(size):
    mark = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mark

def create_mask(inpus):
    look_ahead_mask = create_look_ahead_mark(tf.shape(inpus)[-1])
    decode_targets_padding_mask = create_padding_mark(inpus)
    combine_mask = tf.math.maximum(decode_targets_padding_mask, look_ahead_mask)
    return combine_mask

class RMSNorm(object):
    def __init__(self, epsilon=1e-6, variable_scope=""):
        self.eps = epsilon
        self.variable_scope = variable_scope

    def __call__(self, x):
        with tf.variable_scope(self.variable_scope, reuse=tf.AUTO_REUSE):
            b, n, d = x.get_shape().as_list()
            gamma = tf.Variable(tf.ones([d]), name="gamma")
            rms = tf.sqrt(tf.reduce_mean(tf.square(x), axis=-1, keep_dims=True))
        return gamma * x / (rms + self.eps)

class SwiGLU(object):
    def __init__(self, d_model, variable_scope=""):
        self.d_model = d_model
        self.variable_scope = variable_scope
        self.ffn_factor = 3
        self.activation = "relu"

        self.fc1_layer = tf.layers.Dense(units=self.d_model * self.ffn_factor, activation=None,
                                    name=self.variable_scope + 'fc1')
        self.fc2_layer = tf.layers.Dense(units=self.d_model * self.ffn_factor, activation=None,
                                         name=self.variable_scope + 'fc2')
        self.fc3_layer = tf.layers.Dense(units=self.d_model, activation=None,
                                         name=self.variable_scope + 'fc3')

    def silu(self, x):
        return x * tf.sigmoid(x)
    def __call__(self, x):
        with tf.variable_scope(self.variable_scope, reuse=tf.AUTO_REUSE):
            fc1 = self.silu(self.fc1_layer(x))
            fc2 = self.fc2_layer(x)
            fc3 = self.fc3_layer(fc1*fc2)
            return fc3
class Relative_positional_encoding(object):
    def __init__(self, seq_len, num_buckets, variable_scope, N, all_timestamps):
        self._pos_w = tf.Variable(tf.random.normal([seq_len * 2 + 1], mean=0, stddev=0.02),
                                  name=variable_scope + "_pos_w")
        self._ts_w = tf.Variable(tf.random.normal([num_buckets + 1], mean=0, stddev=0.02),
                                 name=variable_scope + "_ts_w")
        self.num_buckets = num_buckets

        self.N = N
        t = tf.pad(self._pos_w[:2 * N - 1], [[0, N]])
        t = tf.tile(t, [N])
        t = tf.reshape(t[..., :-N], [1, N, 3 * N - 2])
        r = (2 * N - 1) / 2
        self.rel_pos_bias = t[:, :, r:-r]

        ts_expanded = tf.expand_dims(all_timestamps, axis=-1)
        ts_tiled = tf.tile(ts_expanded, [1, 1, self.N])
        time_diff = ts_tiled - tf.transpose(ts_tiled, [0, 2, 1])
        time_diff = tf.abs(time_diff)


        bucketed_timestamps = tf.clip_by_value(
            bucketization_fn(time_diff),
            clip_value_min=0, clip_value_max=self.num_buckets)
        tf.summary.histogram("buc_ts", bucketed_timestamps)
        bucketed_timestamps = tf.stop_gradient(bucketed_timestamps)
        self.rel_ts_bias = tf.gather(self._ts_w, indices=bucketed_timestamps)

    def relative_attention_bias(self):
        return self.rel_pos_bias + self.rel_ts_bias
class MultiHeadSelfAttention(object):
    def __init__(self, d_model, heads, variable_scope):
        self.d_model = d_model
        self.num_heads = heads
        self.variable_scope = variable_scope
        assert 0 == d_model % heads
        self.depth = d_model // heads

    def silu(self, x):
        return x * tf.sigmoid(x)
    def __call__(self, x, mask, ts, ts_encoder):
        with tf.variable_scope(self.variable_scope, reuse=tf.AUTO_REUSE):
            b, n, d = x.get_shape().as_list()
            h, dh = self.num_heads, self.depth

            q = tf.layers.dense(x, self.d_model, activation=None, use_bias=False,
                                   name="q_linear")
            k = tf.layers.dense(x, self.d_model, activation=None, use_bias=False,
                                name="k_linear")
            v = tf.layers.dense(x, self.d_model, activation=None, use_bias=False,
                                name="v_linear")
            q = tf.transpose(tf.reshape(q, [-1, n, h, dh]), [0, 2, 1, 3])
            k = tf.transpose(tf.reshape(k, [-1, n, h, dh]), [0, 2, 3, 1])
            v = tf.transpose(tf.reshape(v, [-1, n, h, dh]), [0, 2, 1, 3])

            attn = tf.matmul(q, k)
            dk = tf.cast(tf.shape(k)[-2], tf.float32)
            attn = attn / tf.math.sqrt(dk)

            ts_rab = tf.expand_dims(ts_encoder.relative_attention_bias(ts), axis=1)

            attn = attn + ts_rab

            if mask is not None:
                attn += (mask * -1e9)
            attn = tf.nn.softmax(attn, axis=-1)
            attn_out = tf.matmul(attn, v)
            attn_out = tf.reshape(tf.transpose(attn_out, [0, 2, 1, 3]), [-1, n, d])
            attn_out = tf.layers.dense(attn_out, self.d_model, activation=None, use_bias=False, name="o_linear")
            return attn_out

class MultiHeadCrossAttention(object):
    def __init__(self, d_model, heads, variable_scope):
        self.d_model = d_model
        self.num_heads = heads
        self.variable_scope = variable_scope
        assert 0 == d_model % heads
        self.depth = d_model // heads

    def silu(self, x):
        return x * tf.sigmoid(x)

    def __call__(self, seq1, seq2):
        with tf.variable_scope(self.variable_scope + "cross", reuse=tf.AUTO_REUSE):
            self.enable_softmax = True
            b, n1, d1 = seq1.get_shape().as_list()
            b, n2, d2 = seq2.get_shape().as_list()
            h, dh = self.num_heads, self.depth
            q1 = tf.layers.dense(seq1, self.d_model, activation=None, use_bias=False,
                                    name="q1_linear1")
            k2 = tf.layers.dense(seq2, self.d_model, activation=None, use_bias=False,
                                   name="k2_linear1")
            v2 = tf.layers.dense(seq2, self.d_model, activation=None, use_bias=False,
                                  name="v2_linear1")

            q1 = tf.transpose(tf.reshape(q1, [-1, n1, h, dh]), [0, 2, 1, 3])
            k2 = tf.transpose(tf.reshape(k2, [-1, n2, h, dh]), [0, 2, 3, 1])
            v2 = tf.transpose(tf.reshape(v2, [-1, n2, h, dh]), [0, 2, 1, 3])

            attn1 = tf.matmul(q1, k2)
            dk = tf.cast(tf.shape(k2)[-1], tf.float32)
            attn1 = attn1 / tf.math.sqrt(dk)
            attn1 = tf.nn.softmax(attn1, axis=-1)

            attn_out1 = tf.matmul(attn1, v2)
            attn_out1 = tf.reshape(tf.transpose(attn_out1, [0, 2, 1, 3]), [-1, n1, d1])
            attn_out1 = tf.layers.dense(attn_out1, self.d_model, activation=None, use_bias=False, name="o_linear1")
            return attn_out1

class SUAN_Block(object):
    def __init__(self, d_model, head, droupout_rate, variable_scope):
        super(SUAN_Block, self).__init__()
        self.attention1 = MultiHeadSelfAttention(d_model, head, variable_scope+"_selfAttention")
        self.cross_attention = MultiHeadCrossAttention(d_model, head, variable_scope + "_crossAttention")
        self.ffn1 = SwiGLU(d_model, variable_scope + "_SwiGLU")

        self.norm1 = RMSNorm(1e-6, variable_scope + "_norm_1")
        self.norm2 = RMSNorm(1e-6, variable_scope + "_norm_2")
        self.norm3 = RMSNorm(1e-6, variable_scope + "_norm_3")

        self.dropout1 = tf.keras.layers.Dropout(droupout_rate)
        self.dropout2 = tf.keras.layers.Dropout(droupout_rate)
        self.dropout3 = tf.keras.layers.Dropout(droupout_rate)

        self.variable_scope = variable_scope

    def MS_SeNet(self, x1, x2, name):
        b, n, d = x1.get_shape().as_list()
        stack = tf.stack([x1, x2], 0)
        sum = x1 + x2
        sum = tf.reduce_mean(sum, axis=1, keepdims=True)
        sum = tf.layers.dense(sum, d/4, activation='relu', use_bias=False, name=name+"_se_net_relu")
        weight1 = tf.layers.dense(sum, d, activation=None, use_bias=False, name=name+"_se_net_weight1")
        weight2 = tf.layers.dense(sum, d, activation=None, use_bias=False, name=name+"_se_net_weight2")
        weight = tf.stack([weight1, weight2], 0)
        weight = tf.nn.softmax(weight, axis=0)
        stack = tf.reduce_sum(stack * weight, 0)
        return stack

    def __call__(self, x, userseq, mask, ts, ts_encoder, training):
        x = self.norm1(x)
        # selfAttention
        x_output = self.attention1(x, mask, ts, ts_encoder)
        x_output = self.dropout1(x_output, training=training)
        x = self.norm2(x_output + x)
        # crossAttention

        cross_x1 = self.cross_attention(x, userseq)
        x_cross = self.MS_SeNet(x, cross_x1, self.variable_scope + "x")
        x_cross = self.dropout2(x_cross, training=training)
        x = self.norm3(x_cross + x)
        # ffn
        ffn_x_output = self.ffn1(x)
        ffn_x_output = self.dropout3(ffn_x_output, training=training)
        ffn_x_output = ffn_x_output + x

        return ffn_x_output, userseq


class Mix1k_SUAN(InputBase):
    def __init__(self, data_struct, params, is_training, hashtable, ptable_lookup):
        super(Mix1k_SUAN, self).__init__(data_struct, params, is_training, hashtable, ptable_lookup)
        self.d_model = self.params['embed_dim']
        self.seq_len = self.params['seq_len']
        self.n_layers = self.params['n_layers']
        self.n_heads = self.params['n_heads']
        self.dropout_rate = self.params['dropout_rate']
        self.num_buckets = self.params['num_buckets']
        self.feature_num = self.params['feature_num']

        self.user_cate_feat_num = self.params['user_cate_feat_num']
        self.user_dense_feat_num = self.params['user_dense_feat_num']
        self.target_context_cate_feat_num = self.params['target_context_cate_feat_num']
        self.target_context_dense_feat_num = self.params['target_context_dense_feat_num']

        self.id_base = ["id_base"]
        self.target_poi = ["target_poi"]
        self.target_third_id = ["target_third_id"]
        self.target_ts = ["target_ts"]
        self.poi_list = ["poi_list[%d]" % _ for _ in range(self.seq_len)]
        self.action_list = ["action_list[%d]" % _ for _ in range(self.seq_len)]
        self.ts_list = ["ts_list[%d]" % _ for _ in range(self.seq_len)]
        self.third_id_list = ["third_id_list[%d]" % _ for _ in range(self.seq_len)]
        self.aor_id_list = ["aor_id_list[%d]" % _ for _ in range(self.seq_len)]
        self.imp_pos_list = ["imp_pos_list[%d]" % _ for _ in range(self.seq_len)]

        self.cate_feat_list = [
            # User_Category
            "user_category_feature1", "user_category_feature2", "user_category_feature3",
            # Target_Context_Category
            "target_context_category_feature1", "target_context_category_feature2", "target_context_category_feature3"
        ]
        self.cat_list = [self.id_base, self.target_poi, self.target_third_id, self.target_ts,
                         self.poi_list, self.action_list, self.third_id_list, self.ts_list, self.aor_id_list,
                         self.imp_pos_list,
                         self.cate_feat_list]
        self.dense_feat_list = [
            # User_Dense
            "user_dense_feature1", "user_dense_feature2", "user_dense_feature3",
            # Target_Context_Dense
            "target_context_dense_feature1", "target_context_dense_feature2", "target_context_dense_feature3"

        ]
        self.dense_list = [self.dense_feat_list]


        self.transformer_blocks1 = [
            SUAN_Block(self.d_model * self.feature_num, self.n_heads, self.dropout_rate, "GenRecTransformerBlock2_{}".format(i)) for i in
            range(self.n_layers)]


        self.pos = tf.Variable(tf.random.normal([self.seq_len + 1, self.d_model * self.feature_num], mean=0, stddev=0.02), name="pos")
    def recieve_gather_features(self, cat_feas, dense_feas):
        # cat_feas
        self.cat_fea_split = cat_feas

        self.id_base, self.target_poi, self.target_third_id, self.target_ts, \
            self.poi_list, self.action_list, self.third_id_list, self.ts_list, self.aor_id_list, self.imp_pos_list = self.cat_fea_split[:10]

        self.cate_fea_emb = self.ptable_lookup(list_ids=cat_feas[10:], v_name=self.__class__.__name__)[0]
        self.dense_fea_split = dense_feas[0]

        self.target_poi = tf.subtract(self.target_poi, self.id_base)
        self.target_third_id = tf.subtract(self.target_third_id, self.id_base)
        self.poi_list = tf.subtract(self.poi_list, self.id_base)
        self.action_list = tf.subtract(self.action_list, self.id_base)
        self.third_id_list = tf.subtract(self.third_id_list, self.id_base)

        self.reverse_poi_list = tf.reverse(self.poi_list, [-1])
        self.reverse_action_list = tf.reverse(self.action_list, [-1])
        self.reverse_third_id_list = tf.reverse(self.third_id_list, [-1])
        self.reverse_ts_list = tf.reverse(self.ts_list, [-1])
        self.reverse_aor_id_list = tf.reverse(self.aor_id_list, [-1])
        self.reverse_imp_pos_list = tf.reverse(self.imp_pos_list, [-1])

        self.mix_list = tf.concat([self.reverse_poi_list, self.target_poi], -1)

        self.cat_fea_split = [self.reverse_poi_list, self.reverse_action_list, self.reverse_third_id_list,
                              self.reverse_ts_list, self.reverse_aor_id_list, self.reverse_imp_pos_list,
                              self.target_poi, self.target_third_id, self.target_ts]
        self.cat_fea_emb = self.ptable_lookup(list_ids=self.cat_fea_split, v_name=self.__class__.__name__)

    def __call__(self, cat_features, dense_features, auxiliary_info, embed_dim, se_block):
        user_cate, user_dense, target_context_cate, target_context_dense = self.get_cate_dense()

        self.reverse_poi_emb, self.reverse_action_emb, self.reverse_third_id_emb, \
            self.reverse_ts_emb, self.reverse_aor_id_emb, self.reverse_imp_pos_emb, \
            self.target_poi_emb, self.target_third_id_emb, self.target_ts_emb = self.cat_fea_emb

        behavior_input_tokens = tf.concat(
            [self.reverse_poi_emb, self.reverse_third_id_emb, self.reverse_aor_id_emb, self.reverse_imp_pos_emb,
             self.reverse_action_emb], -1)
        target_tokens = tf.concat([self.target_poi_emb, self.target_third_id_emb], -1)
        zeros = tf.zeros_like(target_tokens)
        target_tokens = tf.concat([target_tokens, zeros, zeros[:, :, :self.d_model]], -1)

        input_tokens = tf.concat([behavior_input_tokens, target_tokens], 1)

        pos_time_emb = tf.concat([self.reverse_ts_emb, self.target_ts_emb], 1)
        pos_time_emb = tf.layers.dense(pos_time_emb, self.d_model * self.feature_num, activation=None)
        input_tokens += pos_time_emb
        input_tokens += self.pos

        mask = create_mask(self.mix_list)
        # time
        ts = tf.concat([self.reverse_ts_list, self.target_ts], axis=-1)
        self.ts_encoder = Relative_positional_encoding(self.seq_len + 1, self.num_buckets,
                                                             "relative_user_positional", self.seq_len + 1, ts)
        variable_scope = "GenRecTransformer1"
        with tf.variable_scope(variable_scope, reuse=tf.AUTO_REUSE):
            hidden_state = input_tokens
            for block in self.transformer_blocks1:
                hidden_state, user_cate = block(hidden_state, user_cate, mask, ts, self.ts_encoder, self.is_training)
            transformer_output = hidden_state
        last_action_pre = tf.concat([transformer_output[:, -1, :], tf.reshape(user_cate, [-1, self.user_cate_feat_num*self.d_model]), tf.reshape(target_context_cate, [-1, self.target_context_cate_feat_num*self.d_model]), user_dense, target_context_dense], -1)

        return last_action_pre



    def get_cate_dense(self):
        embed_dim = self.d_model
        # cate
        cate_feat_gather = tf.reshape(self.cate_fea_emb, [-1, embed_dim * len(self.cate_feat_list)])
        cate = cate_feat_gather
        user_cate = cate[:, :len(self.user_cate_feat_num) * embed_dim]
        target_context_cate = cate[:, len(self.user_cate_feat_num) * embed_dim:]

        # dense
        dense_feat_size = len(self.dense_feat_list)
        dense_feat = tf.concat(self.dense_fea_split, -1)
        zeros_tensor = tf.zeros([1, dense_feat_size], tf.float32)
        imputation_w1 = tf.get_variable('imputation_w1', shape=[1, dense_feat_size],
                                        initializer=tf.random_normal_initializer())
        imputation_w2 = tf.get_variable('imputation_w2', shape=[1, dense_feat_size],
                                        initializer=tf.random_normal_initializer())
        dense_fea_col = tf.multiply(tf.cast(tf.math.equal(dense_feat, zeros_tensor), tf.float32),
                                    imputation_w1) + \
                        tf.multiply(tf.where(tf.math.equal(dense_feat, zeros_tensor),
                                             tf.zeros_like(dense_feat), dense_feat),
                                    imputation_w2)
        normalized_numerical_fea_col = tf.layers.batch_normalization(inputs=dense_fea_col,
                                                                     training=self.is_training,
                                                                     name='num_fea_batch_norm')
        normed_dense_feat = tf.nn.tanh(normalized_numerical_fea_col)

        dense = normed_dense_feat
        user_dense = dense[:, :len(self.user_dense_feat_num)]
        target_context_dense = dense[:, len(self.user_dense_feat_num):]

        return user_cate, user_dense, target_context_cate, target_context_dense

class Eleme_SUAN(InputBase):
    def __init__(self, data_struct, params, is_training, hashtable, ptable_lookup):
        super(Eleme_SUAN, self).__init__(data_struct, params, is_training, hashtable, ptable_lookup)
        self.d_model = self.params['embed_dim']
        self.seq_len = self.params['seq_len']
        self.n_layers = self.params['n_layers']
        self.n_heads = self.params['n_heads']
        self.dropout_rate = self.params['dropout_rate']
        self.num_buckets = self.params['num_buckets']
        self.feature_num = self.params['feature_num']

        self.cat_fea_list = ["user_id", "gender", "visit_city", "is_supervip", "ctr_30", "ord_30", "shop_id", "item_id",
                             "city_id", "district_id", "shop_aoi_id", "shop_geohash_6", "shop_geohash_12", "brand_id",
                             "category_1_id", "merge_standard_food_id", "rank_7", "rank_30", "rank_90", "times",
                             "hours", "time_type", "weekdays", "geohash12"]
        self.cat_seq_fea = ["shop_id_list[0]", "shop_id_list[1]", "shop_id_list[2]", "shop_id_list[3]",
                            "shop_id_list[4]", "shop_id_list[5]", "shop_id_list[6]", "shop_id_list[7]",
                            "shop_id_list[8]", "shop_id_list[9]", "shop_id_list[10]", "shop_id_list[11]",
                            "shop_id_list[12]", "shop_id_list[13]", "shop_id_list[14]", "shop_id_list[15]",
                            "shop_id_list[16]", "shop_id_list[17]", "shop_id_list[18]", "shop_id_list[19]",
                            "shop_id_list[20]", "shop_id_list[21]", "shop_id_list[22]", "shop_id_list[23]",
                            "shop_id_list[24]", "shop_id_list[25]", "shop_id_list[26]", "shop_id_list[27]",
                            "shop_id_list[28]", "shop_id_list[29]", "shop_id_list[30]", "shop_id_list[31]",
                            "shop_id_list[32]", "shop_id_list[33]", "shop_id_list[34]", "shop_id_list[35]",
                            "shop_id_list[36]", "shop_id_list[37]", "shop_id_list[38]", "shop_id_list[39]",
                            "shop_id_list[40]", "shop_id_list[41]", "shop_id_list[42]", "shop_id_list[43]",
                            "shop_id_list[44]", "shop_id_list[45]", "shop_id_list[46]", "shop_id_list[47]",
                            "shop_id_list[48]", "shop_id_list[49]",
                            "item_id_list[0]", "item_id_list[1]", "item_id_list[2]", "item_id_list[3]",
                            "item_id_list[4]", "item_id_list[5]", "item_id_list[6]", "item_id_list[7]",
                            "item_id_list[8]", "item_id_list[9]", "item_id_list[10]", "item_id_list[11]",
                            "item_id_list[12]", "item_id_list[13]", "item_id_list[14]", "item_id_list[15]",
                            "item_id_list[16]", "item_id_list[17]", "item_id_list[18]", "item_id_list[19]",
                            "item_id_list[20]", "item_id_list[21]", "item_id_list[22]", "item_id_list[23]",
                            "item_id_list[24]", "item_id_list[25]", "item_id_list[26]", "item_id_list[27]",
                            "item_id_list[28]", "item_id_list[29]", "item_id_list[30]", "item_id_list[31]",
                            "item_id_list[32]", "item_id_list[33]", "item_id_list[34]", "item_id_list[35]",
                            "item_id_list[36]", "item_id_list[37]", "item_id_list[38]", "item_id_list[39]",
                            "item_id_list[40]", "item_id_list[41]", "item_id_list[42]", "item_id_list[43]",
                            "item_id_list[44]", "item_id_list[45]", "item_id_list[46]", "item_id_list[47]",
                            "item_id_list[48]", "item_id_list[49]",
                            "category_1_id_list[0]", "category_1_id_list[1]", "category_1_id_list[2]",
                            "category_1_id_list[3]", "category_1_id_list[4]", "category_1_id_list[5]",
                            "category_1_id_list[6]", "category_1_id_list[7]", "category_1_id_list[8]",
                            "category_1_id_list[9]", "category_1_id_list[10]", "category_1_id_list[11]",
                            "category_1_id_list[12]", "category_1_id_list[13]", "category_1_id_list[14]",
                            "category_1_id_list[15]", "category_1_id_list[16]", "category_1_id_list[17]",
                            "category_1_id_list[18]", "category_1_id_list[19]", "category_1_id_list[20]",
                            "category_1_id_list[21]", "category_1_id_list[22]", "category_1_id_list[23]",
                            "category_1_id_list[24]", "category_1_id_list[25]", "category_1_id_list[26]",
                            "category_1_id_list[27]", "category_1_id_list[28]", "category_1_id_list[29]",
                            "category_1_id_list[30]", "category_1_id_list[31]", "category_1_id_list[32]",
                            "category_1_id_list[33]", "category_1_id_list[34]", "category_1_id_list[35]",
                            "category_1_id_list[36]", "category_1_id_list[37]", "category_1_id_list[38]",
                            "category_1_id_list[39]", "category_1_id_list[40]", "category_1_id_list[41]",
                            "category_1_id_list[42]", "category_1_id_list[43]", "category_1_id_list[44]",
                            "category_1_id_list[45]", "category_1_id_list[46]", "category_1_id_list[47]",
                            "category_1_id_list[48]", "category_1_id_list[49]",
                            "merge_standard_food_id_list[0]", "merge_standard_food_id_list[1]",
                            "merge_standard_food_id_list[2]", "merge_standard_food_id_list[3]",
                            "merge_standard_food_id_list[4]", "merge_standard_food_id_list[5]",
                            "merge_standard_food_id_list[6]", "merge_standard_food_id_list[7]",
                            "merge_standard_food_id_list[8]", "merge_standard_food_id_list[9]",
                            "merge_standard_food_id_list[10]", "merge_standard_food_id_list[11]",
                            "merge_standard_food_id_list[12]", "merge_standard_food_id_list[13]",
                            "merge_standard_food_id_list[14]", "merge_standard_food_id_list[15]",
                            "merge_standard_food_id_list[16]", "merge_standard_food_id_list[17]",
                            "merge_standard_food_id_list[18]", "merge_standard_food_id_list[19]",
                            "merge_standard_food_id_list[20]", "merge_standard_food_id_list[21]",
                            "merge_standard_food_id_list[22]", "merge_standard_food_id_list[23]",
                            "merge_standard_food_id_list[24]", "merge_standard_food_id_list[25]",
                            "merge_standard_food_id_list[26]", "merge_standard_food_id_list[27]",
                            "merge_standard_food_id_list[28]", "merge_standard_food_id_list[29]",
                            "merge_standard_food_id_list[30]", "merge_standard_food_id_list[31]",
                            "merge_standard_food_id_list[32]", "merge_standard_food_id_list[33]",
                            "merge_standard_food_id_list[34]", "merge_standard_food_id_list[35]",
                            "merge_standard_food_id_list[36]", "merge_standard_food_id_list[37]",
                            "merge_standard_food_id_list[38]", "merge_standard_food_id_list[39]",
                            "merge_standard_food_id_list[40]", "merge_standard_food_id_list[41]",
                            "merge_standard_food_id_list[42]", "merge_standard_food_id_list[43]",
                            "merge_standard_food_id_list[44]", "merge_standard_food_id_list[45]",
                            "merge_standard_food_id_list[46]", "merge_standard_food_id_list[47]",
                            "merge_standard_food_id_list[48]", "merge_standard_food_id_list[49]",
                            "brand_id_list[0]", "brand_id_list[1]", "brand_id_list[2]", "brand_id_list[3]",
                            "brand_id_list[4]", "brand_id_list[5]", "brand_id_list[6]", "brand_id_list[7]",
                            "brand_id_list[8]", "brand_id_list[9]", "brand_id_list[10]", "brand_id_list[11]",
                            "brand_id_list[12]", "brand_id_list[13]", "brand_id_list[14]", "brand_id_list[15]",
                            "brand_id_list[16]", "brand_id_list[17]", "brand_id_list[18]", "brand_id_list[19]",
                            "brand_id_list[20]", "brand_id_list[21]", "brand_id_list[22]", "brand_id_list[23]",
                            "brand_id_list[24]", "brand_id_list[25]", "brand_id_list[26]", "brand_id_list[27]",
                            "brand_id_list[28]", "brand_id_list[29]", "brand_id_list[30]", "brand_id_list[31]",
                            "brand_id_list[32]", "brand_id_list[33]", "brand_id_list[34]", "brand_id_list[35]",
                            "brand_id_list[36]", "brand_id_list[37]", "brand_id_list[38]", "brand_id_list[39]",
                            "brand_id_list[40]", "brand_id_list[41]", "brand_id_list[42]", "brand_id_list[43]",
                            "brand_id_list[44]", "brand_id_list[45]", "brand_id_list[46]", "brand_id_list[47]",
                            "brand_id_list[48]", "brand_id_list[49]",
                            "shop_aoi_id_list[0]", "shop_aoi_id_list[1]", "shop_aoi_id_list[2]", "shop_aoi_id_list[3]",
                            "shop_aoi_id_list[4]", "shop_aoi_id_list[5]", "shop_aoi_id_list[6]", "shop_aoi_id_list[7]",
                            "shop_aoi_id_list[8]", "shop_aoi_id_list[9]", "shop_aoi_id_list[10]",
                            "shop_aoi_id_list[11]", "shop_aoi_id_list[12]", "shop_aoi_id_list[13]",
                            "shop_aoi_id_list[14]", "shop_aoi_id_list[15]", "shop_aoi_id_list[16]",
                            "shop_aoi_id_list[17]", "shop_aoi_id_list[18]", "shop_aoi_id_list[19]",
                            "shop_aoi_id_list[20]", "shop_aoi_id_list[21]", "shop_aoi_id_list[22]",
                            "shop_aoi_id_list[23]", "shop_aoi_id_list[24]", "shop_aoi_id_list[25]",
                            "shop_aoi_id_list[26]", "shop_aoi_id_list[27]", "shop_aoi_id_list[28]",
                            "shop_aoi_id_list[29]", "shop_aoi_id_list[30]", "shop_aoi_id_list[31]",
                            "shop_aoi_id_list[32]", "shop_aoi_id_list[33]", "shop_aoi_id_list[34]",
                            "shop_aoi_id_list[35]", "shop_aoi_id_list[36]", "shop_aoi_id_list[37]",
                            "shop_aoi_id_list[38]", "shop_aoi_id_list[39]", "shop_aoi_id_list[40]",
                            "shop_aoi_id_list[41]", "shop_aoi_id_list[42]", "shop_aoi_id_list[43]",
                            "shop_aoi_id_list[44]", "shop_aoi_id_list[45]", "shop_aoi_id_list[46]",
                            "shop_aoi_id_list[47]", "shop_aoi_id_list[48]", "shop_aoi_id_list[49]",
                            "shop_geohash6_list[0]", "shop_geohash6_list[1]", "shop_geohash6_list[2]",
                            "shop_geohash6_list[3]", "shop_geohash6_list[4]", "shop_geohash6_list[5]",
                            "shop_geohash6_list[6]", "shop_geohash6_list[7]", "shop_geohash6_list[8]",
                            "shop_geohash6_list[9]", "shop_geohash6_list[10]", "shop_geohash6_list[11]",
                            "shop_geohash6_list[12]", "shop_geohash6_list[13]", "shop_geohash6_list[14]",
                            "shop_geohash6_list[15]", "shop_geohash6_list[16]", "shop_geohash6_list[17]",
                            "shop_geohash6_list[18]", "shop_geohash6_list[19]", "shop_geohash6_list[20]",
                            "shop_geohash6_list[21]", "shop_geohash6_list[22]", "shop_geohash6_list[23]",
                            "shop_geohash6_list[24]", "shop_geohash6_list[25]", "shop_geohash6_list[26]",
                            "shop_geohash6_list[27]", "shop_geohash6_list[28]", "shop_geohash6_list[29]",
                            "shop_geohash6_list[30]", "shop_geohash6_list[31]", "shop_geohash6_list[32]",
                            "shop_geohash6_list[33]", "shop_geohash6_list[34]", "shop_geohash6_list[35]",
                            "shop_geohash6_list[36]", "shop_geohash6_list[37]", "shop_geohash6_list[38]",
                            "shop_geohash6_list[39]", "shop_geohash6_list[40]", "shop_geohash6_list[41]",
                            "shop_geohash6_list[42]", "shop_geohash6_list[43]", "shop_geohash6_list[44]",
                            "shop_geohash6_list[45]", "shop_geohash6_list[46]", "shop_geohash6_list[47]",
                            "shop_geohash6_list[48]", "shop_geohash6_list[49]",
                            "timediff_list[0]", "timediff_list[1]", "timediff_list[2]", "timediff_list[3]",
                            "timediff_list[4]", "timediff_list[5]", "timediff_list[6]", "timediff_list[7]",
                            "timediff_list[8]", "timediff_list[9]", "timediff_list[10]", "timediff_list[11]",
                            "timediff_list[12]", "timediff_list[13]", "timediff_list[14]", "timediff_list[15]",
                            "timediff_list[16]", "timediff_list[17]", "timediff_list[18]", "timediff_list[19]",
                            "timediff_list[20]", "timediff_list[21]", "timediff_list[22]", "timediff_list[23]",
                            "timediff_list[24]", "timediff_list[25]", "timediff_list[26]", "timediff_list[27]",
                            "timediff_list[28]", "timediff_list[29]", "timediff_list[30]", "timediff_list[31]",
                            "timediff_list[32]", "timediff_list[33]", "timediff_list[34]", "timediff_list[35]",
                            "timediff_list[36]", "timediff_list[37]", "timediff_list[38]", "timediff_list[39]",
                            "timediff_list[40]", "timediff_list[41]", "timediff_list[42]", "timediff_list[43]",
                            "timediff_list[44]", "timediff_list[45]", "timediff_list[46]", "timediff_list[47]",
                            "timediff_list[48]", "timediff_list[49]",
                            "hours_list[0]", "hours_list[1]", "hours_list[2]", "hours_list[3]", "hours_list[4]",
                            "hours_list[5]", "hours_list[6]", "hours_list[7]", "hours_list[8]", "hours_list[9]",
                            "hours_list[10]", "hours_list[11]", "hours_list[12]", "hours_list[13]", "hours_list[14]",
                            "hours_list[15]", "hours_list[16]", "hours_list[17]", "hours_list[18]", "hours_list[19]",
                            "hours_list[20]", "hours_list[21]", "hours_list[22]", "hours_list[23]", "hours_list[24]",
                            "hours_list[25]", "hours_list[26]", "hours_list[27]", "hours_list[28]", "hours_list[29]",
                            "hours_list[30]", "hours_list[31]", "hours_list[32]", "hours_list[33]", "hours_list[34]",
                            "hours_list[35]", "hours_list[36]", "hours_list[37]", "hours_list[38]", "hours_list[39]",
                            "hours_list[40]", "hours_list[41]", "hours_list[42]", "hours_list[43]", "hours_list[44]",
                            "hours_list[45]", "hours_list[46]", "hours_list[47]", "hours_list[48]", "hours_list[49]",
                            "time_type_list[0]", "time_type_list[1]", "time_type_list[2]", "time_type_list[3]",
                            "time_type_list[4]", "time_type_list[5]", "time_type_list[6]", "time_type_list[7]",
                            "time_type_list[8]", "time_type_list[9]", "time_type_list[10]", "time_type_list[11]",
                            "time_type_list[12]", "time_type_list[13]", "time_type_list[14]", "time_type_list[15]",
                            "time_type_list[16]", "time_type_list[17]", "time_type_list[18]", "time_type_list[19]",
                            "time_type_list[20]", "time_type_list[21]", "time_type_list[22]", "time_type_list[23]",
                            "time_type_list[24]", "time_type_list[25]", "time_type_list[26]", "time_type_list[27]",
                            "time_type_list[28]", "time_type_list[29]", "time_type_list[30]", "time_type_list[31]",
                            "time_type_list[32]", "time_type_list[33]", "time_type_list[34]", "time_type_list[35]",
                            "time_type_list[36]", "time_type_list[37]", "time_type_list[38]", "time_type_list[39]",
                            "time_type_list[40]", "time_type_list[41]", "time_type_list[42]", "time_type_list[43]",
                            "time_type_list[44]", "time_type_list[45]", "time_type_list[46]", "time_type_list[47]",
                            "time_type_list[48]", "time_type_list[49]",
                            "weekdays_list[0]", "weekdays_list[1]", "weekdays_list[2]", "weekdays_list[3]",
                            "weekdays_list[4]", "weekdays_list[5]", "weekdays_list[6]", "weekdays_list[7]",
                            "weekdays_list[8]", "weekdays_list[9]", "weekdays_list[10]", "weekdays_list[11]",
                            "weekdays_list[12]", "weekdays_list[13]", "weekdays_list[14]", "weekdays_list[15]",
                            "weekdays_list[16]", "weekdays_list[17]", "weekdays_list[18]", "weekdays_list[19]",
                            "weekdays_list[20]", "weekdays_list[21]", "weekdays_list[22]", "weekdays_list[23]",
                            "weekdays_list[24]", "weekdays_list[25]", "weekdays_list[26]", "weekdays_list[27]",
                            "weekdays_list[28]", "weekdays_list[29]", "weekdays_list[30]", "weekdays_list[31]",
                            "weekdays_list[32]", "weekdays_list[33]", "weekdays_list[34]", "weekdays_list[35]",
                            "weekdays_list[36]", "weekdays_list[37]", "weekdays_list[38]", "weekdays_list[39]",
                            "weekdays_list[40]", "weekdays_list[41]", "weekdays_list[42]", "weekdays_list[43]",
                            "weekdays_list[44]", "weekdays_list[45]", "weekdays_list[46]", "weekdays_list[47]",
                            "weekdays_list[48]", "weekdays_list[49]"
                            ]
        self.user_info_list = ["user_id", "gender", "visit_city", "is_supervip", "ctr_30", "ord_30"]
        self.item_info_list = ["shop_id", "item_id", "city_id", "district_id", "shop_aoi_id", "shop_geohash_6",
                               "shop_geohash_12", "brand_id", "category_1_id", "merge_standard_food_id", "rank_7",
                               "rank_30", "rank_90", "hours"]
        self.context_list = ["times", "hours", "time_type", "weekdays", "geohash12"]

        self.shop_list = ["shop_id_list[%d]" % i for i in range(self.seq_len)]
        self.item_list = ["item_id_list[%d]" % i for i in range(self.seq_len)]
        self.cate_list = ["category_1_id_list[%d]" % i for i in range(self.seq_len)]
        self.aoi_list = ["shop_aoi_id_list[%d]" % i for i in range(self.seq_len)]
        self.hour_list = ["hours_list[%d]" % i for i in range(self.seq_len)]
        self.dense_fea = ["avg_price", "total_amt_30"]
        self.dense_seq_fea = ["price_list[0]", "price_list[1]", "price_list[2]", "price_list[3]", "price_list[4]",
                              "price_list[5]", "price_list[6]", "price_list[7]", "price_list[8]", "price_list[9]",
                              "price_list[10]", "price_list[11]", "price_list[12]", "price_list[13]", "price_list[14]",
                              "price_list[15]", "price_list[16]", "price_list[17]", "price_list[18]", "price_list[19]",
                              "price_list[20]", "price_list[21]", "price_list[22]", "price_list[23]", "price_list[24]",
                              "price_list[25]", "price_list[26]", "price_list[27]", "price_list[28]", "price_list[29]",
                              "price_list[30]", "price_list[31]", "price_list[32]", "price_list[33]", "price_list[34]",
                              "price_list[35]", "price_list[36]", "price_list[37]", "price_list[38]", "price_list[39]",
                              "price_list[40]", "price_list[41]", "price_list[42]", "price_list[43]", "price_list[44]",
                              "price_list[45]", "price_list[46]", "price_list[47]", "price_list[48]", "price_list[49]"]
        self.cat_list = [self.user_info_list, self.item_info_list, self.context_list, self.shop_list, self.item_list,
                         self.cate_list, self.aoi_list, self.hour_list]
        self.dense_list = [self.dense_fea, self.dense_seq_fea]


        self.transformer_blocks1 = [
            SUAN_Block(self.d_model * self.feature_num, self.n_heads, self.dropout_rate, "GenRecTransformerBlock2_{}".format(i)) for i in
            range(self.n_layers)]
        self.pos = tf.Variable(tf.random.normal([self.seq_len + 1, self.d_model * self.feature_num], mean=0, stddev=0.02), name="_pos")

    def recieve_gather_features(self, cat_feas, dense_feas):
        # cat_feas
        self.cat_fea_split = cat_feas
        self.user_info_list, self.item_info_list, self.context_list, self.shop_list, self.item_list, self.cate_list, self.aoi_list, self.hour_list = self.cat_fea_split
        self.user_info_list_emb, self.item_info_list_emb, self.context_list_emb, self.shop_list_emb, self.item_list_emb, self.cate_list_emb, self.aoi_list_emb, self.hour_list_emb = self.ptable_lookup(
            list_ids=self.cat_fea_split, v_name=self.__class__.__name__)

        self.dense_fea_split = dense_feas
        self.dense_fea, self.dense_seq_fea = self.dense_fea_split


        self.reverse_ts_list = self.hour_list
        self.target_ts = tf.expand_dims(self.item_info_list[:,-1], -1)
        self.mix_list = tf.concat([self.shop_list, tf.expand_dims(self.item_info_list[:,0], -1)], -1)



    def __call__(self, cat_features, dense_features, auxiliary_info, embed_dim, se_block):
        output_dense = tf.layers.batch_normalization(inputs=self.dense_fea, training=self.is_training,
                                                                     name='num_fea_batch_norm2')


        tgt_emb = tf.reshape(self.item_info_list_emb, [self.item_info_list_emb.shape[0], 1, -1])
        self.target_ts_emb = tgt_emb[:, :, -embed_dim:]
        self.reverse_ts_emb = self.hour_list_emb
        seq_list = tf.concat(
            [self.shop_list_emb, self.item_list_emb, self.cate_list_emb, self.aoi_list_emb, self.hour_list_emb], -1)
        tgt_emb = tf.layers.dense(tgt_emb, embed_dim*self.feature_num, activation=None, name='tgt_emb')

        input_tokens = tf.concat([seq_list, tgt_emb], 1)
        pos_time_emb = tf.concat([self.reverse_ts_emb, self.target_ts_emb], 1)
        pos_time_emb = tf.layers.dense(pos_time_emb, self.d_model * 5, activation=None)
        input_tokens += pos_time_emb
        input_tokens += self.pos

        mask = create_mask(self.mix_list)

        # time
        ts = tf.concat([self.reverse_ts_list, self.target_ts], axis=-1)
        self.ts_encoder = Relative_positional_encoding(self.seq_len + 1, self.num_buckets, "relative_user_positional", self.seq_len + 1, ts)

        user_cate = self.user_info_list_emb
        variable_scope = "GenRecTransformer1"
        with tf.variable_scope(variable_scope, reuse=tf.AUTO_REUSE):
            hidden_state = input_tokens
            for block in self.transformer_blocks1:
                hidden_state, user_cate = block(hidden_state, user_cate, mask, ts, self.ts_encoder, self.is_training)
            transformer_output = hidden_state

        last_action_pre = tf.concat([output_dense, transformer_output[:, -1, :], tf.layers.flatten(user_cate),
                                     tf.layers.flatten(self.item_info_list_emb), tf.layers.flatten(self.context_list_emb)], -1)
        return last_action_pre




