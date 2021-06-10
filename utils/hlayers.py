import numpy as np
import tensorflow as tf
from utils import util
import numpy as np

np.set_printoptions(threshold=np.inf)

# Experimental sparse attention head (for running on datasets such as Pubmed)
# N.B. Because of limitations of current TF implementation, will work _only_ if batch_size = 1!
def sp_prod_att_head(input_seq, num_heads, out_sz, adj_mat, activation, nb_nodes, tr_c, pre_curvature=None, in_drop=0.0, coef_drop=0.0):
    distance_list = []
    seq_list = []
    ret_list = []
    tangent_list = []
    if tr_c == 1:
        is_curv_train = True
    else:
        is_curv_train = False
    with tf.name_scope("prod_radius") as scope:
        c = tf.Variable([1.0], trainable=is_curv_train)
    if pre_curvature == None:
        pre_curvature = c
    for attn_num, seq in enumerate(input_seq):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        seq = tf.squeeze(seq, 0)
        seq = tf.transpose(seq)
        seq_size = seq.shape.as_list()
        with tf.name_scope("prod_att") as scope:
            W = tf.get_variable(name=scope + 'W', shape=[out_sz, seq_size[0]], initializer=tf.contrib.layers.xavier_initializer())

        ## log map
        seq_log = util.tf_my_prod_mat_log_map_zero(seq, pre_curvature)
        seq_fts_log = tf.matmul(W, seq_log)
        seq_fts_exp = tf.transpose(util.tf_my_prod_mat_exp_map_zero(tf.transpose(seq_fts_log), c))
        # attention 
        adj_indices = adj_mat.indices
        adj_idx_x = adj_indices[:, 0]
        adj_idx_y = adj_indices[:, 1]
        fts_x = tf.gather(tf.transpose(seq_fts_exp), adj_idx_x)  # shape: [2*edges+nodes, features]
        fts_y = tf.gather(tf.transpose(seq_fts_exp), adj_idx_y)
        sparse_distance = util.tf_my_mobius_list_distance(fts_x, fts_y, c)
        distance_list.append(sparse_distance)
        seq_fts = tf.transpose(seq_fts_log)
        seq_list.append(seq_fts)
    prod_dis = tf.stack(distance_list, axis=-1)
    prod_dis = tf.reduce_sum(prod_dis ** 2, axis=1)
    lrelu = tf.SparseTensor(indices=adj_indices, values=-tf.sqrt(prod_dis+1e-8), dense_shape=adj_mat.dense_shape)
    coefs = tf.sparse_softmax(lrelu)

    for attn_num, seq_fts in enumerate(seq_list):

        seq_fts = tf.expand_dims(seq_fts, 0)

        if coef_drop != 0.0:
            coefs = tf.SparseTensor(indices=coefs.indices,
                                    values=tf.nn.dropout(coefs.values, 1.0 - coef_drop),
                                    dense_shape=coefs.dense_shape)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        # As tf.sparse_tensor_dense_matmul expects its arguments to have rank-2,
        # here we make an assumption that our input is of batch size 1, and reshape appropriately.
        # The method will fail in all other cases!
        coefs = tf.sparse_reshape(coefs, [nb_nodes, nb_nodes])
        seq_fts = tf.squeeze(seq_fts)
        vals = tf.sparse_tensor_dense_matmul(coefs, seq_fts)
        vals = tf.expand_dims(vals, axis=0)
        vals.set_shape([1, nb_nodes, out_sz])
        ret_before = tf.contrib.layers.bias_add(vals)
        ret = util.tf_my_prod_mat_exp_map_zero(activation(tf.squeeze(ret_before)), c)
        ret = tf.expand_dims(ret, 0)
        ret_list.append(ret)

        # print('tangent')
        # tangent = util.tf_my_prod_mat_log_map_zero(activation(tf.squeeze(ret)), c)
        # tangent = tf.expand_dims(tangent, 0)
        # tangent_list.append(tangent)
    return ret_list, c
