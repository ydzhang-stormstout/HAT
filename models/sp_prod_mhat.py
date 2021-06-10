import numpy as np
import tensorflow as tf
from utils import hlayers as layers
from utils import util
from models.base_hgattn import BaseHGAttN

class SpMHGAT(BaseHGAttN):
    def inference(inputs, nb_classes, nb_nodes, training, attn_drop, ffd_drop,
            bias_mat, hid_units, n_heads, activation=tf.nn.elu, c=0):
        # if agg.upper() == "MHAT":
            # print('===USING Multi-head HAT')
        this_layer = layers.sp_prod_att_head
        # else:
            # raise NotImplementedError
        attns = []
        inputs = tf.transpose(tf.squeeze(inputs, 0))
        print("input exp map zero, shape:", inputs.shape.as_list())
        inputs = tf.transpose(util.tf_mat_exp_map_zero(inputs))
        print("input after exp map zero, shape:", inputs.shape.as_list())
        inputs = tf.expand_dims(inputs, 0)

        # input layer
        input_list = [inputs for _ in range(n_heads[0])]
        # for _ in range(n_heads[0]):
        att, this_c = this_layer(input_list, num_heads=n_heads[0], adj_mat=bias_mat,
                out_sz=hid_units[0], activation=activation, nb_nodes=nb_nodes, tr_c=c,
                in_drop=ffd_drop, coef_drop=attn_drop)
        # attns.append(att)
        attns = att
        # h_1 = tf.concat(attns, axis=-1)
        # if agg.upper() == "MHAT":
        #     this_layer = layers.sp_prod_att_head
        #     h_1 = attns
        # hidden layer
        for i in range(1, len(hid_units)):
            # h_old = h_1
            attns = []
            print('WARNING, # layer > 2')
            exit()
            for _ in range(n_heads[i]):
                att, this_c = this_layer(attns, num_heads=n_heads[i], pre_curvature=this_c, adj_mat=bias_mat,
                    out_sz=hid_units[i], activation=activation, nb_nodes=nb_nodes, tr_c=c,
                    in_drop=ffd_drop, coef_drop=attn_drop)
                attns.append(att)
            # h_1 = tf.concat(attns, axis=-1)
        out = []
        # output layer
        # if agg.upper() == "MHAT":
            # h_1 = attns
        # if mlr == 0:
        # for i in range(n_heads[-1]):
        att, last_c = this_layer(attns, num_heads=n_heads[-1], pre_curvature=this_c,
            adj_mat=bias_mat, out_sz=nb_classes, activation=lambda x: x, nb_nodes=nb_nodes, tr_c=0,
            in_drop=ffd_drop, coef_drop=attn_drop)
        # out.append(att)
        out = att
            # print('-*-*-*-*-*-', tf.stack(att, axis=1).shape.)

        logits = tf.add_n(out) / n_heads[-1]
        # logits = tf.add_n(tangent) / n_heads[-1] # curvature would be smaller
        # elif mlr == 1 and n_heads[-1] == 1 and agg.upper() == "HAT":
        #     raise NotImplementedError

        return logits, tf.concat(attns, axis=-1), this_c
        # return logits, tf.concat(emb_get, axis=-1), this_c
        # return logits, tangent, this_c
