import logging
import tensorflow as tf
import numpy as np
from numpy import linalg as LA
from numpy import random as np_random
import os
import random

PROJ_EPS = 1e-5
EPS = 1e-15
MAX_TANH_ARG = 15.0
clip_value = 0.98

def tf_project_hyp_vecs(x, c):
    # Projection op. Need to make sure hyperbolic embeddings are inside the unit ball.
    return tf.clip_by_norm(t=x, clip_norm=(1. - PROJ_EPS) / np.sqrt(c), axes=[1])

######################## x,y have shape [batch_size, emb_dim] in all tf_* functions ################

# Real x, not vector!
def tf_atanh(x):
    return tf.atanh(tf.minimum(x, 1. - EPS)) # Only works for positive real x.

# Real x, not vector!
def tf_tanh(x):
    return tf.tanh(tf.minimum(tf.maximum(x, -MAX_TANH_ARG), MAX_TANH_ARG))

def tf_my_prod_mob_addition(u, v, c):
    # input [nodes, features]
    norm_u_sq = tf.norm(u, axis=1) ** 2
    norm_v_sq = tf.norm(v, axis=1) ** 2
    uv_dot_times = 4 * tf.reduce_mean(u * v, axis=1) * c
    denominator = 1 + uv_dot_times + norm_u_sq * norm_v_sq * c * c
    coef_1 = (1 + uv_dot_times + c * norm_v_sq) / denominator
    coef_2 = (1 - c * norm_u_sq) / denominator
    return tf.multiply(tf.expand_dims(coef_1, 1), u) + tf.multiply(tf.expand_dims(coef_2, 1), v)

def tf_my_prod_mat_log_map_zero(M, c):
    sqrt_c = tf.sqrt(c)
    # M = tf.transpose(M)
    M = M + EPS
    M = tf.clip_by_norm(M, clip_norm=clip_value / sqrt_c, axes=0)
    m_norm = tf.norm(M, axis=0)
    print("mat log map zero, shape before", M.shape.as_list(), 'norm shape', m_norm.shape.as_list())
    atan_norm = tf.atanh(tf.clip_by_value(m_norm*sqrt_c, clip_value_min=-0.9, clip_value_max=0.9))
    M_cof = atan_norm / m_norm / sqrt_c
    res = M * M_cof
    return res


def tf_my_prod_mat_exp_map_zero(vecs, c):
    sqrt_c = tf.sqrt(c)
    vecs = vecs + EPS
    vecs = tf.transpose(vecs)
    vecs = tf.clip_by_norm(vecs, clip_norm=clip_value, axes=0)
    norms = tf.norm(vecs, axis=0)
    print("exp map, norm size", norms.shape.as_list())
    c_tanh = tf.tanh(norms*sqrt_c)
    coef = c_tanh / norms / sqrt_c
    res = vecs * coef
    return tf.transpose(res)

def tf_my_mobius_list_distance(mat_x, mat_y, c):
    # input: [nodes features]
    mat_add = tf_my_prod_mob_addition(-mat_x, mat_y, c)
    sqrt_c = tf.sqrt(c)
    res_norm = tf.norm(mat_add, axis=1)
    # res = tf.atanh(sqrt_c * res_norm)
    res = tf.atanh(tf.clip_by_value(sqrt_c * res_norm, clip_value_min=1e-8, clip_value_max=clip_value))
    return 2 / sqrt_c * res

def tf_mat_exp_map_zero(M, c=1.):
    M = M + EPS
    sqrt_c = tf.sqrt(c)
    M = tf.clip_by_norm(M, clip_norm=clip_value / sqrt_c, axes=0)
    norms = tf.norm(M, axis=0)
    print("exp map, norm size", norms.shape.as_list())
    c_tanh = tf.tanh(norms * sqrt_c)
    coef = c_tanh / norms / sqrt_c
    res = M * coef
    return res
    # return tf.transpose(res)

def tf_my_prod_mat_exp_map_zero(vecs, c):
    sqrt_c = tf.sqrt(c)
    vecs = vecs + EPS
    vecs = tf.transpose(vecs)
    vecs = tf.clip_by_norm(vecs, clip_norm=clip_value / sqrt_c, axes=0)
    norms = tf.norm(vecs, axis=0)
    print("exp map, norm size", norms.shape.as_list())
    c_tanh = tf.tanh(norms*sqrt_c)
    coef = c_tanh / norms / sqrt_c
    res = vecs * coef
    return tf.transpose(res)


# # each row
def tf_mat_log_map_zero(M, c=1):
    M = M + EPS
    M = tf.clip_by_norm(M, clip_norm=clip_value, axes=0)
    m_norm = tf.norm(M, axis=0)
    print("log map the len is ", m_norm.shape.as_list())
    atan_norm = tf_atanh(m_norm)
    M_cof = atan_norm / m_norm
    res = M * M_cof
    return res
