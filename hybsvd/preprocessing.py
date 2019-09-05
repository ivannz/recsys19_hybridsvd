import numpy as np
import pandas as pd

from polara import RecommenderData
from polara.lib.similarity import stack_features, cosine_similarity
from polara.recommender.coldstart.data import ColdStartSimilarityDataModel
from polara.recommender.hybrid.data import IdentityDiagonalMixin, SideRelationsMixin


class SimilarityDataModel(SideRelationsMixin, RecommenderData):
    pass


def get_similarity_data(meta_info, metric='common', assume_binary=True, fill_diagonal=True):
    feat_mat, lbls = stack_features(meta_info, normalize=False)

    if metric == 'common':
        item_similarity = feat_mat.dot(feat_mat.T)
        item_similarity = item_similarity / item_similarity.data.max()
        item_similarity.setdiag(1.0)

    if (metric == 'cosine') or (metric == 'salton'):
        item_similarity = cosine_similarity(feat_mat,
                                            assume_binary=assume_binary,
                                            fill_diagonal=fill_diagonal)

    if item_similarity.format == 'csr':
        item_similarity = item_similarity.T  # ensure CSC format (matrix is symmetric)

    userid = 'userid'
    itemid = meta_info.index.name
    similarities = {userid: None, itemid: item_similarity}
    indices = {userid: None, itemid: meta_info.index}
    labels = {userid: None, itemid: lbls}
    return similarities, indices, labels


def prepare_data_model(data_label, raw_data, similarities, sim_indices, item_meta, seed=0, feedback=None):
    userid = 'userid'
    itemid = item_meta[data_label].index.name
    data_model = SimilarityDataModel(similarities[data_label],
                                     sim_indices[data_label],
                                     raw_data[data_label],
                                     userid, itemid,
                                     feedback=feedback,
                                     seed=seed)
    data_model.test_fold = 1
    data_model.holdout_size = 1
    data_model.random_holdout = True
    data_model.warm_start = False
    data_model.verbose = False
    data_model.prepare()
    return data_model


def prepare_cold_start_data_model(data_label, raw_data, similarities, sim_indices, item_meta, seed=0, feedback=None):
    userid = 'userid'
    itemid = item_meta[data_label].index.name
    data_model = ColdStartSimilarityDataModel(similarities[data_label],
                                              sim_indices[data_label],
                                              raw_data[data_label],
                                              userid, itemid,
                                              feedback=feedback,
                                              seed=seed)
    data_model.test_fold = 1
    data_model.holdout_size = 1
    data_model.random_holdout = True
    data_model.verbose = False
    data_model.prepare()
    return data_model
