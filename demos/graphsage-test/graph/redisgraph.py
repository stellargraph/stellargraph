"""
Redis Graph, Sampler and Iterator classes

"""

import numpy as np
import json
import pickle
from util.redis_ppi import write_id_shuffle


class RedisSampler:
    def __init__(self, r):
        self._r = r

    def __call__(self, inputs):
        ids, num_samples, id_prefix, *id_suffix = inputs
        id_suffix = id_suffix[0] if len(id_suffix) > 0 else ""
        pipe = self._r.pipeline()
        for i in ids:
            pipe.srandmember(id_prefix + str(i) + id_suffix, -1 * num_samples)
        adj = np.array(pipe.execute()).astype(np.int64)
        return adj


class RedisGraph:
    def __init__(self, r, batch_size, num_samples):
        self._r = r
        self.num_labels = r.get('num_labels')
        self.num_feats = len(np.fromstring(r.get('feat:1'), dtype=np.float32))
        self.num_train = r.llen('train')
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.idx = 0
        self.sampler = RedisSampler(r)

    def _get_labels(self, ids):
        pipe = self._r.pipeline()
        for i in ids:
            pipe.get("label:" + str(i))
        return np.array([json.loads(res) for res in pipe.execute()])

    def _get_feats(self, ids):
        pipe = self._r.pipeline()
        for i in ids:
            pipe.get("feat:" + str(i))
        return np.array([np.fromstring(res, dtype=np.float32) for res in pipe.execute()])

    def _get_layer_feats(self, feats, feat_id_map, ids):
        return [feats[idx] for idx in [feat_id_map[i] for i in ids]]

    def _get_minibatch(self, ids, id_prefix='train:'):
        labels = self._get_labels(ids)
        adj1 = self.sampler((ids, self.num_samples[1], id_prefix))
        ids1 = adj1.flatten()
        adj2 = self.sampler((ids1, self.num_samples[0], id_prefix))
        feat_ids = np.unique(np.concatenate((ids, ids1, adj2.flatten())))
        feat_id_map = {i: idx for idx, i in enumerate(feat_ids)}
        feats = self._get_feats(feat_ids)
        feat0 = self._get_layer_feats(feats, feat_id_map, ids)
        feat1 = self._get_layer_feats(feats, feat_id_map, ids1)
        feat2 = self._get_layer_feats(feats, feat_id_map, adj2.flatten())
        return len(ids), labels, feat0, feat1, feat2

    def train_gen(self):
        self.shuffle()
        while self.idx < self.num_train:
            end_idx = min(self.idx + self.batch_size, self.num_train)
            ids = [pickle.loads(i) for i in self._r.lrange('train', self.idx, end_idx - 1)]
            self.idx = end_idx
            yield self._get_minibatch(ids)

    def test_gen(self):
        ids = np.array(self._r.srandmember('test', self.batch_size)).astype(np.int64)
        yield self._get_minibatch(ids, 'test:')

    def shuffle(self):
        # this loads all nodes in memory. not representative of final goal solution
        print("Shuffling...")
        write_id_shuffle(self._r, 'train', None)
        self.idx = 0
        print("Shuffled!")


class BatchGenerator:
    def __init__(self, n, nb, get_minibatch):
        self.n = n
        self.nb = nb
        self._get_minibatch = get_minibatch

    def __call__(self, *args, **kwargs):
        end = 0
        while end < self.n:
            start = end
            end = min(start + self.nb, self.n)
            yield self._get_minibatch(start, end)


class RedisHin:
    def __init__(self, r):
        self._r = r
        self.num_labels = int(r.get('num_labels'))
        self.num_train = r.llen('train')
        self.sampler = RedisSampler(r)

    def _get_labels(self, ids):
        pipe = self._r.pipeline()
        for i in ids:
            pipe.get("label:" + str(i))
        return np.array([pickle.loads(res) for res in pipe.execute()])

    def _get_feats(self, ids):
        pipe = self._r.pipeline()
        for i in ids:
            pipe.get("feat:" + str(i))
        return np.array([np.fromstring(res, dtype=np.float32) for res in pipe.execute()])

    def _get_layer_feats(self, feats, feat_id_map, ids):
        return [feats[idx] for idx in [feat_id_map[i] for i in ids]]

    def _get_ids(self, start, end, mode):
        return [pickle.loads(x) for x in self._r.lrange(mode, start, end-1)]

    def minibatch(self, ids0, schema, mode):
        labels = self._get_labels(ids0)
        ids = list(zip(*ids0)) if schema.xlen[-1] > 1 else [ids0]
        ids += [[]] * (schema.xlen[0] - len(ids))
        for i, neighs in enumerate(schema.neighs):
            for nid, etype in neighs:
                ids[nid] = self.sampler((ids[i], schema.n_samples[nid], mode + ":", ":" + etype)).flatten()
        feat_ids = np.unique(np.concatenate(ids))
        feat_id_map = {i: idx for idx, i in enumerate(feat_ids)}
        feats_map = self._get_feats(feat_ids)
        feats = [self._get_layer_feats(feats_map, feat_id_map, i) for i in ids]
        return (len(ids0), labels, *feats)

    def train_gen(self, nb, schema):
        write_id_shuffle(self._r, 'train', None)
        return BatchGenerator(
            self.num_train,
            nb,
            lambda start, end: self.minibatch(self._get_ids(start, end, 'train'), schema, 'train')
        )

    def test_gen(self, nb, schema):
        ids = [pickle.loads(i) for i in self._r.srandmember('test', nb)]

        def gen():
            yield self.minibatch(ids, schema, 'test')

        return gen
