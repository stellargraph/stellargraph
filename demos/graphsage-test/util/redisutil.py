import numpy as np
import json
import os
from redis import StrictRedis
from networkx.readwrite import json_graph


def serialise(data, dtype):
    return data.astype(dtype).ravel().tostring()


def deserialise(data, dtype):
    return np.fromstring(data, dtype=dtype)


def save_features(r, features):
    pipe = r.pipeline()
    for i in range(features.shape[0]):
        pipe.set("feat:" + str(i), serialise(features[i], np.float32))
    pipe.execute()


def write_adj(r, nxg, node_prefix, write_deg=False, train_rm=False):
    # deg = np.zeros((len(id_map),))
    written = []
    pipe = r.pipeline()
    for node_id in nxg.nodes():
        if train_rm and (nxg.node[node_id]['test'] or nxg.node[node_id]['val']):
            continue
        neighbours = np.array(
            [nb for nb in nxg.neighbors(node_id) if (not train_rm) or (not nxg[node_id][nb]['train_removed'])]
        )
        # deg[id_map[node_id]] = len(neighbours)
        if len(neighbours) == 0:
            print("Found node without any neighbours! This MIGHT break! node_id: {}".format(node_id))
            continue
        # pipe.set(node_prefix + str(node_id), serialise(neighbours, np.int64))
        pipe.delete(node_prefix + str(node_id))
        pipe.sadd(node_prefix + str(node_id), *neighbours)
        written.append(node_id)
        if write_deg:
            pipe.set('deg:' + str(node_id), str(len(neighbours)))
    pipe.execute()
    return written


def write_labels(r, label_map):
    if isinstance(list(label_map.values())[0], list):
        num_labels = len(list(label_map.values())[0])
    else:
        num_labels = len(set(label_map.values()))

    pipe = r.pipeline()

    for node_id, label in label_map.items():
        pipe.set('label:' + str(node_id), label)

    pipe.set('num_labels', num_labels)
    pipe.execute()  # NOTE: later de-serialise using json.loads


def write_id_shuffle(r, name, ids):
    """
    Shuffle IDs and write to redis.

    :param r:
    :param name:
    :param ids: set to None to re-shuffle from existing redis store
    :return:
    """
    if ids is None:
        ids = r.lrange(name, 0, -1)
        assert len(ids) > 0

    np.random.shuffle(ids)
    r.delete(name)
    r.lpush(name, *ids)


def write_id_set(r, name, ids):
    r.delete(name)
    r.sadd(name, *ids)


def write_to_redis(prefix):
    r = StrictRedis(host='localhost', port=6379)
    nxg, features, _, _, label_map = load_data(prefix)

    if not (features is None):
        # pad with dummy zero vector
        features = np.vstack([features, np.zeros((features.shape[1],))])

        # write features
        save_features(r, features)

    train_nodes = write_adj(r, nxg, 'train:', write_deg=True, train_rm=True)
    write_adj(r, nxg, 'test:', write_deg=False, train_rm=False)
    val_nodes = [node_id for node_id in nxg.nodes() if nxg.node[node_id]['val']]
    test_nodes = [node_id for node_id in nxg.nodes() if nxg.node[node_id]['test']]

    write_labels(r, label_map)

    write_id_shuffle(r, 'train', train_nodes)
    write_id_set(r, 'val', val_nodes)
    write_id_set(r, 'test', test_nodes)

    return r


def load_data(prefix):
    G_data = json.load(open(prefix + "-G.json"))
    G = json_graph.node_link_graph(G_data)

    if os.path.exists(prefix + "-feats.npy"):
        feats = np.load(prefix + "-feats.npy")
    else:
        print("No features present.. Only identity features will be used.")
        feats = None
    id_map = json.load(open(prefix + "-id_map.json"))
    id_map = {k: int(v) for k, v in id_map.items()}
    walks = []
    class_map = json.load(open(prefix + "-class_map.json"))

    # Make sure the graph has edge train_removed annotations
    # (some datasets might already have this..)
    print("Loaded data.. now preprocessing..")
    for edge in G.edges():
        if (G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or
                G.node[edge[0]]['test'] or G.node[edge[1]]['test']):
            G[edge[0]][edge[1]]['train_removed'] = True
        else:
            G[edge[0]][edge[1]]['train_removed'] = False

    if feats is not None:
        from sklearn.preprocessing import StandardScaler
        train_ids = np.array([n for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']])
        train_feats = feats[train_ids]
        scaler = StandardScaler()
        scaler.fit(train_feats)
        feats = scaler.transform(feats)

    return G, feats, id_map, walks, class_map
