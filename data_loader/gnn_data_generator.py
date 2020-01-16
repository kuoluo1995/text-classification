import sys
import numpy as np
import networkx as nx
import pickle as pkl
import scipy.sparse as sp


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_adjacency_matrix(path, dataset_name, name):
    file = open(path + '/ind.{}.{}'.format(dataset_name, name), 'rb')
    if sys.version_info > (3, 0):
        x = pkl.load(file, encoding='latin1')
    else:
        x = pkl.load(file)
    return x


def load_data(path, dataset_name):
    x = load_adjacency_matrix(path, dataset_name, 'x')
    y = load_adjacency_matrix(path, dataset_name, 'y')
    tx = load_adjacency_matrix(path, dataset_name, 'tx')
    ty = load_adjacency_matrix(path, dataset_name, 'ty')
    allx = load_adjacency_matrix(path, dataset_name, 'allx')
    ally = load_adjacency_matrix(path, dataset_name, 'ally')
    adj = load_adjacency_matrix(path, dataset_name, 'adj')
    # adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    features = sp.vstack((allx, tx)).tolil()
    labels = np.vstack((ally, ty))

    train_idx = parse_index_file('{}/ind.{}.train.index'.format(path, dataset_name))
    train_size = len(train_idx)
    test_size = tx.shape[0]

    idx_train = range(len(y))
    idx_test = range(allx.shape[0], allx.shape[0] + test_size)

    mask_train = sample_mask(idx_train, labels.shape[0])
    mask_test = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[mask_train, :] = labels[mask_train, :]
    y_test[mask_test, :] = labels[mask_test, :]
    # 'adj': adjacency_matrix, 'feature': features
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    features = sp.identity(features.shape[0])  # featureless
    features = preprocess_features(features)  # Some preprocessing todo check
    num_support = 1
    support = [preprocess_adj(adj) for _ in range(num_support)]

    return {'train_size': train_size, 'test_size': test_size, 'y_train': y_train, 'y_test': y_test,
            'mask_train': mask_train, 'mask_test': mask_test, 'input_shape': features[2], 'num_class': y_train.shape[1],
            'num_nonzero': features[1].shape, 'features': features, 'support': support, 'num_support': num_support}


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))  # todo check
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj, noTuple=False):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    if noTuple:
        return adj_normalized
    else:
        return sparse_to_tuple(adj_normalized)
