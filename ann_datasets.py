
import numpy as np

def load_fbin(filename):
    header_type = np.dtype([
        ("data_size", "<u4"),
        ("data_dim", "<u4"),
    ])
    header = np.fromfile(filename, dtype=header_type, count=1)
    print(header)
    data = np.fromfile(filename, np.float32, offset=8).reshape(header[0])
    return header[0]["data_size"], header[0]["data_dim"], data

def load_ibin(filename):
    header_type = np.dtype([
        ("data_size", "<u4"),
        ("data_dim", "<u4"),
    ])
    header = np.fromfile(filename, dtype=header_type, count=1)
    print(header)
    data = np.fromfile(filename, np.int32, offset=8).reshape(header[0])
    return header[0]["data_size"], header[0]["data_dim"], data

def load_cagra_graph(filename):
    header_type = np.dtype([
        ("version", "<i4"),
        ("data_size", "<u4"),
        ("data_dim", "<u4"),
        ("graph_degree", "<u4"),
        ("metric", "<i4"),
    ])
    # header = np.fromfile(filename, count=5, offset=3)
    # print(header)
    shape = [(1000006, 32)]
    data = np.fromfile(filename, np.uint32, offset=22, count=shape[0][0] * shape[0][1]).reshape(shape[0])
    data = data[6:, :]
    print(data, data.shape)
    idx = data >= 1000000
    print(data[idx])
    return data.shape[0], 32, data
