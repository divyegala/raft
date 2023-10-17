import struct
import numpy as np
import ann_datasets
import sys

def read_index(hnsw_index_path):
    with open(hnsw_index_path, 'rb') as file:
        offsetLevel0_ = int.from_bytes(file.read(8), byteorder='little')
        max_elements_ = int.from_bytes(file.read(8), byteorder='little')
        cur_element_count_ = int.from_bytes(file.read(8), byteorder='little')
        size_data_per_element_ = int.from_bytes(file.read(8), byteorder='little')
        # M: 16, dim = 128, data_t = float, index_t = uint32_t, list_size_type = uint32_t, labeltype: size_t
        # size_data_per_element_ = M * 2 * sizeof(index_t) + sizeof(list_size_type) + dim * sizeof(data_t) + sizeof(labeltype)
        # size_data_per_element_ = 16 * 2 * 4 + 4 + 128 * 4 + 8 = 652
        label_offset_ = int.from_bytes(file.read(8), byteorder='little')
        # label_offset_ = size_data_per_element_ - sizeof(labeltype)
        offsetData_ = int.from_bytes(file.read(8), byteorder='little')
        # offsetData_ = M * 2 * sizeof(index_t) + sizeof(list_size_type) = 16 * 2 * 4 + 4 = 132
        maxlevel_ = int.from_bytes(file.read(4), byteorder='little')
        enterpoint_node_ = int.from_bytes(file.read(4), byteorder='little')
        maxM_ = int.from_bytes(file.read(8), byteorder='little')
        # M = 16
        maxM0_ = int.from_bytes(file.read(8), byteorder='little')
        # M * 2 = 32
        M_ = int.from_bytes(file.read(8), byteorder='little')
        # M = 16
        mult_ = struct.unpack('d', file.read(8))
        ef_construction_ = int.from_bytes(file.read(8), byteorder='little')
        print(ef_construction_)

def write_index(output_path, data_path, cagra_graph_path):
    data_size, data_dim, data = ann_datasets.load_fbin(data_path)
    graph_size, graph_degree, graph = ann_datasets.load_cagra_graph(cagra_graph_path)
    print(data_size, data_dim, graph_degree)

    with open(output_path, 'wb') as file:
        offsetLevel0_bytes = int(0).to_bytes(8, byteorder='little')
        max_elements_bytes = int(data_size).to_bytes(8, byteorder='little')
        cur_element_count_bytes = int(data_size).to_bytes(8, byteorder='little')
        size_data_per_element = int(graph_degree * 4 + 4 + data_dim * 4 + 8)
        size_data_per_element_bytes = size_data_per_element.to_bytes(8, byteorder='little')
        label_offset_bytes = int(size_data_per_element - 8).to_bytes(8, byteorder='little')
        offset_data_bytes = int(graph_degree * 4 + 4).to_bytes(8, byteorder='little')
        max_level_bytes = int(1).to_bytes(4, byteorder='little')
        enterpoint_node_bytes = int(data_size // 2).to_bytes(4, byteorder='little')
        maxM_bytes = int(graph_degree // 2).to_bytes(8, byteorder='little')
        maxM0_bytes = int(graph_degree).to_bytes(8, byteorder='little')
        M_bytes = int(graph_degree // 2).to_bytes(8, byteorder='little')
        # mult_ can be any number
        mult_bytes = struct.pack('d', 0.42424242)
        ef_construction_bytes = int(500).to_bytes(8, byteorder="little")        

        file.write(offsetLevel0_bytes)
        file.write(max_elements_bytes)
        file.write(cur_element_count_bytes)
        file.write(size_data_per_element_bytes)
        file.write(label_offset_bytes)
        file.write(offset_data_bytes)
        file.write(max_level_bytes)
        file.write(enterpoint_node_bytes)
        file.write(maxM_bytes)
        file.write(maxM0_bytes)
        file.write(M_bytes)
        file.write(mult_bytes)
        file.write(ef_construction_bytes)

        for i in range(data_size):
            data_per_elment = int(graph_degree).to_bytes(4, byteorder='little')
            data_per_elment += graph[i].tobytes()
            data_per_elment += data[i].tobytes()
            data_per_elment += int(i).to_bytes(8, byteorder='little')
            file.write(data_per_elment)
        
        for i in range(data_size):
            file.write(int(0).to_bytes(4, byteorder="little"))

# output_path, data_path, cagra_graph_path
write_index(sys.argv[1], sys.argv[2], sys.argv[3])
# read_index(sys.argv[1])
