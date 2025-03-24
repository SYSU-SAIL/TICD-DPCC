#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <deque>
#include <vector>
#include <algorithm>  // For std::find_if
#include <pybind11/stl.h>
#include <unordered_map>
#include <tuple>
#include <mutex>
#include <queue>
#include <atomic>
#include <omp.h>
#include <iostream>



#define pad_token_id -1

namespace py = pybind11;

// 自定义哈希函数
struct PairHash {
    template <class T1, class T2>
    std::size_t operator() (const std::pair<T1,T2>& pair) const {
        auto hash1 = std::hash<T1>{}(pair.first);
        auto hash2 = std::hash<T2>{}(pair.second);
        return hash1 ^ hash2; // 结合两个哈希值
    }
};

class Octree {
public:
    std::vector<std::vector<int>> octree;
    int current_index = 0;  // 用于记录迭代器的状态
    int node_num = 0;
    int trace_parent_num;
    // 填充节点
    std::vector<int> padded_node = {-1, -1, -1, -1, -1};
    // 缓存结构：存储(idx, n, empty_padding, result)
    using NChildCacheItem = std::tuple<int, int, bool, std::vector<std::vector<int>>>;
    std::deque<NChildCacheItem> nchild_cache;
    // parent list 的缓存
    std::unordered_map<int, std::vector<std::vector<int>>> parent_cache; // 缓存结构
    // n ancestors的缓存
    std::unordered_map<std::pair<int, int>, std::vector<int>, PairHash> ancestor_cache; // 缓存结构
    // context 的缓存
    std::unordered_map<std::pair<int, int>, std::pair<std::vector<std::vector<int>>, std::vector<std::vector<std::vector<int>>>>, PairHash> context_cache;
    // 假设我们增加一个成员变量来存储每个节点的 trace_parent_num 个父节点
//    std::vector<std::vector<std::vector<int>>> parent_map;
    py::array_t<int> parent_map;

    size_t cache_size;  // 缓存最大容量

    Octree(py::array_t<int> octree_data, py::array_t<int> parent_map_data, size_t cache_size = 10) : cache_size(cache_size),parent_map(parent_map_data) {
        // 从 NumPy 数组初始化 octree
        auto buf = octree_data.unchecked<2>(); // 2D array

        // 获取数据
        this->node_num = buf.shape(0);
        this->octree.resize(this->node_num);

        #pragma omp parallel for // 并行化循环
        for (size_t i = 0; i < this->node_num; i++) {
            this->octree[i] = {buf(i, 0), buf(i, 1), buf(i, 2), buf(i, 3), buf(i, 4)};
        }
            // 从输入的 parent_map_data 初始化 parent_map
        auto parent_buf = parent_map_data.unchecked<3>(); // 3D array
        this->trace_parent_num = parent_buf.shape(1); // 赋值 trace_parent_num
    }

    std::vector<std::vector<int>> get_full_window(int context_len) {
        return std::vector<std::vector<int>>(octree.begin(), octree.begin() + std::min(context_len, static_cast<int>(octree.size())));
    }

    std::vector<std::vector<int>> get_full_window_from_end(int context_len) {
        return std::vector<std::vector<int>>(octree.end() - std::min(context_len, static_cast<int>(octree.size())), octree.end());
    }

    // Method to get node by ID
    std::vector<int> get_node_by_id(int idx) {
        auto it = std::find_if(octree.begin(), octree.end(), [idx](const std::vector<int>& node) {
            return node[0] == idx;
        });
        if (it != octree.end()) {
            return *it;
        } else {
            throw std::runtime_error("Node not found for ID: " + std::to_string(idx));
        }
    }


    std::vector<int> preview_next() {
        if (current_index < static_cast<int>(octree.size())) {
            return octree[current_index];
        } else {
            return padded_node;  // This will raise StopIteration in Python
        }
    }

    // Method to get children of node by ID
    std::vector<std::vector<int>> get_children_of_id(int idx, bool empty_padding = false) {
        std::vector<std::vector<int>> children;

        if (!empty_padding) {
            for (const auto& node : octree) {
                if (node[1] == idx) {  // Check parent ID
                    children.push_back(node);
                }
            }
        } else {
            children.resize(8, std::vector<int>(5, pad_token_id));  // Fill with padding
            for (int i = 0; i < 8; ++i) {
                children[i][3] = i;  // Assign octant values
            }
            for (const auto& node : octree) {
                if (node[1] == idx) {
                    children[node[3]] = node;
                }
            }
        }
        return children;
    }

    // Method to get parent of node by ID
    std::vector<int> get_parent_of_id(int idx) {
        if (idx < 0) {
            return padded_node; // 返回填充的节点
        }

        auto node = get_node_by_id(idx);
        int parent_idx = node[1];  // Parent ID

        if (parent_idx >= 0) {
            return get_node_by_id(parent_idx); // 返回父节点
        } else {
            return padded_node; // 返回填充的节点
        }
    }

    std::vector<std::vector<int>> get_parent_list_of_id(int idx) {
        int key = idx;
        if (parent_cache.find(key) != parent_cache.end()) {
            return parent_cache[key]; // 返回缓存结果
        }
        //无缓存读取
        std::vector<std::vector<int>> parent_list(trace_parent_num, std::vector<int>(5)); // 初始化 parent_list
        auto parent_buf = parent_map.unchecked<3>();
        for (int j = 0; j < trace_parent_num; j++) {
                for (int k = 0; k < 5; k++) {
                    parent_list[j][k] = parent_buf(idx, j, k);
                }
        }

        // 更新缓存
        if (parent_cache.size() >= cache_size) {
            parent_cache.erase(parent_cache.begin()); // 移除最旧的缓存
        }
        parent_cache[key] = parent_list;

        return parent_list;
    }

    // Method to get nth parent of node by ID
    std::vector<int> get_n_parent_of_id(int idx, int n = 1) {
        std::pair<int, int> key = {idx, n};
        if (ancestor_cache.find(key) != ancestor_cache.end()) {
            return ancestor_cache[key];
        }

        for (int i = 0; i < n; ++i) {
            auto node = get_node_by_id(idx);
            int parent_idx = node[1];
            idx = parent_idx;
        }
        auto ancestor = get_node_by_id(idx);

        // 将结果存入缓存
        if (ancestor_cache.size() >= cache_size) {
            ancestor_cache.erase(ancestor_cache.begin()); // 移除最旧的缓存
        }
        ancestor_cache[key] = ancestor;
        return ancestor;
    }

    // Method to get the root node
    std::vector<int> get_root_node() {
        auto it = std::find_if(octree.begin(), octree.end(), [](const std::vector<int>& node) {
            return node[2] == 0;  // Depth == 0 (root node)
        });
        if (it != octree.end()) {
            return *it;
        } else {
            throw std::runtime_error("Root node not found");
        }
    }

    // Method to get nth children of node by ID
    std::vector<std::vector<int>> get_n_children_of_id(int idx, int n = 1, bool empty_padding = false) {
        // 1. Check if the result is in cache
        auto cache_it = std::find_if(nchild_cache.begin(), nchild_cache.end(),
                                     [idx, n, empty_padding](const NChildCacheItem& item) {
                                         return std::get<0>(item) == idx &&
                                                std::get<1>(item) == n &&
                                                std::get<2>(item) == empty_padding;
                                     });

        if (cache_it != nchild_cache.end()) {
            // Cache hit: return the cached result
            return std::get<3>(*cache_it);
        }

        // 2. Compute the result
        std::vector<std::vector<int>> parent_list;
        parent_list.push_back(get_node_by_id(idx));  // Initialize with the starting node
        std::vector<std::vector<int>> child_list;

        for (int i = 0; i < n; ++i) {
            child_list.clear();  // Clear previous children

            for (const auto& parent_node : parent_list) {
                auto children = get_children_of_id(parent_node[0], empty_padding);  // Get children
                child_list.insert(child_list.end(), children.begin(), children.end());  // Append to child_list
            }

            parent_list = child_list;  // Move to the next level
        }

         // 3. Add result to cache
        if (nchild_cache.size() >= cache_size) {
            nchild_cache.pop_front();  // Remove the oldest cache entry
        }
        nchild_cache.emplace_back(idx, n, empty_padding, parent_list);  // Add new result to cache

        return parent_list;  // Return the final list of children
    }

    std::pair<std::vector<std::vector<int>>, std::vector<std::vector<std::vector<int>>>> get_context_window(int idx, int context_len) {
        // 检查缓存
        std::pair<int, int> key = {idx, context_len};
        if (context_cache.find(key) != context_cache.end()) {
            return context_cache[key];
        }
        // 无缓存读取
        int half_len = context_len / 2;
        int start_index = idx - half_len;
        int end_index = idx + half_len;

        // 确定窗口的范围
        std::vector<std::vector<int>> window;
        if (start_index < 0) {
            window = get_full_window(context_len);
        } else if (end_index >= octree.size()) {
            window = get_full_window_from_end(context_len);
        } else {
            window = std::vector<std::vector<int>>(octree.begin() + start_index, octree.begin() + start_index + context_len);
        }

        // 获取每个节点的父节点列表
        std::vector<std::vector<std::vector<int>>> parent_list(window.size(), std::vector<std::vector<int>>(trace_parent_num, std::vector<int>(5, 0)));
        auto parent_buf = parent_map.unchecked<3>();
        for (size_t i = 0; i < window.size(); ++i) {
            int node_idx = window[i][0];
            for (int j = 0; j < trace_parent_num; j++) {
                for (int k = 0; k < 5; k++) {
                    parent_list[i][j][k] = parent_buf(node_idx, j, k);
                }
            }
        }

        // 将结果存入缓存
        if (context_cache.size() >= cache_size) {
            context_cache.erase(context_cache.begin()); // 移除最旧的缓存
        }
        context_cache[key] = {window, parent_list};

        return {window, parent_list};
    }


    // Iterator methods
    Octree& __iter__() {
        // Reset the current index when a new iteration starts
        current_index = 0;
        return *this;
    }

    std::vector<int> __next__() {
        if (current_index < static_cast<int>(octree.size())) {
            return octree[current_index++];
        } else {
            throw py::stop_iteration();  // This will raise StopIteration in Python
        }
    }
};


class Octree2Points {
private:
    std::vector<double> min_bound;
    std::vector<double> max_bound;
    int depth;
    std::vector<std::vector<int>> bound_affinity;
    std::mutex mtx;

    struct Task {
        int node_id;
        int depth;
    };

public:
    Octree2Points(py::array_t<double> min_bound_array, py::array_t<double> max_bound_array, int depth)
        : depth(depth) {
        auto min_buf = min_bound_array.unchecked<1>();
        auto max_buf = max_bound_array.unchecked<1>();
        min_bound = {min_buf(0), min_buf(1), min_buf(2)};
        max_bound = {max_buf(0), max_buf(1), max_buf(2)};

        bound_affinity = {
            {0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1},
            {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}};
    }

    std::vector<std::vector<double>>
    parally_convert(const py::array_t<int>& octree_array) {
        auto octree_buf = octree_array.unchecked<2>();
        ssize_t node_len = octree_array.shape(0);

        // 初始化边界
        std::vector<size_t> shape = {node_len, 3};
        py::array_t<double> min_bounds_array(shape);
        py::array_t<double> max_bounds_array(shape);
        auto min_bounds_buf = min_bounds_array.mutable_unchecked<2>();
        auto max_bounds_buf = max_bounds_array.mutable_unchecked<2>();
        std::vector<bool> decoded_list(node_len, false);
        std::vector<bool> ready4decode(node_len, false);
        ready4decode[0] = true;

        // 初始化根节点
        for (int i = 0; i < 3; ++i) {
            min_bounds_buf(0, i) = min_bound[i];
            max_bounds_buf(0, i) = max_bound[i];
        }

        // 生产者函数
        auto producer = [&](std::queue<Task>& task_queue) {
            for (int i = 0; i < node_len; ++i) {
                if (!decoded_list[i] && ready4decode[i]) {
                    std::lock_guard<std::mutex> lock(mtx);
                    task_queue.push({i, octree_buf(i, 2)});
                    decoded_list[i] = true;
                }
            }
        };

        // 结果存储
        std::vector<std::vector<double>> points;

        #pragma omp parallel
        {
            std::queue<Task> task_queue;
            while (true) {
                // 生产新任务
                if (task_queue.empty()) {
                    producer(task_queue);
                    if (task_queue.empty()) {
                        break;
                    }
                }

                // 消费任务
                Task task;
                {
                    std::lock_guard<std::mutex> lock(mtx);
                    if (!task_queue.empty()) {
                        task = task_queue.front();
                        task_queue.pop();
                    } else {
                        continue;
                    }
                }

                // 解码任务
                int node_id = task.node_id;
                int current_depth = task.depth;

                if (current_depth == depth - 1) {
                    continue; // 叶节点跳过
                }

                // 子节点边界计算
                std::vector<std::vector<double>> child_min_bounds(8, std::vector<double>(3));
                std::vector<std::vector<double>> child_max_bounds(8, std::vector<double>(3));
                for (int i = 0; i < 8; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        double size = (max_bounds_buf(node_id, j) - min_bounds_buf(node_id, j)) / 2.0;
                        child_min_bounds[i][j] = min_bounds_buf(node_id, j) + bound_affinity[i][j] * size;
                        child_max_bounds[i][j] = child_min_bounds[i][j] + size;
                    }
                }

                // 查找子节点
                #pragma omp critical
                {
                    for (ssize_t i = 0; i < node_len; ++i) {
                        if (octree_buf(i, 1) == node_id) {
                            int child_id = octree_buf(i, 0);
                            for (int j = 0; j < 3; ++j) {
                                min_bounds_buf(child_id, j) = child_min_bounds[octree_buf(i, 3)][j];
                                max_bounds_buf(child_id, j) = child_max_bounds[octree_buf(i, 3)][j];
                            }
                            ready4decode[child_id] = true;
                        }
                    }
                }
            }
        }

        // 叶节点处理
        #pragma omp parallel for
        for (ssize_t i = 0; i < node_len; ++i) {
            if (octree_buf(i, 2) == depth - 1) { // 叶节点
                std::vector<double> point(3);
                int symbol = octree_buf(i, 4);
                for (int j = 0; j < 8; ++j) {
                    if ((symbol >> (7 - j)) & 1) {
                        for (int k = 0; k < 3; ++k) {
                            point[k] = (min_bounds_buf(i, k) + max_bounds_buf(i, k)) / 2.0;
                        }
                        #pragma omp critical
                        points.push_back(point);
                    }
                }
            }
        }
        return points;
    }
};

PYBIND11_MODULE(octree_module, m) {
    py::class_<Octree>(m, "Octree")
        .def(py::init<py::array_t<int>, py::array_t<int>, size_t>(), py::arg("octree_data"), py::arg("ancestors_data"), py::arg("cache_size") = 64)
        .def("get_node_by_id", &Octree::get_node_by_id)
        .def("preview_next", &Octree::preview_next)
        .def("get_children_of_id", &Octree::get_children_of_id, py::arg("idx"), py::arg("empty_padding") = false)
        .def("get_parent_of_id", &Octree::get_parent_of_id)
        .def("get_n_parent_of_id", &Octree::get_n_parent_of_id, py::arg("idx"), py::arg("n") = 1)
        .def("get_root_node", &Octree::get_root_node)
        .def("get_n_children_of_id", &Octree::get_n_children_of_id, py::arg("idx"), py::arg("n") = 1, py::arg("empty_padding") = false)
        .def("get_parent_list_of_id", &Octree::get_parent_list_of_id, py::arg("idx")) // 注册新方法
        .def("get_context_window", &Octree::get_context_window, py::arg("idx"), py::arg("context_len") = 512) // 注册新方法
        // Define the iterator functions
        .def("__iter__", &Octree::__iter__, py::return_value_policy::reference_internal)  // Return self
        .def("__next__", &Octree::__next__);  // Define next() for Python iteration

    py::class_<Octree2Points>(m, "Octree2Points")
        .def(py::init<py::array_t<double>, py::array_t<double>, int>())
        .def("parally_convert", &Octree2Points::parally_convert);
}
