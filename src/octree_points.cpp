#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <mutex>
#include <queue>
#include <atomic>
#include <omp.h>
#include <iostream>

namespace py = pybind11;

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

PYBIND11_MODULE(octree_bind, m) {
    py::class_<Octree2Points>(m, "Octree2Points")
        .def(py::init<py::array_t<double>, py::array_t<double>, int>())
        .def("parally_convert", &Octree2Points::parally_convert);
}
