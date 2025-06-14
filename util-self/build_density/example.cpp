#include "../density.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h> // Include the "numpy.h" header file
#include <pybind11/stl.h>
#include <torch/extension.h>
namespace py = pybind11;
// using namespace pybind11;

int add(int i, int j) {
    return i + j;
}

std::vector<int> process_points(py::array_t<float> input_array, int startidx, float radius, int max_points, int num_iterations) 
{
    py::buffer_info buf = input_array.request();
    float* ptr = static_cast<float*>(buf.ptr);
    size_t num_points = buf.shape[0];

    std::vector<Point> points(num_points);
    for (size_t i = 0; i < num_points; ++i) {
        points[i].x = ptr[i * 3];
        points[i].y = ptr[i * 3 + 1];
        points[i].z = ptr[i * 3 + 2];
    }

    return processPoints(points,  startidx, radius, max_points, num_iterations);
}
// reload the process_points function
at::Tensor process_pointsV1(
    const at::Tensor& points,
    size_t startidx, 
    const float radius, 
    const size_t maxPoints, 
    const size_t numIterations)
{
    return processPointsV1(points, startidx, radius, maxPoints, numIterations);
}
    
PYBIND11_MODULE(example, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("add", &add, "A function that adds two numbers");
    m.def("processPoints", &processPoints, "Process points");
    m.def("process_points", &process_points, "Process points",
          py::arg("input_array"), py::arg("startidx"), py::arg("radius"), py::arg("max_points"), py::arg("num_iterations"));
    m.def("process_pointsV1", &process_pointsV1, "Process points");
}
