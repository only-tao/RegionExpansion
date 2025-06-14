#ifndef DENSITY_H
#define DENSITY_H
#include <vector>
// #include "cnpy.h"
#include <limits>
#include <cmath>
#include <cfloat>
#include <chrono>
#include <iostream>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>  // for AT_CUDA_CHECK
#include <c10/cuda/CUDAGuard.h>
struct Point {
    float x, y, z;
};
// std::vector<Point> readPointsFromNpy(const std::string& filename);
void initializeCudaMemory(Point* &d_points, bool* &d_selected, const std::vector<Point> &points);
void normalizePoints(std::vector<Point>& points);
// std::vector<int> processPoints(const std::vector<Point>& points, Point startPoint, float radius, int maxPoints, int numIterations);
std::vector<int> processPoints(const std::vector<Point>& points, int startidx, float radius, int maxPoints, int numIterations);
at::Tensor processPointsV1(
    const at::Tensor& points,
    int startidx, 
    const float radius, 
    const size_t maxPoints, 
    const size_t numIterations);
#endif // DENSITY_H