#include "density.h"
#include <fstream>
#include <cuda_runtime.h>
#include <iostream>

void initializeCudaMemory(Point *&d_points, bool *&d_selected, const std::vector<Point> &points) {
    size_t size = points.size() * sizeof(Point);
    cudaMalloc(&d_points, size);
    cudaMemcpy(d_points, points.data(), size, cudaMemcpyHostToDevice);

    cudaMalloc(&d_selected, points.size() * sizeof(bool));
    cudaMemset(d_selected, 0, points.size() * sizeof(bool));
}
__global__ void selectPoints(Point *points, bool *isselected, int numPoints, Point startPoint, float radius, int *selectedPoints, int maxPoints) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints) return;
    if (selectedPoints[0] > maxPoints) return; 
    if (isselected[idx]) return;              

    Point p = points[idx];
    // printf("Point: (%f, %f, %f)\n", p.x, p.y, p.z);
    float dist = sqrtf((p.x - startPoint.x) * (p.x - startPoint.x) + (p.y - startPoint.y) * (p.y - startPoint.y) + (p.z - startPoint.z) * (p.z - startPoint.z));
    if (dist <= radius) {
        int pos = atomicAdd(selectedPoints, 1); 
        if (pos < maxPoints) {
            selectedPoints[pos + 1] = idx;
            isselected[idx] = true;        
        } else {
            atomicSub(selectedPoints, 1);
        }
    }
}
// input tensor, output tensor
void normalizePoints(at::Tensor &points) {
    if (points.size(0) == 0) return;
    float minX = FLT_MAX;
    float maxX = -FLT_MAX;
    float minY = FLT_MAX;
    float maxY = -FLT_MAX;
    float minZ = FLT_MAX;
    float maxZ = -FLT_MAX;
    for (int i = 0; i < points.size(0); i++) {
        if (points[i][0].item<float>() < minX) minX = points[i][0].item<float>();
        if (points[i][0].item<float>() > maxX) maxX = points[i][0].item<float>();
        if (points[i][1].item<float>() < minY) minY = points[i][1].item<float>();
        if (points[i][1].item<float>() > maxY) maxY = points[i][1].item<float>();
        if (points[i][2].item<float>() < minZ) minZ = points[i][2].item<float>();
        if (points[i][2].item<float>() > maxZ) maxZ = points[i][2].item<float>();
    }
    for (int i = 0; i < points.size(0); i++) {
        points[i][0] = (points[i][0].item<float>() - minX) / (maxX - minX);
        points[i][1] = (points[i][1].item<float>() - minY) / (maxY - minY);
        points[i][2] = (points[i][2].item<float>() - minZ) / (maxZ - minZ);
    }
}
void normalizePoints(std::vector<Point> &points) {
    if (points.empty()) return;

    float minX = FLT_MAX;
    float maxX = -FLT_MAX;
    float minY = FLT_MAX;
    float maxY = -FLT_MAX;
    float minZ = FLT_MAX;
    float maxZ = -FLT_MAX;

    for (const auto &point : points) {
        if (point.x < minX) minX = point.x;
        if (point.x > maxX) maxX = point.x;
        if (point.y < minY) minY = point.y;
        if (point.y > maxY) maxY = point.y;
        if (point.z < minZ) minZ = point.z;
        if (point.z > maxZ) maxZ = point.z;
    }

    for (auto &point : points) {
        point.x = (point.x - minX) / (maxX - minX);
        point.y = (point.y - minY) / (maxY - minY);
        point.z = (point.z - minZ) / (maxZ - minZ);
    }
}
std::vector<int> processPoints(const std::vector<Point> &points, Point startPoint, float radius, int maxPoints, int numIterations) {
    std::vector<Point> normalizedPoints = points;
    // normalizePoints(normalizedPoints);
    // startPoint = normalizedPoints[10050];
    Point *d_points;
    bool *d_selected;
    initializeCudaMemory(d_points, d_selected, normalizedPoints);

    int *d_selectedPoints;
    if (cudaMalloc(&d_selectedPoints, (maxPoints + 1) * sizeof(int)) == cudaErrorMemoryAllocation) {
        std::cerr << "Error: cudaMalloc failed" << std::endl;
        return {};
    }

    int numPoints = normalizedPoints.size();
    int threadsPerBlock = 128;
    int blocksPerGrid = (numPoints + threadsPerBlock - 1) / threadsPerBlock;

    std::vector<int> selectedPoints(maxPoints + 1);

    int numSelectedPoints = 1;
    cudaMemcpy(d_selectedPoints, &numSelectedPoints, sizeof(int), cudaMemcpyHostToDevice);
    int pre_numSelectedPoint = 1;
    for (int iter = 0; iter < numIterations; iter++) {
        for (int i = pre_numSelectedPoint; i <= numSelectedPoints; i++) {
            selectPoints<<<blocksPerGrid, threadsPerBlock>>>(d_points, d_selected, numPoints, startPoint, radius, d_selectedPoints, maxPoints);
            startPoint = normalizedPoints[selectedPoints[i]];

            cudaDeviceSynchronize(); 
        }
        pre_numSelectedPoint = numSelectedPoints;
        cudaError_t status = cudaMemcpy(selectedPoints.data(), d_selectedPoints, (maxPoints + 1) * sizeof(int), cudaMemcpyDeviceToHost);
        if (status != cudaSuccess) {
            std::cerr << "Error: " << cudaGetErrorString(status) << std::endl;
            return {};
        }

        numSelectedPoints = selectedPoints[0];
        std::cout << "numSelectedPoints: " << numSelectedPoints << std::endl;
        if (numSelectedPoints <= 0 || numSelectedPoints > maxPoints) {
            std::cerr << "Error: numSelectedPoints out of range: " << numSelectedPoints << std::endl;
            break;
        } else if (numSelectedPoints == maxPoints) {
            std::cout << "numSelectedPoints == maxPoints" << std::endl;
            break;
        }
    }

    std::vector<int> finalSelectedPoints(maxPoints + 1);
    cudaMemcpy(finalSelectedPoints.data(), d_selectedPoints, (maxPoints + 1) * sizeof(int), cudaMemcpyDeviceToHost);

    int numFinalSelectedPoints = numSelectedPoints;
    std::cout << "last Number of selected points: " << numFinalSelectedPoints << std::endl;
    cudaFree(d_points);
    cudaFree(d_selected);
    cudaFree(d_selectedPoints);
    std::vector<int> selectedIndices(finalSelectedPoints.begin() + 1, finalSelectedPoints.end());
    return selectedIndices;
}
// read parameter: startidx,outputfilenameidx
// reload the processPoints function
std::vector<int> processPoints(const std::vector<Point> &points, int startidx, float radius, int maxPoints, int numIterations) {
    std::vector<Point> normalizedPoints = points;
    Point startPoint;
    // Point startPoint = normalizedPoints[startidx];
    Point *d_points;
    bool *d_selected;
    initializeCudaMemory(d_points, d_selected, normalizedPoints);

    int *d_selectedPoints;
    if (cudaMalloc(&d_selectedPoints, (maxPoints + 1) * sizeof(int)) == cudaErrorMemoryAllocation) {
        std::cerr << "Error: cudaMalloc failed" << std::endl;
        return {};
    }

    int numPoints = normalizedPoints.size();
    int threadsPerBlock = 128;
    int blocksPerGrid = (numPoints + threadsPerBlock - 1) / threadsPerBlock;

    std::vector<int> selectedPoints(maxPoints + 1);
    selectedPoints[1] = startidx;
    int numSelectedPoints = 1;
    cudaMemcpy(d_selectedPoints, &numSelectedPoints, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_selectedPoints + 1, &startidx, sizeof(int), cudaMemcpyHostToDevice);
    int pre_numSelectedPoint = 0;
    // cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    for (int iter = 0; iter < numIterations; iter++) {
        for (int i = pre_numSelectedPoint + 1; i <= numSelectedPoints; i++) {
            // std::cout << startPoint.x << " " << startPoint.y << " " << startPoint.z << std::endl;
            startPoint = normalizedPoints[selectedPoints[i]];
            selectPoints<<<blocksPerGrid, threadsPerBlock>>>(d_points, d_selected, numPoints, startPoint, radius, d_selectedPoints, maxPoints);
            cudaDeviceSynchronize(); 
        }
        pre_numSelectedPoint = numSelectedPoints;
        cudaError_t status = cudaMemcpy(selectedPoints.data(), d_selectedPoints, (maxPoints + 1) * sizeof(int), cudaMemcpyDeviceToHost);
        if (status != cudaSuccess) {
            std::cerr << "Error: " << cudaGetErrorString(status) << std::endl;
            return {};
        }

        numSelectedPoints = selectedPoints[0];
        std::cout << "numSelectedPoints: " << numSelectedPoints << std::endl;
        if (numSelectedPoints <= 0 || numSelectedPoints > maxPoints) {
            std::cerr << "Error: numSelectedPoints out of range: " << numSelectedPoints << std::endl;
        } else if (numSelectedPoints == maxPoints) {
            std::cout << "numSelectedPoints == maxPoints" << std::endl;
            break;
        }
    }
    // std::vector<int> finalSelectedPoints(maxPoints + 1);
    // cudaMemcpy(finalSelectedPoints.data(), d_selectedPoints, (maxPoints + 1) * sizeof(int), cudaMemcpyDeviceToHost);

    int numFinalSelectedPoints = numSelectedPoints;
    std::cout << "last Number of selected points: " << numFinalSelectedPoints << std::endl;
    cudaFree(d_points);
    cudaFree(d_selected);
    cudaFree(d_selectedPoints);
    // AT_CUDA_CHECK(cudaGetLastError());

    std::vector<int> selectedIndices(selectedPoints.begin() + 1, selectedPoints.end());
    return selectedIndices;
}
// return std::vector<int>
// template <typename scalar_t>

// reload the global selectPoints function , input is Tensor !!!
template <typename scalar_t>
__global__ void selectPointsV1(
    const scalar_t *__restrict__ points,
    bool *__restrict__ isselected,
    const size_t numPoints,
    // size_t startidx,
    const float radiusSquare,
    int *selectedPoints,
    const size_t maxPoints,
    int pre_numSelectedPoint,
    int numSelectedPoints) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // int pre_numSelectedPoint =  selectedPoints[0];
    if (idx >= numPoints || pre_numSelectedPoint > maxPoints || isselected[idx]) return;
    for (int i = pre_numSelectedPoint + 1; i <= numSelectedPoints; i++) { // 0 1 ; 1...x ; x...y
        int startidx = selectedPoints[i];
        scalar_t x1 = points[startidx * 3];
        scalar_t y1 = points[startidx * 3 + 1];
        scalar_t z1 = points[startidx * 3 + 2];
        scalar_t x2 = points[idx * 3];
        scalar_t y2 = points[idx * 3 + 1];
        scalar_t z2 = points[idx * 3 + 2];
        scalar_t dist = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2);
        if (dist <= radiusSquare) {
            int pos = atomicAdd(selectedPoints, 1); 

            if (pos < maxPoints) {
                selectedPoints[pos + 1] = idx;
                isselected[idx] = true;        
                break;
            } else {
                atomicSub(selectedPoints, 1);
            }
        }
        // __syncthreads();
    }
}

at::Tensor processPointsV1(
    const at::Tensor &points,
    int startidx,
    const float radius,
    const size_t maxPoints,
    const size_t numIterations) {
    // check
    at::CheckedFrom c = "processPoints";

    // set the device for the kernel launch based on the device of the input tensor
    at::cuda::CUDAGuard device_guard(points.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const size_t numPoints = points.size(0);
    const size_t D = points.size(1);
    TORCH_CHECK(D == 3, "points should have 3 columns");
    TORCH_CHECK(radius > 0, "radius should be in (0, ");
    TORCH_CHECK(maxPoints > 0 && maxPoints < numPoints, "maxPoints should be in (0, numPoints)");
    TORCH_CHECK(startidx >= 0 && startidx < numPoints, "startidx should be in [0, numPoints)");
    const int threadsPerBlock = 128;
    const int blocksPerGrid = (numPoints + threadsPerBlock - 1) / threadsPerBlock;

    at::Tensor isselected = at::zeros({static_cast<long>(numPoints)}, points.options().dtype(at::kBool)); 
    at::Tensor d_selectedPoints = at::zeros({static_cast<long>(maxPoints + 1)}, points.options().dtype(at::kInt));

    at::TensorArg points_t{points, "points", 1}, isselected_t{isselected, "isselected", 2}, d_selectedPoints_t{d_selectedPoints, "d_selectedPoints", 3};
    at::checkAllSameGPU(c, {points_t, isselected_t, d_selectedPoints_t});
    // at::checkAllSameType(c, {points_t, isselected, d_selectedPoints});
    if (d_selectedPoints.numel() == 0) {
        AT_CUDA_CHECK(cudaGetLastError());
        std::cerr << "Error: cudaMalloc failed" << std::endl;
        return d_selectedPoints;
    }
    int pre_numSelectedPoint = 0;
    int numSelectedPoints = 1;
    float radiusSquare = radius * radius;
    // set numSelectedPoints to the first element of selectedPoints
    AT_CUDA_CHECK(cudaMemcpy(d_selectedPoints.data_ptr<int>(), &numSelectedPoints, sizeof(int), cudaMemcpyHostToDevice));
    // set the startidx to the second element of selectedPoints
    AT_CUDA_CHECK(cudaMemcpy(d_selectedPoints.data_ptr<int>() + 1, &startidx, sizeof(int), cudaMemcpyHostToDevice));
    cudaMemcpy(isselected.data_ptr<bool>() + startidx, &numSelectedPoints, sizeof(bool), cudaMemcpyHostToDevice);
#if DEBUG
    // std::cout << "begin AT_DISPATCH_ALL_TYPES" << std::endl;
#endif
    AT_DISPATCH_ALL_TYPES(
        points.scalar_type(), "processPoints", ([&] {
            for (int iter = 0; iter < numIterations; iter++) {
                    selectPointsV1<scalar_t><<<threadsPerBlock, blocksPerGrid, 0, stream>>>(
                    points.contiguous().data_ptr<scalar_t>(),
                    isselected.data_ptr<bool>(),
                    numPoints,
                    // startidx,
                    radiusSquare,
                    d_selectedPoints.data_ptr<int>(),
                    maxPoints,
                    pre_numSelectedPoint,
                    numSelectedPoints);
                pre_numSelectedPoint = numSelectedPoints;
                // copy the selectedPoints[0] from device to host
                AT_CUDA_CHECK(
                    cudaMemcpy(&numSelectedPoints, d_selectedPoints.data_ptr<int>(), sizeof(int), cudaMemcpyDeviceToHost));
#if DEBUG
                std::cout << "numSelectedPoints: " << numSelectedPoints << std::endl;
#endif
                if (numSelectedPoints <= 0 || numSelectedPoints > maxPoints) {
                    std::cerr << "Error: numSelectedPoints out of range: " << numSelectedPoints << std::endl;
                } else if (numSelectedPoints == maxPoints) {
#if DEBUG
                    std::cout << "numSelectedPoints == maxPoints" << std::endl;
#endif
                    break;
                }
            }
        }));
    // AT_CUDA_CHECK(cudaGetLastError());
    // return d_selectedPoints[1:]
    return d_selectedPoints.slice(0, 1, maxPoints + 1);
    // return d_selectedPoints;
}
// reload the global selectPoints function , input is Tensor !!!

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./density <startidx> <outputfilenameidx>" << std::endl;
        return 1;
    }

    int startidx = std::stoi(argv[1]);
    std::string outputfilenameidx = argv[2];
    std::cout << "startidx: " << startidx << std::endl;
    std::cout << "outputfilenameidx: " << outputfilenameidx << std::endl;
    // points read from file, npy file (python numpy)
    std::string filename = "~/Data/mpeg/soldier/fps/soldier_vox10_0536.npy";
    // at::GlobalContext::manual_seed(0);
    at::TensorOptions options(at::kCUDA);
    at::Tensor points = at::rand({30000, 3}, options);
    // at::Tensor points = at::from_file(filename);
    if (at::cuda::is_available()) {
        points = points.to(at::kCUDA);
    }

    float radius = 0.03f;
    int maxPoints = 1000;
    int numIterations = 22; 
    at::Tensor selectedIndices;
    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "begin to processPointsV1" << std::endl;

    try {
        selectedIndices = processPointsV1(points, startidx, radius, maxPoints, numIterations);
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    std::cout << "end to processPointsV1" << std::endl;

    std::ofstream file("selected_points" + outputfilenameidx + ".txt");
    // selectedIndices to cpus
    selectedIndices = selectedIndices.to(at::kCPU);
    std::vector<int> selectedIndices_vec(selectedIndices.data_ptr<int>(), selectedIndices.data_ptr<int>() + selectedIndices.size(0));
    if (file.is_open()) {
        for (int idx : selectedIndices_vec) {
            file << idx << std::endl;
        }
        file.close();
    } else {
        std::cerr << "Unable to open file for writing" << std::endl;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " s" << std::endl;

    return 0;
}