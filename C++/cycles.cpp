#include <x86intrin.h>
#include <chrono>
#include <iostream>
#include <vector>

void test_size(size_t N) {
    size_t total = N * N;

    std::vector<int> A(total, 1);
    std::vector<int> B(total, 2);
    std::vector<int> C(total);

    std::cout << "\n=============================\n";
    std::cout << "Size: " << N << " x " << N << "\n";

    // =========================
    // ROW-MAJOR ACCESS
    // =========================
    {
        auto start_time = std::chrono::steady_clock::now();
        unsigned long long start = __rdtsc();

        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j < N; j++) {
                C[i*N + j] = A[i*N + j] + B[i*N + j];
            }
        }

        unsigned long long end = __rdtsc();
        auto end_time = std::chrono::steady_clock::now();

        auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);

        std::cout << "\nRow-major:\n";
        std::cout << "Elapsed time: " << diff.count() << " ms\n";
        std::cout << "Cycles: " << (end - start) << "\n";
        std::cout << "Cycles per element: "
                  << (double)(end - start) / total << "\n";
    }

    // =========================
    // COLUMN-MAJOR ACCESS
    // =========================
    {
        auto start_time = std::chrono::steady_clock::now();
        unsigned long long start = __rdtsc();

        for (size_t j = 0; j < N; j++) {
            for (size_t i = 0; i < N; i++) {
                C[i*N + j] = A[i*N + j] + B[i*N + j];
            }
        }

        unsigned long long end = __rdtsc();
        auto end_time = std::chrono::steady_clock::now();

        auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);

        std::cout << "\nColumn-major:\n";
        std::cout << "Elapsed time: " << diff.count() << " ms\n";
        std::cout << "Cycles: " << (end - start) << "\n";
        std::cout << "Cycles per element: "
                  << (double)(end - start) / total << "\n";
    }
}

int main() {
    test_size(100);
    test_size(1000);
    test_size(10000);
    test_size(50000);
}

