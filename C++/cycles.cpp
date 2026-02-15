#include <x86intrin.h>

#include <chrono>
#include <iostream>
int main() {
  auto start_time = std::chrono::steady_clock::now();
  unsigned long long start = __rdtsc();

  // for (int i = 0; i < 1000000; i++) {
  //    std::cout << "hello world\n";
  // }
 
  volatile unsigned int c;

for (unsigned long long i = 0; i < 1000000ULL; i++){
   // int a = 10;
   // int b = 20;
   // c = a + b;
  std::cout<<"hello world\n";
}

  unsigned long long end = __rdtsc();

  auto end_time = std::chrono::steady_clock::now();
  auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                                    start_time);
  std::cout << "Elapsed time: " << diff.count() << " ms" << std::endl;
  std::cout << "Cycles: " << (end - start) << "\n";
}
