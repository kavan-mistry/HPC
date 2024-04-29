
#include <stdio.h>

__global__ void cuda_hello_one() {
    printf("Hello World from GPU !\n");
}

int main() {
    cuda_hello_one<<<1,4>>>();
    cudaDeviceSynchronize(); // Make sure all GPU work is done before exiting
    return 0;
}
