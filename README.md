# Latency cores vs throughput cores

Latency cores stand for CPU and throughput cores stand for GPU cores. This is due to the difference between GPUs and CPUs. 

## See Fig. 1
In CPUs we have a few number of powerful arithmetic logic units (ALU) for reducing the computation latency, a large cache memory for reducing the data accessing latency and a control unit for reducing the data flowing latency. 

In GPUs we have a large number of power efficient and slow ALUs and small cache memory units. In every data cycle ALUs throughput the outputs. The cache memory units different from CPUs is used for consolidating the data distribution between the ALUs. In CPUs there are a few registry 

while on GPUs there are many registry in order to make the threading process possible.

CPUs are used for the serial computations in which low latency is needed while GPUs are used for parallel computation in which high throughput is required. 

![CPU vs GPU](./images/cpu-gpu.png)

# What is CUDA?

CUDA is a programming API for hetregenous parallel programming. 
The idea of CUDA programming is different parts of the data can be analysed independently from each other. The parallel computing using CUDA is based on device+host, in which host means the CPU and device means the GPU. On the host the application is run in serial and on the device we have the parallel computation which its function is called kernel. A kernel is notated by <img src="https://latex.codecogs.com/svg.latex?\;Kernel%20%3C%3C%3C%20nBl,%20nTr%20%3E%3E%3E%20(args)"/>, in which <img src="https://latex.codecogs.com/svg.latex?\;args"/> is the input arguments, <img sr="https://latex.codecogs.com/svg.latex?\;nBl"/> and <img src="https://latex.codecogs.com/svg.latex?\;nTr"/> are the configurations of the grid of threads. 

![CPU vs GPU](./images/grids.png)
