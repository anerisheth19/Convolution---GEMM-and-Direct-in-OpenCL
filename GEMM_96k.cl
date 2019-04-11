
/*Kernel for GEMM without local memory - SqueezeNet*/
 
__kernel void convolute(__global unsigned char* output,  __global unsigned char* image, __global unsigned char* filter, int filter_size, int rows, int cols, int kf)
{
  
   int tx = get_global_id(0); 
   int ty = get_global_id(1);
 
   int sum = 0;
   for (int k = 0; k < (filter_size * filter_size); ++k)
   {
      int a = filter[(tx*kf) + k]; //filter
      int b = image[k*(rows-filter_size+1)*(cols-filter_size+1)*3 + tx+ty]; //input
      sum += a + b; //convolute add
   }
 
   output[ty+ (kf*tx)] = sum; //accumulate sum
}
