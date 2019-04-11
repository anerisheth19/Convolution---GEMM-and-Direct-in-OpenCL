
/* Kernel for GEMM with local memory*/ 

__kernel void convolute(__global unsigned char* output, __global unsigned char* image, __global unsigned char* filter, int filter_size, int rows, int cols)
{
    __local unsigned char local_mem_block[1000];
    int g1 = get_group_id(0) * get_local_size(0); 
    int g2 = get_group_id(1) * get_local_size(1);
	
    int localCol = get_local_id(0);
    int localRow = get_local_id(1);
	
    int globalCol = g1 + localCol;
    int globalRow = g2 + localRow;
    int i,j,k,l;  
   
   for(i = localRow; i < 10; i+=get_local_size(1)) {
	int currRow = g2 + i;
	for(j = localCol; j < 10; j+=get_local_size(0)) {
		int currCol = g1 + j;
		if(currRow < rows && currCol < cols) {
			local_mem_block[i * 10 + j] = image[currRow * cols + currCol];
		}
	}
   } 
   barrier(CLK_LOCAL_MEM_FENCE);

   int sum = 0;
   for (int k = 0; k < (filter_size* filter_size); ++k)
   {
      int a = filter[k];
      int b = local_mem_block[k*10 + localCol];
      sum += a+b;

   }
    barrier(CLK_LOCAL_MEM_FENCE);

   output[globalRow + globalCol] = sum;
}
