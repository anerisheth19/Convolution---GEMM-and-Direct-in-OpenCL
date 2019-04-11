
/* Kernel for Direct Convolution with local memory*/

__kernel void convolute(__global unsigned char* output, __global unsigned char* image, __global int* filter, int rows, int cols, int filter_size ) {

	
	__local unsigned char local_mem_block[100];
	int temp = (filter_size)/2; //half filter size
	int padding = temp * 2;

	int g1 = get_group_id(0) * get_local_size(0); 
	int g2 = get_group_id(1) * get_local_size(1);
	
	int localCol = get_local_id(0);
	int localRow = get_local_id(1);
	
	int globalCol = g1 + localCol;
	int globalRow = g2 + localRow;
	int i,j,k,l;

	for(i = localRow; i < 10; i+=get_local_size(1)) { //rows
		int currRow = g2 + i;
		for(j = localCol; j < 10; j+=get_local_size(0)) { //cols
			int currCol = g1 + j;

			if(currRow < rows && currCol < cols) {
				local_mem_block[i * 10 + j] = image[currRow * cols + currCol];
			}
		}
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	if(globalRow < rows - padding && globalCol < cols - padding) { //kernel rows
		int sum = 0;
		int f_in=0;

		for(k = localRow; k < localRow + filter_size; k++) { //kernel cols
			int offset = k * 10;
			for(l = localCol; l < localCol + filter_size; l++) {
				sum += local_mem_block[offset + l] * filter[f_in++]; //sum
			}
		}
		output[(globalRow + temp) * cols + (globalCol + temp)] = sum; //accumulate sum
	}

}
