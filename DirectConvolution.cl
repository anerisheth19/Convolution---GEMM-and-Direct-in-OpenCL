
/*Kernel for Direct Convolution without local memory*/

__kernel void convolute(__global unsigned char* output, __global unsigned char* image, __global int* filter, int rows, int cols, int filter_size ) 
{	
	int tx = get_global_id(0);
	int ty = get_global_id(1);

	int temp = (filter_size)/2;  //half filter size

	int sum = 0; //sum
	int x_in = 0; //index of x
	int y_in = 0; //index of y
	int f_in = 0; //filter index

	int i,j;
	for(i = -temp; i<= temp; i++){ //rows of filter
		y_in = ty + i;
		for(j = -temp; j<= temp; j++,f_in++){ //cols of filter
			x_in = tx + j;
			if (y_in < 0 || x_in < 0) 
				sum +=  0 * filter[f_in]; //padding	
			else
 				sum +=  image[y_in * cols + x_in] * filter[f_in]; //convolution operation
		}
	}
	output[ty * cols + tx] = sum; //accumulate sum in final output
}
