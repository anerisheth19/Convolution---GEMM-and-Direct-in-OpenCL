//ECGR 6090 Heterogeneous Computing Homework 2
// Problem 3  - GEMM in OpenCL for SqueezeNet
//Written by Aneri Sheth - 801085402
//Reference taken from https://github.com/cnugteren/myGEMM and Slides by Dr. Tabkhi

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <CL/cl.h>
#include <stdbool.h>

#define H 120
#define W 160
#define K 7

int decode_image(char* frame, char filename[]);
void seperateChannels(unsigned char* imd,unsigned char* im1,unsigned char* im2,unsigned char* im3); 
int im2col_cpu(unsigned char* data_im, int channels, int height, int width, int ksize, int stride, int pad, unsigned char* data_col);
unsigned char im2col_get_pixel(unsigned char *im, int height, int width, int channels, int row, int col, int channel, int pad);
void read_filters(int *filters);

int decode_image(char* frame, char filename[]) {
    FILE *pFile;
    pFile = fopen(filename, "r");
    fseek(pFile, 15, SEEK_SET);
    fread(frame, sizeof(char), H*W*3, pFile);   
    fclose(pFile);
    return 0; 
}

void read_filters(int *filters) 
{

	FILE *fp;	
   	char num_buff[255];
	double n;
   	fp = fopen("snweights-ints3.txt", "r");
	int sizeInt = 7*7*3*96*sizeof(int);
	int i=0;
	for(i=1;i<7*7*3*96+1;i++)
	{	
		fscanf(fp, "%s", num_buff);
		n = atof(num_buff);
		filters[i-1]=n;
	}
       fclose(fp);
}

void seperateChannels(unsigned char* imd,unsigned char* im1,unsigned char* im2,unsigned char* im3){
    int i,j;    
    for(i=0,j=0; i<H*W; i++,j+=3){
        im1[i] = imd[j];
        im2[i] = imd[j+1];
        im3[i] = imd[j+2];                
    }
}
int im2col_cpu(unsigned char* data_im,
     int channels,  int height,  int width,
     int ksize,  int stride, int pad, unsigned char* data_col) 
{
    int c,h,w;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;
    int col_index;
    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
            }
        }
    }
    return col_index;
}
unsigned char im2col_get_pixel(unsigned char *im, int height, int width, int channels, int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||row >= height || col >= width)   return 0;
    return im[col + width*(row + height*channel)];
}

long LoadOpenCLKernel(char const* path, char **buf)
{
    FILE  *fp;
    size_t fsz;
    long   off_end;
    int    rc;

    /* Open the file */
    fp = fopen(path, "r");
    if( NULL == fp ) {
        return -1L;
    }

    /* Seek to the end of the file */
    rc = fseek(fp, 0L, SEEK_END);
    if( 0 != rc ) {
        return -1L;
    }

    /* Byte offset to the end of the file (size) */
    if( 0 > (off_end = ftell(fp)) ) {
        return -1L;
    }
    fsz = (size_t)off_end;

    /* Allocate a buffer to hold the whole file */
    *buf = (char *) malloc( fsz+1);
    if( NULL == *buf ) {
        return -1L;
    }

    /* Rewind file pointer to start of file */
    rewind(fp);

    /* Slurp file into buffer */
    if( fsz != fread(*buf, 1, fsz, fp) ) {
        free(*buf);
        return -1L;
    }

    /* Close the file */
    if( EOF == fclose(fp) ) {
        free(*buf);
        return -1L;
    }


    /* Make sure the buffer is NUL-terminated, just in case */
    (*buf)[fsz] = '\0';

    /* Return the file size */
    return (long)fsz;
}

int main(int argc, char** argv)
{
   int err;                            // error code returned from api calls

   cl_device_id device_id;             // compute device id 
   cl_context context;                 // compute context
   cl_command_queue commands;          // compute command queue
   cl_program program;                 // compute program
   cl_kernel kernel;                   // compute kernel
   cl_event myevent;
   cl_ulong start;
   cl_ulong end;
   cl_float kernelExecTimeNs;

    // OpenCL device memory for matrices
   cl_mem d_image;
   cl_mem d_filter;
   cl_mem d_C;
 
   int imagecount = 0;
printf("Initializing OpenCL device...\n");

   cl_uint dev_cnt = 0;
   clGetPlatformIDs(0, 0, &dev_cnt);

   cl_platform_id platform_ids[100];
   clGetPlatformIDs(dev_cnt, platform_ids, NULL);

   // Connect to a compute device
       int gpu = 1;
          err = clGetDeviceIDs(platform_ids[0], gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
             if (err != CL_SUCCESS)
                {
                       printf("Error: Failed to create a device group!\n");
                              return EXIT_FAILURE;
                                 }
   
                                    // Create a compute context 
        context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
                                          if (!context)
                                             {
                                                    printf("Error: Failed to create a compute context!\n");
                                                           return EXIT_FAILURE;
                                                              }
                                                               
   while(imagecount<120)
  {
    
    char numbuff[12];
    snprintf (numbuff, sizeof(numbuff), "%d",imagecount);
    //printf("%s\n",numbuff);
    char filebuff[100];
    strcpy(filebuff,"viptraffic");
    strcat(filebuff,numbuff);
    strcat(filebuff,".ppm");
    imagecount++;
    
    //decode_image(image,filebuff); //call the read function
 

   //Allocate host memory for image with 3 channels
   unsigned int size_image = W * H * K;
   unsigned int mem_size_image = sizeof(unsigned char) * size_image;
   unsigned char* image = (unsigned char*) malloc(mem_size_image);

   //Allocate host memory for filter
   unsigned int size_filter = K * K*96*3;
   unsigned int mem_size_filter = sizeof(int) * size_filter;
   int* h_filter = (int*) malloc(mem_size_filter);

   //Allocate host memory for filter
   unsigned int size_op_im2col = K*K*(H)*(W)*3;
   unsigned int mem_size_op_im2col = sizeof(unsigned char) * size_op_im2col;
   unsigned char* h_op_im2col = (unsigned char*) malloc(mem_size_op_im2col);   
   
   decode_image(image, filebuff);
   int i,j;   
   read_filters(h_filter);
   im2col_cpu(image,1,H,W*3,K,2,0,h_op_im2col);


   //Allocate host memory for the result C
   unsigned int size_C = H * W * 3 * 96;
   unsigned int mem_size_C = sizeof(unsigned char) * size_C;
   unsigned char* h_C = (unsigned char*) malloc(mem_size_C);
 
   // Create a command commands
   commands = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
   if (!commands)
   {
       printf("Error: Failed to create a command commands!\n");
       return EXIT_FAILURE;
   }

   // Create the compute program from the source file
   char *KernelSource;
   long lFileSize;

   lFileSize = LoadOpenCLKernel("GEMM_96k.cl", &KernelSource);
   if( lFileSize < 0L ) {
       perror("File read failed");
       return 1;
   }

   program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource, NULL, &err);
   if (!program)
   {
       printf("Error: Failed to create compute program!\n");
       return EXIT_FAILURE;
   }

   // Build the program executable
   err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
   if (err != CL_SUCCESS)
   {
       size_t len;
       char buffer[2048];
       printf("Error: Failed to build program executable!\n");
       clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
       printf("%s\n", buffer);
       exit(1);
   }

   // Create the compute kernel in the program we wish to run
   //
   kernel = clCreateKernel(program, "convolute", &err);
   if (!kernel || err != CL_SUCCESS)
   {
       printf("Error: Failed to create compute kernel!\n");
       exit(1);
   }

   // Create the input and output arrays in device memory for our calculation
   d_C = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size_C, NULL, &err);
   d_image = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_op_im2col, h_op_im2col, &err);
   d_filter = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_filter, h_filter, &err);

   if (!d_image || !d_filter || !d_C)
   {
       printf("Error: Failed to allocate device memory!\n");
       exit(1);
   }    
    
   //Launch OpenCL kernel
   size_t localWorkSize[2], globalWorkSize[2];
 
   int wK = K;
   int wH = H;
   int wW = W;
   int wK_H = 96;
   err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_C);
   err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_image);
   err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_filter);
   err |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&wK);
   err |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&wH);
   err |= clSetKernelArg(kernel, 5, sizeof(int), (void *)&wW);
   err |= clSetKernelArg(kernel, 6, sizeof(int), (void *)&wK_H);


   if (err != CL_SUCCESS)
   {
       printf("Error: Failed to set kernel arguments! %d\n", err);
       exit(1);
   }
   //set the local and globar work group size 
   localWorkSize[0] = 8;
   localWorkSize[1] = 9;
   globalWorkSize[0] = 96*3;
   globalWorkSize[1] = 13509;

   err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, &myevent);
   clFinish(commands);
   clGetEventProfilingInfo(myevent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
   clGetEventProfilingInfo(myevent, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
   kernelExecTimeNs += end-start;
   
   if (err != CL_SUCCESS)
   {
       printf("Error: Failed to execute kernel! %d\n", err);
       exit(1);
   }
   printf("Kernel Execution Time:  %f \n", kernelExecTimeNs/1000000000);
  //Retrieve result from device
   err = clEnqueueReadBuffer(commands, d_C, CL_TRUE, 0, mem_size_C, h_C, 0, NULL, NULL);
   clFinish(commands);

   if (err != CL_SUCCESS)
   {
       printf("Error: Failed to read output array! %d\n", err);
       exit(1);
   }

   //Shutdown and cleanup
   free(image);
   free(h_filter);
   free(h_C);
   free(h_op_im2col);

 
   clReleaseMemObject(d_image);
   clReleaseMemObject(d_filter);
   clReleaseMemObject(d_C);

   clReleaseProgram(program);
   clReleaseKernel(kernel);
   clReleaseCommandQueue(commands);
   clReleaseContext(context);
}
   return 0;
}
