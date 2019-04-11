//ECGR 6090 Heterogeneous Computing Homework 2
// Problem 1  - Direct Convolution in OpenCL
//Written by Aneri Sheth - 801085402
//Reference taken from Lecture Slides by Dr. Tabkhi and code given by TA Arnab Purkayastha

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

#define HEIGHT 160
#define WIDTH 120
#define K 9

unsigned char image[HEIGHT * WIDTH * K]; //image with 3 input channels RGB
int decode_image(unsigned char frame[HEIGHT * WIDTH * K], char filename[]); 

//Function to read the image files in C
int decode_image(unsigned char frame[HEIGHT * WIDTH * K],char filename[])
{
  FILE *pFile;
      pFile = fopen(filename, "r"); //read mode
      fseek(pFile, 0, SEEK_SET);
      fread(frame, sizeof(unsigned char), HEIGHT * WIDTH * K, pFile);   
      fclose(pFile);
      return 0;
}

//Function to load OpenCL kernel - taken from code given by T.A. Arnab 
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


//This is the main function
int main(int argc, char** argv) {

	//define memory for inputs and kernel
	int* filter = (int*) malloc(K*K*sizeof(int));
	unsigned char* image_r = (unsigned char*) malloc(HEIGHT * WIDTH * sizeof(unsigned char)); //R channel
	unsigned char* image_g = (unsigned char*) malloc(HEIGHT * WIDTH * sizeof(unsigned char)); //G channel
	unsigned char* image_b = (unsigned char*) malloc(HEIGHT * WIDTH * sizeof(unsigned char)); //B channel
	int a;
	int imagecount  = 0; //counter for 120 images
	for (a = 0; a < K * K; a++){
   		if (a == 40)
			filter[a] = 2;
		else 
			filter[a] = 3;
		
	}
	
	int err;
	cl_device_id device_id;             // compute device id 
   	cl_context context;                 // compute context
   	cl_command_queue commands;          // compute command queue
   	cl_program program;                 // compute program
   	cl_kernel kernel;                   // compute kernel

   	cl_mem d_image; //input image
   	cl_mem d_filter; //filter
   	cl_mem d_output; //output image
	cl_event myevent; //timing - profiling
	cl_ulong start; //event start
	cl_ulong end; //event stop
	cl_float kernelExecTimeNs; //measure time
   	
 	unsigned char* output_r = (unsigned char*) malloc(HEIGHT * WIDTH * sizeof(unsigned char)); //R output
        unsigned char* output_g = (unsigned char*) malloc(HEIGHT * WIDTH * sizeof(unsigned char)); //G output
        unsigned char* output_b = (unsigned char*) malloc(HEIGHT * WIDTH * sizeof(unsigned char)); //B output
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

	   lFileSize = LoadOpenCLKernel("DirectConvolution.cl", &KernelSource);
	   if( lFileSize < 0L ) {
	       perror("File read failed");
	       return 1;
	   }

	   program = clCreateProgramWithSource(context, 1, (const char **) &KernelSource, NULL, &err);
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
	   kernel = clCreateKernel(program, "convolute", &err);
	   if (!kernel || err != CL_SUCCESS)
	   {
	       printf("Error: Failed to create compute kernel!\n");
	       exit(1);
	   }

	   //do this for all images 
	   while(imagecount<120) {
	  
	    int count = 0; 
	    char numbuff[12];
	    snprintf (numbuff, sizeof(numbuff), "%d",imagecount); 
	    //printf("%s\n",numbuff);
	    char filebuff[100];
	    strcpy(filebuff,"viptraffic");
	    strcat(filebuff,numbuff);
	    strcat(filebuff,".ppm"); //to read all files one by one using concatenation
	    //printf("Iterator : %d\n",imagecount);
	    imagecount++;
	    
	    decode_image(image,filebuff); //call the file read function
	    
	    int i,j,k,l;
	  
	    //Separate R,G and B pixels
	    for(i = 0;i<HEIGHT * WIDTH * K;i+=3)
	    {
	      image_r[count] = image[i];
	      count++;
	      
	    }
	    count = 0;

	    for(j = 1;j<HEIGHT * WIDTH * K;j+=3)
	    {
	      image_g[count] = image[j]; 
	      count++;
	    }
	    count = 0;
	    
	    for(k = 2;k<HEIGHT * WIDTH * K;k+=3)
	    {
	      image_b[count] = image[k];
	      count++; 
	    }

	    //Create buffer for device
	    d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, HEIGHT*WIDTH*sizeof(unsigned char), NULL,&err);
	    d_image = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, HEIGHT*WIDTH*sizeof(unsigned char), image_r,&err);
	    d_filter = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, K*K*sizeof(int), filter, &err);

	     if (!d_image || !d_filter || !d_output)
	     {
		 printf("Error: Failed to allocate device memory!\n");
		 exit(1);
	     }    
		
	    err = clEnqueueWriteBuffer(commands, d_image, CL_TRUE, 0, HEIGHT*WIDTH*sizeof(unsigned char), image_r, 0, NULL, NULL);   
	    err |= clEnqueueWriteBuffer(commands, d_filter, CL_TRUE, 0, K*K*sizeof(int), filter, 0, NULL, NULL);   
	   
	   if (err != CL_SUCCESS)
	   {
	       printf("Error: Failed to write data to device! %d\n", err);
	       exit(1);
	   }

	   size_t localWorkSize[2], globalWorkSize[2];
	   int rows = HEIGHT;
	   int cols = WIDTH;
	   int filtersize = K;
	   err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_output);
	   err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_image);
	   err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_filter);
	   err |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&rows);
	   err |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&cols);
	   err |= clSetKernelArg(kernel, 5, sizeof(int), (void *)&filtersize);
	      if (err != CL_SUCCESS)
	   { 
	       printf("Error: Failed to set kernel arguments! %d\n", err);
	       exit(1);
	   }

	   localWorkSize[0] = 10;
	   localWorkSize[1] = 10;
	   globalWorkSize[0] = 120;
	   globalWorkSize[1] = 160;
	 
	   err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL,globalWorkSize, localWorkSize, 0, NULL, &myevent);   
		
		if (err != CL_SUCCESS)
	   {
	       printf("Error: Failed to execute kernel! %d\n", err);
	       exit(1);
	   }
	   clWaitForEvents(1,&myevent);	 
	   clFinish(commands);   
	   clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	   clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_END,sizeof(cl_ulong), &end, NULL);
       	   kernelExecTimeNs += end - start;
	   err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, HEIGHT*WIDTH*sizeof(unsigned char), output_r, 0, NULL, NULL);

	   if (err != CL_SUCCESS)
	   {
	       printf("Error: Failed to read output array! %d\n", err);
	       exit(1);
	   }

	   err = clEnqueueWriteBuffer(commands, d_image, CL_TRUE, 0, HEIGHT*WIDTH*sizeof(unsigned char), image_g, 0, NULL, NULL);	
	   
	   if (err != CL_SUCCESS)
	   {
	       printf("Error: Failed to write data to device! %d\n", err);
	       exit(1);
	   }
	   
	   err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_output);
	   err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_image);
	   err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_filter);
	   err |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&rows);
	   err |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&cols);
	   err |= clSetKernelArg(kernel, 5, sizeof(int), (void *)&filtersize);
	      if (err != CL_SUCCESS)
	   {
	       printf("Error: Failed to set kernel arguments! %d\n", err);
	       exit(1);
	   }

	   err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL,globalWorkSize, localWorkSize, 0, NULL, &myevent);

		if (err != CL_SUCCESS)
	   {
	       printf("Error: Failed to execute kernel! %d\n", err);
	       exit(1);
	   }
	   clWaitForEvents(1,&myevent);
	   clFinish(commands);
	   clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	   clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_END,sizeof(cl_ulong), &end, NULL);
	   kernelExecTimeNs += end - start;
	   err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, HEIGHT*WIDTH*sizeof(unsigned char), output_g, 0, NULL, NULL);

	   if (err != CL_SUCCESS)
	   {
	       printf("Error: Failed to read output array! %d\n", err);
	       exit(1);
	   }

	    err = clEnqueueWriteBuffer(commands, d_image, CL_TRUE, 0, HEIGHT*WIDTH*sizeof(unsigned char), image_b, 0, NULL, NULL);

	   if (err != CL_SUCCESS)
	   {
	       printf("Error: Failed to write data to device! %d\n", err);
	       exit(1);
	   }
	   
	   err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_output);
	   err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_image);
	   err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_filter);
	   err |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&rows);
	   err |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&cols);
	   err |= clSetKernelArg(kernel, 5, sizeof(int), (void *)&filtersize);
	      if (err != CL_SUCCESS)
	   {
	       printf("Error: Failed to set kernel arguments! %d\n", err);
	       exit(1);
	   }

	   err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL,globalWorkSize, localWorkSize, 0, NULL,&myevent);

		if (err != CL_SUCCESS)
	   {
	       printf("Error: Failed to execute kernel! %d\n", err);
	       exit(1);
	   }
	   clWaitForEvents(1,&myevent);
	   clFinish(commands);
	   clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
	   clGetEventProfilingInfo(myevent,CL_PROFILING_COMMAND_END,sizeof(cl_ulong), &end, NULL);
	   kernelExecTimeNs += end - start;	
	   err = clEnqueueReadBuffer(commands, d_output, CL_TRUE, 0, HEIGHT*WIDTH*sizeof(unsigned char), output_b, 0, NULL, NULL);

	   if (err != CL_SUCCESS)
	   {
	       printf("Error: Failed to read output array! %d\n", err);
	       exit(1);
	   }
	
	  }
	printf("Kernel Execution time: %f seconds\n",kernelExecTimeNs/1000000000); 

	//Shutdown and cleanup
	free(image_r);
	free(image_g);
	free(image_b);
	free(filter);
	free(output_r);
	free(output_g);
	free(output_b);
	 
	clReleaseMemObject(d_image);
	clReleaseMemObject(d_output);
	clReleaseMemObject(d_filter);

	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(commands);
	clReleaseContext(context);

	   return 0;
	}
