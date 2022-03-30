#include <OpenCL/cl.h>
#include <cstdio>
#include <string>
#include <time.h>
#include <sys/time.h>

struct Runtime {
    cl_device_id device;
    cl_context context;
    cl_program program;
    cl_kernel kernel;
    cl_command_queue queue;
    cl_mem input_buffer;
    cl_mem bias_buffer;
    cl_mem output_buffer;

    ~Runtime() {
        clReleaseKernel(kernel);
        clReleaseMemObject(input_buffer);
        clReleaseMemObject(output_buffer);
        clReleaseCommandQueue(queue);
        clReleaseProgram(program);
        clReleaseContext(context);
    }
};

void check_status(cl_int status, const std::string &message) {
    if (status < 0) {
        printf("%s", message.c_str());
        exit(1);
    };
}

constexpr size_t ARRAY_SIZE = 100000;
constexpr char const *PROGRAM_FILE = "add.cl";
constexpr char const *KERNEL_FUNC = "add";

cl_device_id init_device() {

    cl_platform_id platform;
    cl_device_id dev_id;
    int status;

    // try to get a platform
    status = clGetPlatformIDs(1, &platform, nullptr);
    if (status < 0) {
        printf("could not get a platform");
        exit(1);
    }

    // try to get gpu device
    // GPU
    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev_id, nullptr);
    if (status == CL_DEVICE_NOT_FOUND) {
        // CPU
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev_id, nullptr);
    }
    if (status < 0) {
        printf("no devices found");
        exit(1);
    }
    return dev_id;
}

/// create program from file
cl_program build_program(cl_context ctx, cl_device_id device_id, const char *filename) {

    cl_program program;
    FILE *program_handle;
    char *program_buffer, *program_log;
    size_t program_size, log_size;
    int status;

    // read kernel source int buffer
    program_handle = fopen(filename, "r");
    check_status(status, "find opencl kernel failed");

    fseek(program_handle, 0, SEEK_END);
    program_size = ftell(program_handle);
    rewind(program_handle);
    program_buffer = (char *) malloc(program_size + 1);
    program_buffer[program_size] = '\0';
    fread(program_buffer, sizeof(char), program_size, program_handle);
    fclose(program_handle);

    program = clCreateProgramWithSource(ctx, 1,
                                        (const char **) &program_buffer, &program_size, &status);
    check_status(status, "create program failed");
    free(program_buffer);

    // build opencl program
    status = clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr);
    if (status < 0) {

        // pull opencl build log
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
                              0, nullptr, &log_size);
        program_log = (char *) malloc(log_size + 1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
                              log_size + 1, program_log, nullptr);
        printf("%s\n", program_log);
        free(program_log);
        exit(1);
    }

    return program;
}

int main(int argc, char **argv) {
    struct timeval start, end;
    // get the start time
    gettimeofday(&start, NULL);
    Runtime runtime{};

    cl_int status;
    // init device
    runtime.device = init_device();

    // create context
    runtime.context = clCreateContext(nullptr, 1, &runtime.device, nullptr, nullptr, &status);
    check_status(status, "create context failed");

    runtime.program = build_program(runtime.context, runtime.device, PROGRAM_FILE);

    // create queue
    runtime.queue = clCreateCommandQueue(runtime.context, runtime.device, CL_QUEUE_PROFILING_ENABLE, &status);
    check_status(status, "create command queue failed");

    // create kernel
    runtime.kernel = clCreateKernel(runtime.program, KERNEL_FUNC, &status);
    check_status(status, "create kernel failed");


    // init input data
    float input_data[ARRAY_SIZE];
    float bias_data[ARRAY_SIZE];
    float output_data[ARRAY_SIZE];

    for (int i = 0; i < ARRAY_SIZE; i++) {
        input_data[i] = 1.f * (float) i;
        bias_data[i] = 10000.f;
    }

    // create buffers
    runtime.input_buffer = clCreateBuffer(runtime.context, CL_MEM_READ_ONLY |
        CL_MEM_COPY_HOST_PTR, ARRAY_SIZE * sizeof(float), input_data, &status);
    runtime.bias_buffer = clCreateBuffer(runtime.context, CL_MEM_READ_ONLY |
        CL_MEM_COPY_HOST_PTR, ARRAY_SIZE * sizeof(float), bias_data, &status);
    runtime.output_buffer = clCreateBuffer(runtime.context, CL_MEM_READ_ONLY |
        CL_MEM_COPY_HOST_PTR, ARRAY_SIZE * sizeof(float), output_data, &status);

    check_status(status, "create buffer failed");


    // config cl args
    status = clSetKernelArg(runtime.kernel, 0, sizeof(cl_mem), &runtime.input_buffer);
    status |= clSetKernelArg(runtime.kernel, 1, sizeof(cl_mem), &runtime.bias_buffer);
    status |= clSetKernelArg(runtime.kernel, 2, sizeof(cl_mem), &runtime.output_buffer);

    check_status(status, "create kernel failed");
    // cl gpu time profile
    cl_event timing_event;
    cl_ulong t_queued, t_submit, t_start, t_end;

    // clEnqueueNDRangeKernel
    status = clEnqueueNDRangeKernel(runtime.queue, runtime.kernel, 1, nullptr, &ARRAY_SIZE,
                                    nullptr, 0, nullptr, &timing_event);
    clWaitForEvents(1, &timing_event);

    check_status(status, "clEnqueueNDRangeKernel failed");

    clGetEventProfilingInfo(timing_event, CL_PROFILING_COMMAND_QUEUED,
                            sizeof(cl_ulong), &t_queued, nullptr);
    clGetEventProfilingInfo(timing_event, CL_PROFILING_COMMAND_SUBMIT,
                            sizeof(cl_ulong), &t_submit, nullptr);
    clGetEventProfilingInfo(timing_event, CL_PROFILING_COMMAND_START,
                            sizeof(cl_ulong), &t_start, nullptr);
    clGetEventProfilingInfo(timing_event, CL_PROFILING_COMMAND_END,
                            sizeof(cl_ulong), &t_end, nullptr);

    printf("t_queued at %llu  \n"
           "t_start at %llu  \n"
           "t_submit at %llu  \n"
           "t_end at %llu  \n"
           "kernel execute cost %f ns \n"
           "", t_queued, t_start, t_submit, t_end, (t_end - t_start) * 1e-0);

    // Read the kernel's output
    status = clEnqueueReadBuffer(runtime.queue, runtime.output_buffer, CL_TRUE, 0,
                                 sizeof(output_data), output_data, 0, nullptr, nullptr);
    check_status(status, "clEnqueueReadBuffer failed");

    // finish opencl
    status = clFinish(runtime.queue);

    // get the end time
    gettimeofday(&end, NULL);
    // Print the total execution time
    double elapsed_time = (end.tv_sec - start.tv_sec) * 1000. + \
                (end.tv_usec - start.tv_usec) / 1000.;
    printf("cpu all cost %f ms \n", elapsed_time);

    for (int i = 0; i < ARRAY_SIZE; ++i) {
        float output = bias_data[i] + input_data[i];
        if (abs(output - output_data[i]) > 1e5) {
            printf("%d %f vs %f \n", i, output, output_data[i]);
            check_status(-1, "CHECK RESULT FAILED");
        }
    }

    printf("ALL PASSED");
    return 0;
}

