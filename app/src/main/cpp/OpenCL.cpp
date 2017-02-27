//
// Created by liuyi(695183065@qq.com) on 2017/2/17.
//

#include "OpenCL.h"
#include <vector>
#include <string>


//转换错误信息为字符
const char* opencl_error_to_str (cl_int error)
{
#define CASE_CL_CONSTANT(NAME) case NAME: return #NAME;

    // Suppose that no combinations are possible.
    switch(error)
    {
        CASE_CL_CONSTANT(CL_SUCCESS)
        CASE_CL_CONSTANT(CL_DEVICE_NOT_FOUND)
        CASE_CL_CONSTANT(CL_DEVICE_NOT_AVAILABLE)
        CASE_CL_CONSTANT(CL_COMPILER_NOT_AVAILABLE)
        CASE_CL_CONSTANT(CL_MEM_OBJECT_ALLOCATION_FAILURE)
        CASE_CL_CONSTANT(CL_OUT_OF_RESOURCES)
        CASE_CL_CONSTANT(CL_OUT_OF_HOST_MEMORY)
        CASE_CL_CONSTANT(CL_PROFILING_INFO_NOT_AVAILABLE)
        CASE_CL_CONSTANT(CL_MEM_COPY_OVERLAP)
        CASE_CL_CONSTANT(CL_IMAGE_FORMAT_MISMATCH)
        CASE_CL_CONSTANT(CL_IMAGE_FORMAT_NOT_SUPPORTED)
        CASE_CL_CONSTANT(CL_BUILD_PROGRAM_FAILURE)
        CASE_CL_CONSTANT(CL_MAP_FAILURE)
        CASE_CL_CONSTANT(CL_MISALIGNED_SUB_BUFFER_OFFSET)
        CASE_CL_CONSTANT(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST)
        CASE_CL_CONSTANT(CL_INVALID_VALUE)
        CASE_CL_CONSTANT(CL_INVALID_DEVICE_TYPE)
        CASE_CL_CONSTANT(CL_INVALID_PLATFORM)
        CASE_CL_CONSTANT(CL_INVALID_DEVICE)
        CASE_CL_CONSTANT(CL_INVALID_CONTEXT)
        CASE_CL_CONSTANT(CL_INVALID_QUEUE_PROPERTIES)
        CASE_CL_CONSTANT(CL_INVALID_COMMAND_QUEUE)
        CASE_CL_CONSTANT(CL_INVALID_HOST_PTR)
        CASE_CL_CONSTANT(CL_INVALID_MEM_OBJECT)
        CASE_CL_CONSTANT(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR)
        CASE_CL_CONSTANT(CL_INVALID_IMAGE_SIZE)
        CASE_CL_CONSTANT(CL_INVALID_SAMPLER)
        CASE_CL_CONSTANT(CL_INVALID_BINARY)
        CASE_CL_CONSTANT(CL_INVALID_BUILD_OPTIONS)
        CASE_CL_CONSTANT(CL_INVALID_PROGRAM)
        CASE_CL_CONSTANT(CL_INVALID_PROGRAM_EXECUTABLE)
        CASE_CL_CONSTANT(CL_INVALID_KERNEL_NAME)
        CASE_CL_CONSTANT(CL_INVALID_KERNEL_DEFINITION)
        CASE_CL_CONSTANT(CL_INVALID_KERNEL)
        CASE_CL_CONSTANT(CL_INVALID_ARG_INDEX)
        CASE_CL_CONSTANT(CL_INVALID_ARG_VALUE)
        CASE_CL_CONSTANT(CL_INVALID_ARG_SIZE)
        CASE_CL_CONSTANT(CL_INVALID_KERNEL_ARGS)
        CASE_CL_CONSTANT(CL_INVALID_WORK_DIMENSION)
        CASE_CL_CONSTANT(CL_INVALID_WORK_GROUP_SIZE)
        CASE_CL_CONSTANT(CL_INVALID_WORK_ITEM_SIZE)
        CASE_CL_CONSTANT(CL_INVALID_GLOBAL_OFFSET)
        CASE_CL_CONSTANT(CL_INVALID_EVENT_WAIT_LIST)
        CASE_CL_CONSTANT(CL_INVALID_EVENT)
        CASE_CL_CONSTANT(CL_INVALID_OPERATION)
        CASE_CL_CONSTANT(CL_INVALID_GL_OBJECT)
        CASE_CL_CONSTANT(CL_INVALID_BUFFER_SIZE)
        CASE_CL_CONSTANT(CL_INVALID_MIP_LEVEL)
        CASE_CL_CONSTANT(CL_INVALID_GLOBAL_WORK_SIZE)
        CASE_CL_CONSTANT(CL_INVALID_PROPERTY)

        default:
            return "UNKNOWN ERROR CODE";
    }

#undef CASE_CL_CONSTANT
}

#define SAMPLE_CHECK_ERRORS(ERR)                                                      \
    if(ERR != CL_SUCCESS)                                                             \
    {                                                                                 \
        LOGE                                                                          \
        (                                                                             \
            "OpenCL error with code %s happened in file %s at line %d. Exiting.\n",   \
            opencl_error_to_str(ERR), __FILE__, __LINE__                              \
        );                                                                            \
                                                                                      \
        return;                                                                       \
    }


/* Container for all OpenCL-specific objects used in the sample.
 *
 * The container consists of the following parts:
 *   - Regular OpenCL objects, used in almost each
 *     OpenCL application.
 *   - Specific OpenCL objects - buffers, used in this
 *     particular sample.
 *
 * For convenience, collect all objects in one structure.
 * Avoid global variables and make easier the process of passing
 * all arguments in functions.
 */
struct OpenCLObjects
{
    // 固定有的:
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;

    // 根据不同程序指定
    cl_program header;



    bool isInputBufferInitialized;
    cl_mem inputBuffer;
    cl_mem maskInputBuffer;
    cl_mem outputBuffer;

    cl_mem v1Buffer;
    cl_mem v2Buffer;
    cl_mem resultBuffer;
};
// Hold all OpenCL objects.
OpenCLObjects openCLObjects;


using namespace std;

/**
 * 初始化OpenCL
 * @param *env JVM 指针
 * @param openCLProgramText 内核程序
 * @param openCLObjects openCL信息封装类
 */
void initOpenCL(JNIEnv * env,  jstring openCLProgramText){
    const std::string kernelName = "convolution";

    cl_device_type device_type = CL_DEVICE_TYPE_GPU;   //使用的设备类型
    cl_int err = CL_SUCCESS;                           //报错信息

    /**----------------------------------------------------------------------
     * 1.获取可用平台信息
     */

    //获取平台ID,先取数量,在获取具体ID
    cl_uint num_of_platforms = 0;
    err = clGetPlatformIDs(0, 0, &num_of_platforms);
    SAMPLE_CHECK_ERRORS(err);

    if (num_of_platforms == 0){
        LOGE("There is no found a suitable OpenCL platform");
        return;
    }

    LOGD("Number of available platforms: %u", num_of_platforms);
    vector<cl_platform_id> platforms(num_of_platforms);
    //获取所有平台ID
    err = clGetPlatformIDs(num_of_platforms, &platforms[0], 0);
    SAMPLE_CHECK_ERRORS(err);

    //可以根据平台的名字等信息选择平台,这里选择第一个可用的平台
    LOGD("OpenCL platform names:");
    cl_uint selected_platform_index = 0;
    openCLObjects.platform = platforms[selected_platform_index];

    //获取品台名称长度
    size_t platform_name_length = 0;
    err = clGetPlatformInfo(
            platforms[selected_platform_index],
            CL_PLATFORM_NAME,
            0,
            0,
            &platform_name_length
    );
    SAMPLE_CHECK_ERRORS(err);
    //获取平台名称
    vector<char> platform_name_buffer(platform_name_length);
    err = clGetPlatformInfo(
            platforms[selected_platform_index],
            CL_PLATFORM_NAME,
            platform_name_length,
            &platform_name_buffer[0],
            0
    );
    SAMPLE_CHECK_ERRORS(err);
    string platform_name = &platform_name_buffer[0];//平台名称
    LOGD("platform name:%s", platform_name.c_str());
    //获取版本
    size_t  version_length = 0;
    err = clGetPlatformInfo(
            platforms[selected_platform_index],
            CL_PLATFORM_VERSION,
            0,
            0,
            &version_length
    );
    vector<char> platform_version_buffer(version_length);
    err = clGetPlatformInfo(
            platforms[selected_platform_index],
            CL_PLATFORM_VERSION,
            version_length,
            &platform_version_buffer[0],
            0
    );
    SAMPLE_CHECK_ERRORS(err);
    string platform_version = &platform_version_buffer[0];//平台名称
    LOGD("platform version:%s", platform_version.c_str());


    /**----------------------------------------------------------------------
     * 2. 创建上下文
     *
     *  通常，Context是指管理OpenCL对象和资源的上下文环境。为了管理OpenCL程序，下面的一些对象都要和Context关联起来：
     *  — 设备（Devices）:执行Kernel程序对象。
     *  — 程序对象（Program objects）: kernel程序源代码
     *  — Kernels : 运行在OpenCL设备上的函数。
     *  — 内存对象（Memory objects）: device处理的数据对象。
     *  — 命令队列（Command queues）: 设备之间的交互机制。
     */

    cl_context_properties context_props[] = {
            CL_CONTEXT_PLATFORM,
            cl_context_properties(openCLObjects.platform),
            0
    };
    openCLObjects.context = clCreateContextFromType(
            context_props,
            device_type,
            0,
            0,
            &err);
    SAMPLE_CHECK_ERRORS(err);


    /**----------------------------------------------------------------------
     * 3. 查找设备信息
     */

    err = clGetContextInfo(
            openCLObjects.context,
            CL_CONTEXT_DEVICES,
            sizeof(openCLObjects.device),
            &openCLObjects.device,
            0);

    SAMPLE_CHECK_ERRORS(err);

    //最大工作维度
    cl_uint maxWorkDimensions = 0;
    err = clGetDeviceInfo(
            openCLObjects.device,
            CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
            sizeof(maxWorkDimensions),
            &maxWorkDimensions,
            NULL
    );
    SAMPLE_CHECK_ERRORS(err);
    LOGD("最大工作维度:%i",maxWorkDimensions);
    //最大工作 item 数向量
    size_t size = 0;
    err = clGetDeviceInfo(
            openCLObjects.device,
            CL_DEVICE_MAX_WORK_ITEM_SIZES,
            NULL,
            NULL,
            &size
    );
    SAMPLE_CHECK_ERRORS(err);
    size_t maxWorkItemSize[size];
    err = clGetDeviceInfo(
            openCLObjects.device,
            CL_DEVICE_MAX_WORK_ITEM_SIZES,
            size,
            &maxWorkItemSize,
            NULL
    );
    SAMPLE_CHECK_ERRORS(err);
    for (int i = 0; i < maxWorkDimensions; ++i) {
        LOGD("工作维度:%i 大小:%ld",i,maxWorkItemSize[i]);
    }
    //最大工作组大小
    cl_ulong maxWorkGroupSize = 0;
    err = clGetDeviceInfo(
            openCLObjects.device,
            CL_DEVICE_MAX_WORK_GROUP_SIZE,
            sizeof(cl_ulong),
            &maxWorkGroupSize,
            NULL
    );
    SAMPLE_CHECK_ERRORS(err);
    LOGD("最大工作组大小:%ld",maxWorkGroupSize);

    //期望的向量宽度,TODO 结果为1...
    cl_uint vectorWidth = 0;
    err = clGetDeviceInfo(
            openCLObjects.device,
            CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT,
            sizeof(vectorWidth),
            &vectorWidth,
            NULL
    );
    SAMPLE_CHECK_ERRORS(err);
    LOGD("设备期望的向量宽度:%i",vectorWidth);

    //设备最大计算单元数目, 4
    cl_uint unitNum = 0;
    err = clGetDeviceInfo(
            openCLObjects.device,
            CL_DEVICE_MAX_COMPUTE_UNITS,
            sizeof(cl_uint),
            &unitNum,
            NULL
    );
    SAMPLE_CHECK_ERRORS(err);
    LOGD("设备最大计算单元数目:%i",unitNum);

    //设备的全局内存大小, 本机 3G左右
    cl_ulong globalMemorySzie = 0;
    err = clGetDeviceInfo(
            openCLObjects.device,
            CL_DEVICE_GLOBAL_MEM_SIZE,
            sizeof(cl_ulong),
            &globalMemorySzie,
            NULL
    );
    SAMPLE_CHECK_ERRORS(err);
    LOGD("设备最大计算单元数目:%lu",globalMemorySzie/1024/1024);


    /**----------------------------------------------------------------------
     * 4. 创建 openCl 程序
     * 有两种情况: 有无 header
     * 无 header TODO: 利用 Binary 对程序的加载进行优化
     * 有 header 需要先编译再连接
     */

    const char* openCLProgramTextNative = env->GetStringUTFChars(openCLProgramText, 0);
    LOGD("OpenCL program text:\n%s", openCLProgramTextNative);

    openCLObjects.program =
            clCreateProgramWithSource
                    (
                            openCLObjects.context,
                            1,
                            &openCLProgramTextNative,
                            0,
                            &err
                    );

    SAMPLE_CHECK_ERRORS(err);

    /**----------------------------------------------------------------------
     * 5. 构建 CL 程序
     * 对context中的每个设备，这个函数编译、连接源代码对象，产生device可以执行的文件，
     * 对GPU而言就是设备对应shader汇编。如果device_list参数被提供，
     * 则只对这些设备进行编译连接。options参数主要提供一些附加的编译选项，比如宏定义、优化开关标志等等。
     *
     */

    string options = "-cl-std=CL2.0";
    const  char*  op = options.c_str();

    err = clBuildProgram(openCLObjects.program, 0, 0, op, 0, 0);

    if(err == CL_BUILD_PROGRAM_FAILURE)
    {
        size_t log_length = 0;
        err = clGetProgramBuildInfo(
                openCLObjects.program,
                openCLObjects.device,
                CL_PROGRAM_BUILD_LOG,
                0,
                0,
                &log_length
        );
        SAMPLE_CHECK_ERRORS(err);

        vector<char> log(log_length);

        err = clGetProgramBuildInfo(
                openCLObjects.program,
                openCLObjects.device,
                CL_PROGRAM_BUILD_LOG,
                log_length,
                &log[0],
                0
        );
        SAMPLE_CHECK_ERRORS(err);

        LOGE("Error happened during the build of OpenCL program.\nBuild log:%s", &log[0]);
        return;
    }




    /**-----------------------------------------------------------------------
     * 6. 提取内核
     * Kernel就是在程序代码中的一个函数，这个函数能在OpenCL设备上执行。
     * 一个Kernel对象就是kernel函数以及其相关的输入参数。
     */

    openCLObjects.kernel = clCreateKernel(openCLObjects.program, kernelName.c_str(), &err);
    SAMPLE_CHECK_ERRORS(err);

    /* -----------------------------------------------------------------------
     * 7. 创建指令队列
     */

    openCLObjects.queue =
            clCreateCommandQueue
                    (
                            openCLObjects.context,
                            openCLObjects.device,
                            0,    // Creating queue properties, refer to the OpenCL specification for details.
                            &err
                    );
    SAMPLE_CHECK_ERRORS(err);

    // -----------------------------------------------------------------------

    LOGD("initOpenCL finished successfully");

    env->ReleaseStringUTFChars(openCLProgramText, openCLProgramTextNative);//释放 code String
}

void shutdownOpenCL(){
    cl_int err = CL_SUCCESS;

    if(openCLObjects.isInputBufferInitialized)
    {
        err = clReleaseMemObject(openCLObjects.inputBuffer);
        SAMPLE_CHECK_ERRORS(err);
    }

    err = clReleaseKernel(openCLObjects.kernel);
    SAMPLE_CHECK_ERRORS(err);

    err = clReleaseProgram(openCLObjects.program);
    SAMPLE_CHECK_ERRORS(err);

    err = clReleaseCommandQueue(openCLObjects.queue);
    SAMPLE_CHECK_ERRORS(err);

    err = clReleaseContext(openCLObjects.context);
    SAMPLE_CHECK_ERRORS(err);

}

void vectorAdd(int* v1, int* v2, int* result, unsigned int size_){
    void* v1Buffer = v1;
    void* v2Buffer = v2;

    size_t size = size_* sizeof(int);
    cl_int err = CL_SUCCESS;

    openCLObjects.v1Buffer =
            clCreateBuffer(
                    openCLObjects.context,
                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    size,   // Buffer size in bytes.
                    v1Buffer,  // Bytes for initialization.
                    &err
            );
    SAMPLE_CHECK_ERRORS(err);
    openCLObjects.v2Buffer =
            clCreateBuffer(
                    openCLObjects.context,
                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    size,   // Buffer size in bytes.
                    v2Buffer,  // Bytes for initialization.
                    &err
            );
    SAMPLE_CHECK_ERRORS(err);

    cl_mem outputBuffer =
            clCreateBuffer
                    (
                            openCLObjects.context,
                            CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
                            size,    // Buffer size in bytes, same as the input buffer.
                            result,  // Area, above which the buffer is created.
                            &err
                    );
    SAMPLE_CHECK_ERRORS(err);

    //核函数参数设置
    //设置 srcBuffer
    err = clSetKernelArg(openCLObjects.kernel, 0, sizeof(openCLObjects.v1Buffer), &openCLObjects.v1Buffer);
    SAMPLE_CHECK_ERRORS(err);

    //设置 maskBuffer
    err = clSetKernelArg(openCLObjects.kernel, 1, sizeof(openCLObjects.v2Buffer), &openCLObjects.v2Buffer);
    SAMPLE_CHECK_ERRORS(err);

    //设置 maskSize
    err = clSetKernelArg(openCLObjects.kernel, 2, sizeof(openCLObjects.resultBuffer), &outputBuffer);
    SAMPLE_CHECK_ERRORS(err);

    size_t globalSize[1] = { size };

    err = clEnqueueNDRangeKernel
            (
                    openCLObjects.queue,
                    openCLObjects.kernel,
                    1,
                    0,
                    globalSize,
                    0,
                    0, 0, 0
            );

    SAMPLE_CHECK_ERRORS(err);

    err = clFinish(openCLObjects.queue);
    SAMPLE_CHECK_ERRORS(err);

    err = clEnqueueReadBuffer(
            openCLObjects.queue,
            outputBuffer,
            CL_TRUE,
            0,
            size,
            result,
            0, NULL, NULL);
    SAMPLE_CHECK_ERRORS(err);

//    clEnqueueMapBuffer
//            (
//                    openCLObjects.queue,
//                    outputBuffer,
//                    CL_TRUE,    // to use the host pointer in the next call
//                    CL_MAP_READ,
//                    0,
//                    size,
//                    0, 0, 0,
//                    &err
//            );
//    SAMPLE_CHECK_ERRORS(err);
//
//    err = clEnqueueUnmapMemObject
//            (
//                    openCLObjects.queue,
//                    outputBuffer,
//                    outputMatBuffer,
//                    0, 0, 0
//            );
//    SAMPLE_CHECK_ERRORS(err);

    // Call clFinish to guarantee that the output region is updated.
    err = clFinish(openCLObjects.queue);
    SAMPLE_CHECK_ERRORS(err);

    err = clReleaseMemObject(outputBuffer);
    SAMPLE_CHECK_ERRORS(err);


}

void dot(float* colMat, const int colMatSize,
         float* maskMat, const int maskMatSize,
         float* dstMat){
    void* inputBuffer = colMat;
    void* maskBuffer = maskMat;
    size_t inputBufferSize =  colMatSize * sizeof(float);
    size_t maskBufferSize = maskMatSize * sizeof(float);
    size_t outputBufferSize = maskMatSize* sizeof (float);

    cl_int err = CL_SUCCESS;

    openCLObjects.inputBuffer =
            clCreateBuffer(
                    openCLObjects.context,
                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    inputBufferSize,   // Buffer size in bytes.
                    inputBuffer,  // Bytes for initialization.
                    &err
            );
    SAMPLE_CHECK_ERRORS(err);

    openCLObjects.maskInputBuffer =
            clCreateBuffer(
                    openCLObjects.context,
                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    maskBufferSize,
                    maskBuffer,
                    &err
            );
    SAMPLE_CHECK_ERRORS(err);

    openCLObjects.isInputBufferInitialized = true;

    cl_mem outputBuffer =
            clCreateBuffer
                    (
                            openCLObjects.context,
                            CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
                            outputBufferSize,    // Buffer size in bytes, same as the input buffer.
                            dstMat,  // Area, above which the buffer is created.
                            &err
                    );
    SAMPLE_CHECK_ERRORS(err);



    //核函数参数设置
    //设置 srcBuffer
    err = clSetKernelArg(openCLObjects.kernel, 0, sizeof(openCLObjects.inputBuffer), &openCLObjects.inputBuffer);
    SAMPLE_CHECK_ERRORS(err);

    //设置 maskBuffer
    err = clSetKernelArg(openCLObjects.kernel, 1, sizeof(openCLObjects.maskInputBuffer), &openCLObjects.maskInputBuffer);
    SAMPLE_CHECK_ERRORS(err);

    err = clSetKernelArg(openCLObjects.kernel, 2, sizeof(outputBuffer), &outputBuffer);
    SAMPLE_CHECK_ERRORS(err);

    //工作空间大小为
    size_t globalSize[3] = {
            9 * 3 * sizeof(unsigned int),
            100 * sizeof(unsigned int),
            10 * 3 * sizeof(unsigned int),
    };

    size_t localSize[3] = {
            3,
            1,
            1,
    };

    timeval start;
    timeval end;

    gettimeofday(&start, NULL);

    err = clEnqueueNDRangeKernel
            (
                    openCLObjects.queue,
                    openCLObjects.kernel,
                    3,
                    NULL,
                    globalSize,
                    localSize,
                    0, NULL, NULL
            );
    SAMPLE_CHECK_ERRORS(err);

    err = clFinish(openCLObjects.queue);
    SAMPLE_CHECK_ERRORS(err);
    gettimeofday(&end, NULL);
    float ndrangeDuration =
            (end.tv_sec + end.tv_usec * 1e-6) - (start.tv_sec + start.tv_usec * 1e-6);

    LOGD("NDRangeKernel time: %f", ndrangeDuration);


    clEnqueueMapBuffer
            (
                    openCLObjects.queue,
                    outputBuffer,
                    CL_TRUE,    // to use the host pointer in the next call
                    CL_MAP_READ,
                    0,
                    outputBufferSize,
                    0, 0, 0,
                    &err
            );
    SAMPLE_CHECK_ERRORS(err);

    err = clEnqueueUnmapMemObject
            (
                    openCLObjects.queue,
                    outputBuffer,
                    dstMat,
                    0, 0, 0
            );
    SAMPLE_CHECK_ERRORS(err);

    // Call clFinish to guarantee that the output region is updated.
    err = clFinish(openCLObjects.queue);
    SAMPLE_CHECK_ERRORS(err);

    err = clReleaseMemObject(outputBuffer);
    SAMPLE_CHECK_ERRORS(err);

    LOGD("nativeConvOpenCL ends successfully");
}

void testOpenCLConv(float * srcColMat, unsigned int numSrcMat, unsigned int srcH, unsigned int srcW,
                    float * maskMat, unsigned int numMask, unsigned int maskSize,
                    float * dstMat, unsigned int dstH, unsigned int dstW){

    void* inputBuffer = srcColMat;
    void* maskBuffer = maskMat;

    size_t inputBufferSize = numSrcMat * srcH * srcW * sizeof (float);
    size_t maskBufferSize = numMask * maskSize * sizeof (float);
    size_t outputBufferSize = numSrcMat * numSrcMat * dstH * dstW * sizeof (float);

    cl_int err = CL_SUCCESS;

    openCLObjects.inputBuffer =
            clCreateBuffer(
                    openCLObjects.context,
                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    inputBufferSize,   // Buffer size in bytes.
                    inputBuffer,  // Bytes for initialization.
                    &err
            );
    SAMPLE_CHECK_ERRORS(err);

    openCLObjects.maskInputBuffer =
            clCreateBuffer(
                    openCLObjects.context,
                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    maskBufferSize,
                    maskBuffer,
                    &err
            );
    SAMPLE_CHECK_ERRORS(err);

    openCLObjects.isInputBufferInitialized = true;

    cl_mem outputBuffer =
            clCreateBuffer
                    (
                            openCLObjects.context,
                            CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
                            outputBufferSize,    // Buffer size in bytes, same as the input buffer.
                            dstMat,  // Area, above which the buffer is created.
                            &err
                    );
    SAMPLE_CHECK_ERRORS(err);

    //核函数参数设置
    //设置 srcBuffer
    err = clSetKernelArg(openCLObjects.kernel, 0, sizeof(openCLObjects.inputBuffer), &openCLObjects.inputBuffer);
    SAMPLE_CHECK_ERRORS(err);

    //设置 long
    err = clSetKernelArg(openCLObjects.kernel, 1, sizeof(unsigned int), &srcW);
    SAMPLE_CHECK_ERRORS(err);

    //设置 maskBuffer
    err = clSetKernelArg(openCLObjects.kernel, 2, sizeof(openCLObjects.maskInputBuffer), &openCLObjects.maskInputBuffer);
    SAMPLE_CHECK_ERRORS(err);

//    //设置 maskNum
//    err = clSetKernelArg(openCLObjects.kernel, 3, sizeof(unsigned int), &numMaskMat);
//    SAMPLE_CHECK_ERRORS(err);

    //设置 maskSize
    err = clSetKernelArg(openCLObjects.kernel, 3, sizeof(unsigned int), &maskSize);
    SAMPLE_CHECK_ERRORS(err);

    //设置 dstBuffer
    err = clSetKernelArg(openCLObjects.kernel, 4, sizeof(outputBuffer), &outputBuffer);
    SAMPLE_CHECK_ERRORS(err);

    //最大的工作空间大小为 3
    size_t globalSize[3] = {
            maskSize * sizeof(unsigned int), //行数应该由 dstSize 确定这里一样
            numSrcMat * sizeof(unsigned int),
            numMask * sizeof(unsigned int)};

    timeval start;
    timeval end;

    gettimeofday(&start, NULL);

    err = clEnqueueNDRangeKernel
            (
                    openCLObjects.queue,
                    openCLObjects.kernel,
                    3,
                    0,
                    globalSize,
                    0,
                    0, 0, 0
            );

    SAMPLE_CHECK_ERRORS(err);

    err = clFinish(openCLObjects.queue);
    SAMPLE_CHECK_ERRORS(err);
    gettimeofday(&end, NULL);

    float ndrangeDuration =
            (end.tv_sec + end.tv_usec * 1e-6) - (start.tv_sec + start.tv_usec * 1e-6);

    LOGD("NDRangeKernel time: %f", ndrangeDuration);


    clEnqueueMapBuffer
            (
                    openCLObjects.queue,
                    outputBuffer,
                    CL_TRUE,    // to use the host pointer in the next call
                    CL_MAP_READ,
                    0,
                    outputBufferSize,
                    0, 0, 0,
                    &err
            );
    SAMPLE_CHECK_ERRORS(err);

    err = clEnqueueUnmapMemObject
            (
                    openCLObjects.queue,
                    outputBuffer,
                    dstMat,
                    0, 0, 0
            );
    SAMPLE_CHECK_ERRORS(err);

    // Call clFinish to guarantee that the output region is updated.
    err = clFinish(openCLObjects.queue);
    SAMPLE_CHECK_ERRORS(err);

    err = clReleaseMemObject(outputBuffer);
    SAMPLE_CHECK_ERRORS(err);

    LOGD("nativeConvOpenCL ends successfully");
}

/**
 * flags参数指定buffer对象的读写属性，host_ptr可以是NULL，如果不为NULL，一般是一个有效的host buffer对象，
 * 这时，函数创建OpenCL buffer对象后，会把对应host buffer的内容拷贝到OpenCL buffer中。
 */
void creatBuffer(cl_context context,cl_mem_flags flags, size_t size,  void * host_prt,  cl_int * err){
    cl_mem buffer = clCreateBuffer(
            context,
            flags,
            size,
            host_prt,
            err
    );
}

/**
 * 在Kernel执行之前，host中原始输入数据必须显式的传到device中，Kernel执行完后，结果也要从device内存中传回到host内存中。
 * 我们主要通过函数clEnqueue{Read|Write}{Buffer|Image}来实现这两种操作。
 * 从host到device，我们用clEnqueueWrite，从device到host，我们用clEnqueueRead。
 * clEnqueueWrite命令包括初始化内存对象以及把host 数据传到device内存这两种操作。
 * 当然，也可以把host buffer指针直接用在CreateBuffer函数中来实现隐式的数据写操作。
 */
void writeBuffer(){
//    clEnqueueWriteBuffer()
}




