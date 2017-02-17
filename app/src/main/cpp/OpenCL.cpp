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
    // Regular OpenCL objects:
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;

    // Objects that are specific for this sample.
    bool isInputBufferInitialized;
    cl_mem inputBuffer;
    cl_mem outputBuffer;
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


    /**----------------------------------------------------------------------
     * 2. 创建上下文
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

    /**----------------------------------------------------------------------
     * 4. 创建 openCl 程序
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
     */

    err = clBuildProgram(openCLObjects.program, 0, 0, 0, 0, 0);

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
     */

    openCLObjects.kernel = clCreateKernel(openCLObjects.program, "stepKernel", &err);
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





