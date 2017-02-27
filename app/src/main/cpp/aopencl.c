/*******************************************************************************
 * 以前的方式是拉取 OpenCl 库,在编译时连接,现在直接在系统中寻找调用。
 * 这是一种更为合理的调用方式,但是现在处于探索阶段,以后使用。
 * TODO: 动态调用 OpenCL 共享库。
 ******************************************************************************/
#include <CL/cl.h>
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

#include "aopencl.h"
#include "android/log.h"
#define LOG_TAG "COP_Native"
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))

static int loadedCL;
#define LIB_OPENCL "/system/vendor/lib/libOpenCL.so"
#define LIB64_OPENCL "/system/vendor/lib64/libOpenCL.so"

static void *getCLHandle(){
    void *res = NULL;
    char* so_name="Unknown Shared library for OpenCL";
    res = dlopen(LIB64_OPENCL,RTLD_LAZY);
    if(res != NULL){
        so_name = LIB64_OPENCL;
        LOGD("Loaded library name:%s\n",so_name);
    } else {
        res = dlopen(LIB_OPENCL,RTLD_LAZY);
    }

    if (res != NULL){
        so_name = LIB_OPENCL;
        LOGD("Loaded library name:%s\n",so_name);
    } else{
        printf("Could not open library :(\n");
    }

    return res;
}

void initFns(){
  loadedCL = 0;
  void *handle = getCLHandle();
  if(handle==NULL){
      LOGD("Loaded library filed");
      return;
  }

/* Platform API */
 IAH();
 aclGetPlatformIDs = (cl_int  (*)(cl_uint          /* num_entries */,
                 cl_platform_id * /* platforms */,
                 cl_uint *        /* num_platforms */)) dlsym(handle,"clGetPlatformIDs");

 IAH();
 aclGetPlatformInfo = (cl_int   (*)(cl_platform_id   /* platform */, 
                  cl_platform_info /* param_name */,
                  size_t           /* param_value_size */, 
                  void *           /* param_value */,
                  size_t *         /* param_value_size_ret */)) dlsym(handle,"clGetPlatformInfo");

/* Device APIs */
 IAH();
 aclGetDeviceIDs = (cl_int  (*)(cl_platform_id   /* platform */,
               cl_device_type   /* device_type */, 
               cl_uint          /* num_entries */, 
               cl_device_id *   /* devices */, 
               cl_uint *        /* num_devices */)) dlsym(handle,"clGetDeviceIDs");

 IAH();
 aclGetDeviceInfo = (cl_int  (*)(cl_device_id    /* device */,
                cl_device_info  /* param_name */, 
                size_t          /* param_value_size */, 
                void *          /* param_value */,
                size_t *        /* param_value_size_ret */)) dlsym(handle,"clGetDeviceInfo");
    

    
    
/* Context APIs  */
 IAH();
 aclCreateContext = (cl_context  (*)(const cl_context_properties * /* properties */,
                cl_uint                 /* num_devices */,
                const cl_device_id *    /* devices */,
                void (CL_CALLBACK * /* pfn_notify */)(const char *, const void *, size_t, void *),
                void *                  /* user_data */,
                cl_int *                /* errcode_ret */)) dlsym(handle,"clCreateContext");

 IAH();
 aclCreateContextFromType = (cl_context  (*)(const cl_context_properties * /* properties */,
                        cl_device_type          /* device_type */,
                        void (CL_CALLBACK *     /* pfn_notify*/ )(const char *, const void *, size_t, void *),
                        void *                  /* user_data */,
                        cl_int *                /* errcode_ret */)) dlsym(handle,"clCreateContextFromType");

 IAH();
 aclRetainContext = (cl_int  (*)(cl_context /* context */)) dlsym(handle,"clRetainContext");

 IAH();
 aclReleaseContext = (cl_int  (*)(cl_context /* context */)) dlsym(handle,"clReleaseContext");

 IAH();
 aclGetContextInfo = (cl_int  (*)(cl_context         /* context */, 
                 cl_context_info    /* param_name */, 
                 size_t             /* param_value_size */, 
                 void *             /* param_value */, 
                 size_t *           /* param_value_size_ret */)) dlsym(handle,"clGetContextInfo");

/* Command Queue APIs */
 IAH();
 aclCreateCommandQueue = (cl_command_queue  (*)(cl_context                     /* context */, 
                     cl_device_id                   /* device */, 
                     cl_command_queue_properties    /* properties */,
                     cl_int *                       /* errcode_ret */)) dlsym(handle,"clCreateCommandQueue");

 IAH();
 aclRetainCommandQueue = (cl_int  (*)(cl_command_queue /* command_queue */)) dlsym(handle,"clRetainCommandQueue");

 IAH();
 aclReleaseCommandQueue = (cl_int  (*)(cl_command_queue /* command_queue */)) dlsym(handle,"clReleaseCommandQueue");

 IAH();
 aclGetCommandQueueInfo = (cl_int  (*)(cl_command_queue      /* command_queue */,
                      cl_command_queue_info /* param_name */,
                      size_t                /* param_value_size */,
                      void *                /* param_value */,
                      size_t *              /* param_value_size_ret */)) dlsym(handle,"clGetCommandQueueInfo");

/* Memory Object APIs */
 IAH();
 aclCreateBuffer = (cl_mem  (*)(cl_context   /* context */,
               cl_mem_flags /* flags */,
               size_t       /* size */,
               void *       /* host_ptr */,
               cl_int *     /* errcode_ret */)) dlsym(handle,"clCreateBuffer");

 IAH();
 aclCreateSubBuffer = (cl_mem  (*)(cl_mem                   /* buffer */,
                  cl_mem_flags             /* flags */,
                  cl_buffer_create_type    /* buffer_create_type */,
                  const void *             /* buffer_create_info */,
                  cl_int *                 /* errcode_ret */)) dlsym(handle,"clCreateSubBuffer");

                       
 IAH();
 aclRetainMemObject = (cl_int  (*)(cl_mem /* memobj */)) dlsym(handle,"clRetainMemObject");

 IAH();
 aclReleaseMemObject = (cl_int  (*)(cl_mem /* memobj */)) dlsym(handle,"clReleaseMemObject");

 IAH();
 aclGetSupportedImageFormats = (cl_int  (*)(cl_context           /* context */,
                           cl_mem_flags         /* flags */,
                           cl_mem_object_type   /* image_type */,
                           cl_uint              /* num_entries */,
                           cl_image_format *    /* image_formats */,
                           cl_uint *            /* num_image_formats */)) dlsym(handle,"clGetSupportedImageFormats");
                                    
 IAH();
 aclGetMemObjectInfo = (cl_int  (*)(cl_mem           /* memobj */,
                   cl_mem_info      /* param_name */, 
                   size_t           /* param_value_size */,
                   void *           /* param_value */,
                   size_t *         /* param_value_size_ret */)) dlsym(handle,"clGetMemObjectInfo");

 IAH();
 aclGetImageInfo = (cl_int  (*)(cl_mem           /* image */,
               cl_image_info    /* param_name */, 
               size_t           /* param_value_size */,
               void *           /* param_value */,
               size_t *         /* param_value_size_ret */)) dlsym(handle,"clGetImageInfo");

 IAH();
 aclSetMemObjectDestructorCallback = (cl_int  (*)(  cl_mem /* memobj */, 
                                    void (CL_CALLBACK * /*pfn_notify*/)( cl_mem /* memobj */, void* /*user_data*/), 
                                    void * /*user_data */ )) dlsym(handle,"clSetMemObjectDestructorCallback");  

/* Sampler APIs */
 IAH();
 aclCreateSampler = (cl_sampler  (*)(cl_context          /* context */,
                cl_bool             /* normalized_coords */, 
                cl_addressing_mode  /* addressing_mode */, 
                cl_filter_mode      /* filter_mode */,
                cl_int *            /* errcode_ret */)) dlsym(handle,"clCreateSampler");

 IAH();
 aclRetainSampler = (cl_int  (*)(cl_sampler /* sampler */)) dlsym(handle,"clRetainSampler");

 IAH();
 aclReleaseSampler = (cl_int  (*)(cl_sampler /* sampler */)) dlsym(handle,"clReleaseSampler");

 IAH();
 aclGetSamplerInfo = (cl_int  (*)(cl_sampler         /* sampler */,
                 cl_sampler_info    /* param_name */,
                 size_t             /* param_value_size */,
                 void *             /* param_value */,
                 size_t *           /* param_value_size_ret */)) dlsym(handle,"clGetSamplerInfo");
                            
/* Program Object APIs  */
 IAH();
 aclCreateProgramWithSource = (cl_program  (*)(cl_context        /* context */,
                          cl_uint           /* count */,
                          const char **     /* strings */,
                          const size_t *    /* lengths */,
                          cl_int *          /* errcode_ret */)) dlsym(handle,"clCreateProgramWithSource");

 IAH();
 aclCreateProgramWithBinary = (cl_program  (*)(cl_context                     /* context */,
                          cl_uint                        /* num_devices */,
                          const cl_device_id *           /* device_list */,
                          const size_t *                 /* lengths */,
                          const unsigned char **         /* binaries */,
                          cl_int *                       /* binary_status */,
                          cl_int *                       /* errcode_ret */)) dlsym(handle,"clCreateProgramWithBinary");


 IAH();
 aclRetainProgram = (cl_int  (*)(cl_program /* program */)) dlsym(handle,"clRetainProgram");

 IAH();
 aclReleaseProgram = (cl_int  (*)(cl_program /* program */)) dlsym(handle,"clReleaseProgram");

 IAH();
 aclBuildProgram = (cl_int  (*)(cl_program           /* program */,
               cl_uint              /* num_devices */,
               const cl_device_id * /* device_list */,
               const char *         /* options */, 
               void (CL_CALLBACK *  /* pfn_notify */)(cl_program /* program */, void * /* user_data */),
               void *               /* user_data */)) dlsym(handle,"clBuildProgram");





 IAH();
 aclGetProgramInfo = (cl_int  (*)(cl_program         /* program */,
                 cl_program_info    /* param_name */,
                 size_t             /* param_value_size */,
                 void *             /* param_value */,
                 size_t *           /* param_value_size_ret */)) dlsym(handle,"clGetProgramInfo");

 IAH();
 aclGetProgramBuildInfo = (cl_int  (*)(cl_program            /* program */,
                      cl_device_id          /* device */,
                      cl_program_build_info /* param_name */,
                      size_t                /* param_value_size */,
                      void *                /* param_value */,
                      size_t *              /* param_value_size_ret */)) dlsym(handle,"clGetProgramBuildInfo");
                            
/* Kernel Object APIs */
 IAH();
 aclCreateKernel = (cl_kernel  (*)(cl_program      /* program */,
               const char *    /* kernel_name */,
               cl_int *        /* errcode_ret */)) dlsym(handle,"clCreateKernel");

 IAH();
 aclCreateKernelsInProgram = (cl_int  (*)(cl_program     /* program */,
                         cl_uint        /* num_kernels */,
                         cl_kernel *    /* kernels */,
                         cl_uint *      /* num_kernels_ret */)) dlsym(handle,"clCreateKernelsInProgram");

 IAH();
 aclRetainKernel = (cl_int  (*)(cl_kernel    /* kernel */)) dlsym(handle,"clRetainKernel");

 IAH();
 aclReleaseKernel = (cl_int  (*)(cl_kernel   /* kernel */)) dlsym(handle,"clReleaseKernel");

 IAH();
 aclSetKernelArg = (cl_int  (*)(cl_kernel    /* kernel */,
               cl_uint      /* arg_index */,
               size_t       /* arg_size */,
               const void * /* arg_value */)) dlsym(handle,"clSetKernelArg");

 IAH();
 aclGetKernelInfo = (cl_int  (*)(cl_kernel       /* kernel */,
                cl_kernel_info  /* param_name */,
                size_t          /* param_value_size */,
                void *          /* param_value */,
                size_t *        /* param_value_size_ret */)) dlsym(handle,"clGetKernelInfo");


 IAH();
 aclGetKernelWorkGroupInfo = (cl_int  (*)(cl_kernel                  /* kernel */,
                         cl_device_id               /* device */,
                         cl_kernel_work_group_info  /* param_name */,
                         size_t                     /* param_value_size */,
                         void *                     /* param_value */,
                         size_t *                   /* param_value_size_ret */)) dlsym(handle,"clGetKernelWorkGroupInfo");

/* Event Object APIs */
 IAH();
 aclWaitForEvents = (cl_int  (*)(cl_uint             /* num_events */,
                const cl_event *    /* event_list */)) dlsym(handle,"clWaitForEvents");

 IAH();
 aclGetEventInfo = (cl_int  (*)(cl_event         /* event */,
               cl_event_info    /* param_name */,
               size_t           /* param_value_size */,
               void *           /* param_value */,
               size_t *         /* param_value_size_ret */)) dlsym(handle,"clGetEventInfo");
                            
 IAH();
 aclCreateUserEvent = (cl_event  (*)(cl_context    /* context */,
                  cl_int *      /* errcode_ret */)) dlsym(handle,"clCreateUserEvent");               
                            
 IAH();
 aclRetainEvent = (cl_int  (*)(cl_event /* event */)) dlsym(handle,"clRetainEvent");

 IAH();
 aclReleaseEvent = (cl_int  (*)(cl_event /* event */)) dlsym(handle,"clReleaseEvent");

 IAH();
 aclSetUserEventStatus = (cl_int  (*)(cl_event   /* event */,
                     cl_int     /* execution_status */)) dlsym(handle,"clSetUserEventStatus");
                     
 IAH();
 aclSetEventCallback = (cl_int  (*)( cl_event    /* event */,
                    cl_int      /* command_exec_callback_type */,
                    void (CL_CALLBACK * /* pfn_notify */)(cl_event, cl_int, void *),
                    void *      /* user_data */)) dlsym(handle,"clSetEventCallback");

/* Profiling APIs */
 IAH();
 aclGetEventProfilingInfo = (cl_int  (*)(cl_event            /* event */,
                        cl_profiling_info   /* param_name */,
                        size_t              /* param_value_size */,
                        void *              /* param_value */,
                        size_t *            /* param_value_size_ret */)) dlsym(handle,"clGetEventProfilingInfo");
                                
/* Flush and Finish APIs */
 IAH();
 aclFlush = (cl_int  (*)(cl_command_queue /* command_queue */)) dlsym(handle,"clFlush");

 IAH();
 aclFinish = (cl_int  (*)(cl_command_queue /* command_queue */)) dlsym(handle,"clFinish");

/* Enqueued Commands APIs */
 IAH();
 aclEnqueueReadBuffer = (cl_int  (*)(cl_command_queue    /* command_queue */,
                    cl_mem              /* buffer */,
                    cl_bool             /* blocking_read */,
                    size_t              /* offset */,
                    size_t              /* size */, 
                    void *              /* ptr */,
                    cl_uint             /* num_events_in_wait_list */,
                    const cl_event *    /* event_wait_list */,
                    cl_event *          /* event */)) dlsym(handle,"clEnqueueReadBuffer");
                            
 IAH();
 aclEnqueueReadBufferRect = (cl_int  (*)(cl_command_queue    /* command_queue */,
                        cl_mem              /* buffer */,
                        cl_bool             /* blocking_read */,
                        const size_t *      /* buffer_offset */,
                        const size_t *      /* host_offset */, 
                        const size_t *      /* region */,
                        size_t              /* buffer_row_pitch */,
                        size_t              /* buffer_slice_pitch */,
                        size_t              /* host_row_pitch */,
                        size_t              /* host_slice_pitch */,                        
                        void *              /* ptr */,
                        cl_uint             /* num_events_in_wait_list */,
                        const cl_event *    /* event_wait_list */,
                        cl_event *          /* event */)) dlsym(handle,"clEnqueueReadBufferRect");
                            
 IAH();
 aclEnqueueWriteBuffer = (cl_int  (*)(cl_command_queue   /* command_queue */, 
                     cl_mem             /* buffer */, 
                     cl_bool            /* blocking_write */, 
                     size_t             /* offset */, 
                     size_t             /* size */, 
                     const void *       /* ptr */, 
                     cl_uint            /* num_events_in_wait_list */, 
                     const cl_event *   /* event_wait_list */, 
                     cl_event *         /* event */)) dlsym(handle,"clEnqueueWriteBuffer");
                            
 IAH();
 aclEnqueueWriteBufferRect = (cl_int  (*)(cl_command_queue    /* command_queue */,
                         cl_mem              /* buffer */,
                         cl_bool             /* blocking_write */,
                         const size_t *      /* buffer_offset */,
                         const size_t *      /* host_offset */, 
                         const size_t *      /* region */,
                         size_t              /* buffer_row_pitch */,
                         size_t              /* buffer_slice_pitch */,
                         size_t              /* host_row_pitch */,
                         size_t              /* host_slice_pitch */,                        
                         const void *        /* ptr */,
                         cl_uint             /* num_events_in_wait_list */,
                         const cl_event *    /* event_wait_list */,
                         cl_event *          /* event */)) dlsym(handle,"clEnqueueWriteBufferRect");
                            
                            
 IAH();
 aclEnqueueCopyBuffer = (cl_int  (*)(cl_command_queue    /* command_queue */, 
                    cl_mem              /* src_buffer */,
                    cl_mem              /* dst_buffer */, 
                    size_t              /* src_offset */,
                    size_t              /* dst_offset */,
                    size_t              /* size */, 
                    cl_uint             /* num_events_in_wait_list */,
                    const cl_event *    /* event_wait_list */,
                    cl_event *          /* event */)) dlsym(handle,"clEnqueueCopyBuffer");
                            
 IAH();
 aclEnqueueCopyBufferRect = (cl_int  (*)(cl_command_queue    /* command_queue */, 
                        cl_mem              /* src_buffer */,
                        cl_mem              /* dst_buffer */, 
                        const size_t *      /* src_origin */,
                        const size_t *      /* dst_origin */,
                        const size_t *      /* region */, 
                        size_t              /* src_row_pitch */,
                        size_t              /* src_slice_pitch */,
                        size_t              /* dst_row_pitch */,
                        size_t              /* dst_slice_pitch */,
                        cl_uint             /* num_events_in_wait_list */,
                        const cl_event *    /* event_wait_list */,
                        cl_event *          /* event */)) dlsym(handle,"clEnqueueCopyBufferRect");
                            
 IAH();
 aclEnqueueReadImage = (cl_int  (*)(cl_command_queue     /* command_queue */,
                   cl_mem               /* image */,
                   cl_bool              /* blocking_read */, 
                   const size_t *       /* origin[3] */,
                   const size_t *       /* region[3] */,
                   size_t               /* row_pitch */,
                   size_t               /* slice_pitch */, 
                   void *               /* ptr */,
                   cl_uint              /* num_events_in_wait_list */,
                   const cl_event *     /* event_wait_list */,
                   cl_event *           /* event */)) dlsym(handle,"clEnqueueReadImage");

 IAH();
 aclEnqueueWriteImage = (cl_int  (*)(cl_command_queue    /* command_queue */,
                    cl_mem              /* image */,
                    cl_bool             /* blocking_write */, 
                    const size_t *      /* origin[3] */,
                    const size_t *      /* region[3] */,
                    size_t              /* input_row_pitch */,
                    size_t              /* input_slice_pitch */, 
                    const void *        /* ptr */,
                    cl_uint             /* num_events_in_wait_list */,
                    const cl_event *    /* event_wait_list */,
                    cl_event *          /* event */)) dlsym(handle,"clEnqueueWriteImage");

                            
 IAH();
 aclEnqueueCopyImage = (cl_int  (*)(cl_command_queue     /* command_queue */,
                   cl_mem               /* src_image */,
                   cl_mem               /* dst_image */, 
                   const size_t *       /* src_origin[3] */,
                   const size_t *       /* dst_origin[3] */,
                   const size_t *       /* region[3] */, 
                   cl_uint              /* num_events_in_wait_list */,
                   const cl_event *     /* event_wait_list */,
                   cl_event *           /* event */)) dlsym(handle,"clEnqueueCopyImage");

 IAH();
 aclEnqueueCopyImageToBuffer = (cl_int  (*)(cl_command_queue /* command_queue */,
                           cl_mem           /* src_image */,
                           cl_mem           /* dst_buffer */, 
                           const size_t *   /* src_origin[3] */,
                           const size_t *   /* region[3] */, 
                           size_t           /* dst_offset */,
                           cl_uint          /* num_events_in_wait_list */,
                           const cl_event * /* event_wait_list */,
                           cl_event *       /* event */)) dlsym(handle,"clEnqueueCopyImageToBuffer");

 IAH();
 aclEnqueueCopyBufferToImage = (cl_int  (*)(cl_command_queue /* command_queue */,
                           cl_mem           /* src_buffer */,
                           cl_mem           /* dst_image */, 
                           size_t           /* src_offset */,
                           const size_t *   /* dst_origin[3] */,
                           const size_t *   /* region[3] */, 
                           cl_uint          /* num_events_in_wait_list */,
                           const cl_event * /* event_wait_list */,
                           cl_event *       /* event */)) dlsym(handle,"clEnqueueCopyBufferToImage");
#if 0
void * 
(*aclEnqueueMapBuffer)(cl_command_queue /* command_queue */,
                   cl_mem           /* buffer */,
                   cl_bool          /* blocking_map */, 
                   cl_map_flags     /* map_flags */,
                   size_t           /* offset */,
                   size_t           /* size */,
                   cl_uint          /* num_events_in_wait_list */,
                   const cl_event * /* event_wait_list */,
                   cl_event *       /* event */,
                   cl_int *         /* errcode_ret */)) dlsym(handle,"");

void * 
(*aclEnqueueMapImage)(cl_command_queue  /* command_queue */,
                  cl_mem            /* image */, 
                  cl_bool           /* blocking_map */, 
                  cl_map_flags      /* map_flags */, 
                  const size_t *    /* origin[3] */,
                  const size_t *    /* region[3] */,
                  size_t *          /* image_row_pitch */,
                  size_t *          /* image_slice_pitch */,
                  cl_uint           /* num_events_in_wait_list */,
                  const cl_event *  /* event_wait_list */,
                  cl_event *        /* event */,
                  cl_int *          /* errcode_ret */)) dlsym(handle,"");
#endif

 IAH();
 aclEnqueueUnmapMemObject = (cl_int  (*)(cl_command_queue /* command_queue */,
                        cl_mem           /* memobj */,
                        void *           /* mapped_ptr */,
                        cl_uint          /* num_events_in_wait_list */,
                        const cl_event *  /* event_wait_list */,
                        cl_event *        /* event */)) dlsym(handle,"clEnqueueUnmapMemObject");

 IAH();
 aclEnqueueNDRangeKernel = (cl_int  (*)(cl_command_queue /* command_queue */,
                       cl_kernel        /* kernel */,
                       cl_uint          /* work_dim */,
                       const size_t *   /* global_work_offset */,
                       const size_t *   /* global_work_size */,
                       const size_t *   /* local_work_size */,
                       cl_uint          /* num_events_in_wait_list */,
                       const cl_event * /* event_wait_list */,
                       cl_event *       /* event */)) dlsym(handle,"clEnqueueNDRangeKernel");

 IAH();
 aclEnqueueTask = (cl_int  (*)(cl_command_queue  /* command_queue */,
              cl_kernel         /* kernel */,
              cl_uint           /* num_events_in_wait_list */,
              const cl_event *  /* event_wait_list */,
              cl_event *        /* event */)) dlsym(handle,"clEnqueueTask");

 IAH();
 aclEnqueueNativeKernel = (cl_int  (*)(cl_command_queue  /* command_queue */,
					  void (CL_CALLBACK * /*user_func*/)(void *), 
                      void *            /* args */,
                      size_t            /* cb_args */, 
                      cl_uint           /* num_mem_objects */,
                      const cl_mem *    /* mem_list */,
                      const void **     /* args_mem_loc */,
                      cl_uint           /* num_events_in_wait_list */,
                      const cl_event *  /* event_wait_list */,
                      cl_event *        /* event */)) dlsym(handle,"clEnqueueNativeKernel");

  loadedCL = 1;
}


