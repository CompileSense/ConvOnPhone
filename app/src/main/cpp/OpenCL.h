//
// Created by 神经元002 on 2017/2/17.
//

#ifndef CONVONPHONE_OPENCL_H
#define CONVONPHONE_OPENCL_H
#include <CL/cl.h>
#include <android/log.h>
#include <jni.h>

#define LOG_TAG "OpenCL"
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))
#define  LOGE(...)  __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

/**
 * 初始化 OpenCL
 * @param *env JVM 指针
 * @param openCLProgramText 内核程序
 */
void initOpenCL(JNIEnv * env,  jstring openCLProgramText);
void shutdownOpenCL();//关闭 OpenCL

void dot(float* colMat, const int colMatSize,
         float* maskMat, const int maskMatSize,
         float* dstMat);

void testOpenCLConv(float * srcColMat, unsigned int numSrcMat, unsigned int srcH, unsigned int srcW,
                    float * maskMat, unsigned int numMask, unsigned int maskSize,
                    float * dstMat, unsigned int dstH, unsigned int dstW);//测试 OpenCL 卷积内核函数

void vectorAdd(int* v1, int* v2, int* result, unsigned int size);

#endif //CONVONPHONE_OPENCL_H
