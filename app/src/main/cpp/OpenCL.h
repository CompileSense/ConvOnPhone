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
void shutdownOpenCL();//关闭OpenCL

#endif //CONVONPHONE_OPENCL_H
