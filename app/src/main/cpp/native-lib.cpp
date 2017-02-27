#include <jni.h>
#include <string>
#include <android/log.h>
#include <vector>
#include <time.h>
#include "Convolution.h"
#include "cblas.h"
#include "OpenCL.h"
#include "neonTest.h"

#define LOG_TAG "COP_Native"
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))

static double now_ms(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec*1000. + tv.tv_usec/1000.;
}

struct TestData{
    bool hadInit = false;

    const int srcNum = 100;
    const int srcSize = 25;
    float srcMat[100*5*5];
    const int srcColSize =81;
    float srcColMat[100*9*9];
    const int maskNum = 10;
    const int maskSize = 9;
    float maskMat[10*3*3];
//    float dstMat[100*10*3*3];
} testData;

//用于验证卷积是否正确
struct TestData2{
    float srcMat[SRC_H*SRC_W]={
            1,2,3,4,5,
            1,2,3,4,5,
            1,2,3,4,5,
            1,2,3,4,5,
            1,2,3,4,5 };
    float srcColMat[81]={
            1,2,3,1,2,3,1,2,3,
            2,3,4,2,3,4,2,3,4,
            3,4,5,3,4,5,3,4,5,
            1,2,3,1,2,3,1,2,3,
            2,3,4,2,3,4,2,3,4,
            3,4,5,3,4,5,3,4,5,
            1,2,3,1,2,3,1,2,3,
            2,3,4,2,3,4,2,3,4,
            3,4,5,3,4,5,3,4,5,};
    float maskMat[MASK_H * MASK_W] = {1,2,3,4,5,6,7,8,9};
} testData2;

void generateTestData(float* srcDate, const int srcNum,
                      float* srcColData,
                      float* maskDate, const int maskNum
){
    double st = now_ms();
    getRandomMask(maskDate, maskNum);
    LOGD("mask init time: %lf", now_ms() - st);

    st = now_ms();
    const int srcSize = 25;
    for (int i = 0; i < srcNum; ++i) {
        getRandomSrcData(&srcDate[i*srcSize]);
    }

    LOGD("src init time: %lf", now_ms() - st);


    st = now_ms();
    im2col(srcDate, srcNum, srcColData);
    LOGD("im2col time: %lf", now_ms() - st);
}

void generateTestData(){
    if (!testData.hadInit){
        generateTestData(testData.srcMat,100,
                         testData.srcColMat,
                         testData.maskMat,10);
        testData.hadInit = true;
    }
}



#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT void JNICALL
Java_com_compilesense_liuyi_convonphone_MainActivity_testOpenCLConv(JNIEnv *env, jobject instance) {

//    double st = now_ms();
//    float A[SRC_H*SRC_W]={
//            1,2,3,4,5,
//            1,2,3,4,5,
//            1,2,3,4,5,
//            1,2,3,4,5,
//            1,2,3,4,5 };//test
//
//    float colMat[9*16];
//    im2col_vector8( A, colMat);
//
//    float B[MASK_H * MASK_W] = {1,2,3,4,5,6,7,8,9};//test
//    float dst[DST_H * DST_W];


    double st = now_ms();
    float masks[10][9];// 3*3 = 9
    for (int i = 0; i < 10; ++i) {
        getRandomMask(masks[i]);
    }
    LOGD("mask init time: %lf", now_ms() - st);

    st = now_ms();
    float srcDate[100][SRC_H * SRC_W];
    for (int i = 0; i < 100; ++i) {
        getRandomSrcData(srcDate[i]);
    }
    LOGD("src init time: %lf", now_ms() - st);

    st = now_ms();
    const int im2col_dst_h = 9;
    const int im2col_dst_w = 16;
    float  srcColData[100][im2col_dst_h][im2col_dst_w];
    im2col((float *) srcDate, 100, (float *) srcColData);
    LOGD("im2col time: %lf", now_ms() - st);
    st = now_ms();
    float dst[1000][DST_H * DST_W];
    dot((float *) srcColData, 9 * (3*3) * 100,
        (float *) masks, 10 * (3*3),
        (float *) dst
    );
    LOGD("cl conv time: %lf", now_ms() - st);

//    for (int j = 0; j < 9; ++j){
//            LOGD("result[%i]:%f",j,dst[j]);
//    }

}

JNIEXPORT void JNICALL
Java_com_compilesense_liuyi_convonphone_MainActivity_shutdownOpenCL(JNIEnv *env, jobject instance) {

    shutdownOpenCL();
    LOGD("shutdownOpenCL(openCLObjects) was called");
}

JNIEXPORT void JNICALL
Java_com_compilesense_liuyi_convonphone_MainActivity_intiOpenCL(JNIEnv *env, jobject instance,
                                                                jstring openCLProgramText_) {
    initOpenCL(env, openCLProgramText_);
}

JNIEXPORT void JNICALL
Java_com_compilesense_liuyi_convonphone_MainActivity_intiOpenCL2(JNIEnv *env, jobject instance,
                                                                jstring openCLProgramText_,
                                                                jstring openCLHeaderText_) {
    initOpenCL(env, openCLProgramText_);
};

JNIEXPORT void JNICALL
Java_com_compilesense_liuyi_convonphone_MainActivity_convTestNeon(JNIEnv *env, jobject instance) {
    if (!testData.hadInit){
        generateTestData();
    }
    double st = now_ms();
    float dstMat[1000*9];
    int index = 0;
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 100; ++j) {
            matVecDot(&testData.srcColMat[j * 81], &testData.maskMat[i * 9], &dstMat[index * 9]);
//            mVDot(&testData.srcColMat[j * 81], &testData.maskMat[i * 9], &dstMat[index * 9]);
            index++;
        }
    }
    LOGD("neon conv time: %lf", now_ms() - st);
    //正确验证
//    float dst[9];
//    matVecDot(testData2.srcColMat, testData2.maskMat, dst);
//    for (int i = 0; i < 9; ++i) {
//        LOGD("result [%i]:%f",i,dst[i]);
//    }
}

jstring
Java_com_compilesense_liuyi_convonphone_MainActivity_convTest(
        JNIEnv *env,
        jobject /* this */) {
    std::string hello = "Hello from C++";

    double st = now_ms();
    if (!testData.hadInit){
        generateTestData();
    }

    st = now_ms();
    int index = 0;
    float dst[1000][DST_H*DST_W];
    for (int i = 0; i < testData.maskNum; ++i) {
        for (int j = 0; j < 100; ++j) {
            conv2F(&testData.srcMat[j*testData.srcSize],
                   &testData.maskMat[i*testData.maskSize],
                   dst[index]);
            index++;
        }
    }
    LOGD("native conv time: %lf", now_ms() - st);


    return env->NewStringUTF(hello.c_str());
}

jstring
Java_com_compilesense_liuyi_convonphone_MainActivity_blasTest(
        JNIEnv *env,
        jobject /* this */) {
    std::string hello = "Hello from C++";

    double st;
    const int im2col_dst_h = (SRC_H - MASK_H +1)*(SRC_W - MASK_W + 1);
    const int im2col_dst_w = MASK_H * MASK_H;

    if (!testData.hadInit){
        generateTestData();
    }

    st = now_ms();
    int index = 0;
    float dst[1000][DST_H * DST_W];
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 100; ++j) {
            blasConvColData(&testData.srcColMat[j*81],
                            im2col_dst_h, im2col_dst_w,
                            &testData.maskMat[i*9], MASK_H, MASK_W,
                            dst[index]);
            index++;
        }
    }
    LOGD("blas conv time: %lf", now_ms() - st);


    return env->NewStringUTF(hello.c_str());
}
#ifdef __cplusplus
}
#endif
