#include <jni.h>
#include <string>
#include <android/log.h>
#include <vector>
#include <time.h>
#include "Convolution.h"
#include "cblas.h"
#include "OpenCL.h"

#define LOG_TAG "COP_Native"
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))

static double now_ms(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec*1000. + tv.tv_usec/1000.;
}

void cblastest()
{
    blasint n = 10;
    blasint in_x =1;
    blasint in_y =1;

    std::vector<double> x(n);
    std::vector<double> y(n);

    double alpha = 10;

    std::fill(x.begin(),x.end(),1.0);
    std::fill(y.begin(),y.end(),2.0);

    cblas_daxpy( n, alpha, &x[0], in_x, &y[0], in_y);

    //Print y
    for(int j=0;j<n;j++)
    {
        LOGD("print y[%i]=%i",j,y[j]);
    }

}




#ifdef __cplusplus
extern "C" {
#endif
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

jstring
Java_com_compilesense_liuyi_convonphone_MainActivity_convTest(
        JNIEnv *env,
        jobject /* this */) {
    std::string hello = "Hello from C++";

    double st = now_ms();
    int masks[10][MASK_H][MASK_W];
    for (int i = 0; i < 10; ++i) {
        getRandomMask(masks[i]);
    }
    LOGD("mask init time: %lf", now_ms() - st);

    st = now_ms();
    int srcDate[100][SRC_H][SRC_W];
    for (int i = 0; i < 100; ++i) {
        getRandomSrcData(srcDate[i]);
    }
    LOGD("src init time: %lf", now_ms() - st);

    st = now_ms();
    int index = 0;
    int dst[1000][DST_H][DST_W];
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 100; ++j) {
            conv2(srcDate[j], masks[i], dst[index]);
            index++;
        }
    }
    LOGD("conv time: %lf", now_ms() - st);



    return env->NewStringUTF(hello.c_str());
}

jstring
Java_com_compilesense_liuyi_convonphone_MainActivity_blasTest(
        JNIEnv *env,
        jobject /* this */) {
    std::string hello = "Hello from C++";

    double st = now_ms();
    float masks[10][MASK_H * MASK_W];
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
    const int im2col_dst_h = (SRC_H - MASK_H +1)*(SRC_W - MASK_W + 1);
    const int im2col_dst_w = MASK_H * MASK_H;
    float  srcColData[100][im2col_dst_h][im2col_dst_w];
    im2col((int *) srcDate, SRC_H, SRC_W, 100, MASK_W, (float *) srcColData);
    LOGD("im2col time: %lf", now_ms() - st);

    st = now_ms();
    int index = 0;
    float dst[1000][DST_H * DST_W];
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 100; ++j) {
//            blasConv((int *) srcDate[j], SRC_H, SRC_W,
//                     (float *) masks[i], MASK_H, MASK_W,
//                     (float *) dst[index]);
            blasConvColData((float *) srcColData[j], im2col_dst_h, im2col_dst_w,
                            masks[i], MASK_H, MASK_W,
                            dst[index]);
            index++;
        }
    }
    LOGD("conv time: %lf", now_ms() - st);


    return env->NewStringUTF(hello.c_str());
}
#ifdef __cplusplus
}
#endif
