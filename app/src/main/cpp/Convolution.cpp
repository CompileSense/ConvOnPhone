//
// Created by liuyi on 2017/2/14.
//

#include <stdlib.h>
#include <cblas.h>
#include <android/log.h>
#include "Convolution.h"
#include <arm_neon.h>

#define LOG_TAG "COP_Native"
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))
#define Random(x) (rand() % x) //通过取余取得指定范围的随机数

void print2DArray(int* array, int h, int w){
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j){
            LOGD("print2DArray array[%i][%i]=%i",i,j,array[i*w + j]);
        }
    }
}

void print2DArray(float * array, int h, int w){
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j){
            LOGD("print2DArray array[%i][%i]=%f",i,j,array[i*w + j]);
        }
    }
}


void print2DArrayF(const float * array, int h, int w){
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j){
            LOGD("print2DArray array[%i][%i]=%f",i,j,array[i*w + j]);
        }
    }
}

void setDataTo2DArray(int** array, int data, int w, int x, int y){
    *((int*) array + x*w + y) = data;
}

void setDataTo2DArray(int** array, float data, int w, int x, int y){
    *((int*) array + x*w + y) = (int) data;
}


int getDataFrom2DArray (int** array, int w, int x, int y){
    return *((int*) array + x*w + y);
}

float getDataFrom2DArray (float ** array, int w, int x, int y){
    return *((float*) array + x*w + y);
}

void oneD2TowD (int* input, int** output, int h, int w){
    int index = 0;
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j){
            setDataTo2DArray(output, input[i], w, i, j);
            index++;
        }
    }
}

void oneD2TowD (float* input, int** output, int h, int w){
    int index = 0;
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j){
            setDataTo2DArray(output, input[index], w, i, j);
            index++;
        }
    }
}

void tD2OneD (int** input, int h, int w, float * output){
    int index = 0;
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j){
            output[index] = getDataFrom2DArray(input,w,i,j);
            index++;
        }
    }
}

void tD2OneD (int** input, int h, int w, int* output){
    int index = 0;
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j){
            output[index] = getDataFrom2DArray(input,w,i,j);
            index++;
        }
    }
}

void conv2 (int x[SRC_H][SRC_W], int y[MASK_H][MASK_W], int z[DST_H][DST_W])
{
    int i, j, n, m;
    for (i = 0; i < DST_H; i++) {
        for (j = 0; j<DST_W; j++) {
            int temp = 0;
            for (m = 0; m < MASK_H; m++) {
                for (n = 0; n < MASK_W; n++) {
                    temp += y[m][n]*x[i+m][j+n];
                }
            }
            z[i][j] = temp;
        }
    }
}

void conv2F (float* x, float* y, float* z)
{
    int i, j, n, m;
    for (i = 0; i < DST_H; i++) {
        for (j = 0; j < DST_W; j++) {
            int temp = 0;
            #pragma unroll
            for (m = 0; m < MASK_H; m++) {
                #pragma unroll
                for (n = 0; n < MASK_W; n++) {
                    temp += y[m * MASK_W + n] * x[(i+m) * SRC_W + j+n];
                }
            }
            z[i+j] = temp;
        }
    }
}


void getRandomSrcData (int data[SRC_H][SRC_W]) {
    for (int i = 0; i < DST_H; i++) {
        for (int j = 0; j < DST_W; j++) {
            data[i][j] = Random(99);
        }
    }
}

void getRandomSrcData (float data[SRC_H * SRC_W]) {
    for (int i = 0; i < DST_H * SRC_W; i++) {
        data[i] = Random(99);
    }
}

void getRandomMask(int mask[MASK_H][MASK_H]){
    for (int i = 0; i < MASK_H; i++) {
        for (int j = 0; j < MASK_H; j++) {
            mask[i][j] = Random(9);
        }
    }
}

void getRandomMask(float mask[MASK_H * MASK_H]){
    for (int i = 0; i < MASK_H * MASK_W; i++) {
        mask[i] = Random(99);
    }
}
void getRandomMask(float* mask, const int maskNum){

    for (int j = 0; j < maskNum; ++j) {
        for (int i = 0; i < MASK_H * MASK_W; i++) {
            mask[i + j * MASK_H * MASK_W] = Random(99);
        }
    }
}

void getRandomMask_vector8(float *mask){
    for (int i = 0; i < 16; i++) {
        if (i <= 8){
            mask[i] = Random(10);
        } else{
            mask[i] = 0;
        }
    }
}

//实现 im2col 算法
void im2col (int** input, int i_h, int i_w, int maskSize, float *result){
    int dst_h = (i_h - maskSize +1)*(i_w - maskSize + 1);
    int dst_w = maskSize*maskSize;
    for (int i = 0; i < dst_h; ++i){
        int indexX = 0;
        int indexY = 0;

        for (int j = 0; j < dst_w; ++j) {
            if (j%maskSize == 0 && j != 0){
                ++indexY;
                indexX = 0;
            }

            int offSetX = (i<maskSize)? i : i%maskSize;
            int offSetY = i/maskSize;
            int inputI = getDataFrom2DArray(input,i_w, indexY + offSetY, indexX + offSetX);
            result[i*dst_w + j] = inputI;
            ++indexX;
        }
    }
}

void im2col (float** input, int i_h, int i_w, int maskSize, float *result){
    int dst_h = (i_h - maskSize +1)*(i_w - maskSize + 1);
    int dst_w = maskSize*maskSize;
    for (int i = 0; i < dst_h; ++i){
        int indexX = 0;
        int indexY = 0;

        for (int j = 0; j < dst_w; ++j) {
            if (j%maskSize == 0 && j != 0){
                ++indexY;
                indexX = 0;
            }

            int offSetX = (i<maskSize)? i : i%maskSize;
            int offSetY = i/maskSize;
            float inputI = getDataFrom2DArray(input,i_w, indexY + offSetY, indexX + offSetX);
            result[i*dst_w + j] = inputI;
            ++indexX;
        }
    }
}

void im2col_vector8 (float* input, float *result){
    const int i_w = 5;
    const int dst_w = 16;
    const int dst_h = 9;
    const int maskSize = 3;

    for (int i = 0; i < dst_h; ++i){
        int indexX = 0;
        int indexY = 0;

        for (int j = 0; j < dst_w; ++j) {
            if (j > 8){
                result[i*dst_w + j] = 0;
                continue;
            }

            if (j%maskSize == 0 && j != 0){
                ++indexY;
                indexX = 0;
            }

            int offSetX = i % maskSize;
            int offSetY = i / maskSize;
            float inputI = input[(indexY + offSetY)*i_w + (indexX + offSetX)];
            result[i*dst_w + j] = inputI;
            ++indexX;
        }
    }
}

void im2col (int* input, int i_h, int i_w, int maskSize, float *result){
    int dst_h = (i_h - maskSize +1)*(i_w - maskSize + 1);
    int dst_w = maskSize*maskSize;
    for (int i = 0; i < dst_h; ++i){
        int indexX = 0;
        int indexY = 0;

        for (int j = 0; j < dst_w; ++j) {
            if (j%maskSize == 0 && j != 0){
                ++indexY;
                indexX = 0;
            }
            int offSetX = (i<maskSize)? i : i%maskSize;
            int offSetY = i/maskSize;
            int inputI = input[(indexY + offSetY)*i_w + (indexX + offSetX)];
            result[i*dst_w + j] = inputI;
            ++indexX;
        }
    }
}

//批量转换
void im2col (float * input, int i_elevation, float *result){


    const int src_w = 5;

    const int maskSize = 3;
    const int dst_h = 9;
    const int dst_w = 9;

    const int srcFrameSize = 25;
    const int dstFrameSize = 81;

    for (int i = 0; i < dst_h; ++i){
        int indexX = 0;
        int indexY = 0;

        for (int j = 0; j < dst_w; ++j) {
            if (j%maskSize == 0 && j != 0){
                ++indexY;
                indexX = 0;
            }

            int offSetX = i%maskSize;
            int offSetY = i/maskSize;

            for (int e = 0; e < i_elevation; ++e) {


                float inputI = input[
                        (indexY + offSetY)*src_w
                        + (indexX + offSetX)
                        + e * srcFrameSize];
//                if (e == 0){
//                    LOGD("im2col inSide, i:%i, j:%i, elevation:%i, input:%f",i,j,e,inputI);
//                }
                result[i*dst_w + j + e * dstFrameSize] = inputI;
            }
            ++indexX;
        }
    }
}


void im2col_vector8 (float * input,  int i_elevation, float *result){
    const int maskSize = 3;
    const int i_h = 5;
    const int i_w = 5;

    const int dst_h = 9;
    const int dst_w = 16;

    const int srcFrameSize = i_h * i_w;
    const int dstFrameSize = dst_h * dst_w;

    for (int i = 0; i < dst_h; ++i){
        int indexX = 0;
        int indexY = 0;

        for (int j = 0; j < dst_w; ++j) {
            if (j%maskSize == 0 && j != 0){
                ++indexY;
                indexX = 0;
            }
            int offSetX = i%maskSize;
            int offSetY = i/maskSize;

            for (int elevation = 0; elevation < i_elevation; ++elevation) {

                if (j > 8 ){
                    result[i*dst_w + j + elevation * dstFrameSize] = 0;
                } else{
                    float inputI = input[
                            (indexY + offSetY)*i_w
                            + (indexX + offSetX)
                            + elevation * srcFrameSize];
                    result[i*dst_w + j + elevation * dstFrameSize] = inputI;
                }
            }

            ++indexX;
        }
    }
}



void blasConv(int** matSrc, int srcH, int srcW,
              int** matMask, int maskH, int maskW,
              int** matDst){

    int im2col_dst_h = (srcH - maskH +1)*(srcW - maskW + 1);
    int im2col_dst_w = maskH * maskW;
    float inData [im2col_dst_h * im2col_dst_w];
    im2col(matSrc, srcH, srcW, maskH, inData);

    float maskData[maskH*maskW];
    tD2OneD(matMask, maskH, maskW, maskData);

    const enum CBLAS_ORDER Order=CblasRowMajor;
    const enum CBLAS_TRANSPOSE TransA=CblasNoTrans;
    const enum CBLAS_TRANSPOSE TransB=CblasNoTrans;
    const int M=im2col_dst_h;//A的行数，C的行数
    const int N=1;//B的列数，C的列数
    const int K=im2col_dst_w;//A的列数，B的行数
    const float alpha=1;
    const float beta=0;
    const int lda=K;//A的列
    const int ldb=N;//B的列
    const int ldc=N;//C的列

//    const float A[M*K]={1,2,3,1,2,3,1,2,3,
//                        2,3,4,2,3,4,2,3,4,
//                        3,4,5,3,4,5,3,4,5,
//                        1,2,3,1,2,3,1,2,3,
//                        2,3,4,2,3,4,2,3,4,
//                        3,4,5,3,4,5,3,4,5,
//                        1,2,3,1,2,3,1,2,3,
//                        2,3,4,2,3,4,2,3,4,
//                        3,4,5,3,4,5,3,4,5,};//test
//
//    const float B[K*N]={1,2,3,4,5,6,7,8,9};//test

    float C[M*N];
    cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, inData, lda, maskData, ldb, beta, C, ldc);

    int matDstH = srcH - maskH + 1;
    int matDstW = srcW - maskW + 1;

    oneD2TowD(C, matDst, matDstH, matDstW);

//    LOGD("print dstMat:");
//    for(int i=0; i<matDstH; i++)
//    {
//        for(int j=0; j<matDstW; j++)
//        {
//            LOGD("matDst[%i][%i]=%i",i,j,getDataFrom2DArray(matDst,matDstW,i,j));
//        }
//
//    }
//    LOGD("print dstMat end.");

}

void blasConv(int* matSrc, int srcH, int srcW,
              float* matMask, int maskH, int maskW,
              float * matDst){

    int im2col_dst_h = (srcH - maskH +1)*(srcW - maskW + 1);
    int im2col_dst_w = maskH * maskW;
    float inData [im2col_dst_h * im2col_dst_w];
    im2col(matSrc, srcH, srcW, maskH, inData);

    const enum CBLAS_ORDER Order=CblasRowMajor;
    const enum CBLAS_TRANSPOSE TransA=CblasNoTrans;
    const enum CBLAS_TRANSPOSE TransB=CblasNoTrans;
    const int M=im2col_dst_h;//A的行数，C的行数
    const int N=1;//B的列数，C的列数
    const int K=im2col_dst_w;//A的列数，B的行数
    const float alpha=1;
    const float beta=0;
    const int lda=K;//A的列
    const int ldb=N;//B的列
    const int ldc=N;//C的列

    cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, inData, lda, matMask, ldb, beta, matDst, ldc);
}

void blasConvColData(float * matColSrc, int colSrcH, int colSrcW,
              float* matMask, int maskH, int maskW,
              float * matDst){

//    int im2col_dst_h = (srcH - maskH +1)*(srcW - maskW + 1);
//    int im2col_dst_w = maskH * maskW;
//    float inData [im2col_dst_h * im2col_dst_w];
//    im2col(matSrc, srcH, srcW, maskH, inData);

    const enum CBLAS_ORDER Order=CblasRowMajor;
    const enum CBLAS_TRANSPOSE TransA=CblasNoTrans;
    const enum CBLAS_TRANSPOSE TransB=CblasNoTrans;
    const int M=colSrcH;//A的行数，C的行数
    const int N=1;//B的列数，C的列数
    const int K=colSrcW;//A的列数，B的行数
    const float alpha=1;
    const float beta=0;
    const int lda=K;//A的列
    const int ldb=N;//B的列
    const int ldc=N;//C的列
    cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, matColSrc, lda, matMask, ldb, beta, matDst, ldc);
}
//不使用 neon 对比
void mVDot(float* srcData, float* maskData, float* dstData){
    const int srcColH = 9;
    const int srcColW = 9;

    for (int i = 0; i < srcColH; i++) {
        float temp = 0;
        for (int j = 0; j < srcColW; ++j) {
            float srcItem = srcData[i*9 + j];
            float maskItem = maskData[j];

            temp += srcItem*maskItem;
        }
        dstData[i] = temp;
    }
}

//TODO NEON 有16个64位向量寄存器,充分利用
void matVecDot(float* srcData, float* maskData, float* dstData){
    const int srcColH = 9;
    const int srcColW = 9;

    const float32x4_t maskV1 = vld1q_f32(maskData);
    const float32x4_t maskV2 = vld1q_f32(maskData + 4);
    const float maskLast = maskData[8];

    int index = 0;

    for (int i = 0; i < srcColH; i++) {
        float32x4_t srcV1 = vld1q_f32(srcData + index);
        float32x4_t srcV2 = vld1q_f32(srcData + index + 4);
        float srcLast = srcData[index + 8];
        index += srcColW;

        float32x4_t rV = vmulq_f32(srcV1,maskV1);
        rV = vmlaq_f32(rV,srcV2,maskV2);

        float result = vaddvq_f32(rV) + srcLast * maskLast;
        dstData[i] = result;
    }
}

/** 尝试同时使用 11 个向量寄存器,视乎效果不明显。
      for (int i = 0; i < srcColH; i += 3) {
      float32x4_t srcV1 = vld1q_f32(srcData + index*9);
      float32x4_t srcV2 = vld1q_f32(srcData + index*9 + 4);
      float srcLast = srcData[index*9 + 8];
      index += srcColW;

      float32x4_t rV = vmulq_f32(srcV1, maskV1);
      rV = vmlaq_f32(rV,srcV2,maskV2);

      float result = vaddvq_f32(rV) + srcLast * maskLast;
      dstData[i] = result;

      float32x4_t src2V1 = vld1q_f32(srcData + index*9);
      float32x4_t src2V2 = vld1q_f32(srcData + index*9 + 4);
      float src2Last = srcData[index*9 + 8];
      index += srcColW;

      float32x4_t r2V = vmulq_f32(src2V1, maskV1);
      r2V = vmlaq_f32(rV,src2V2,maskV2);

      float result2 = vaddvq_f32(r2V) + src2Last * maskLast;
      dstData[i+1] = result2;

      float32x4_t src3V1 = vld1q_f32(srcData + index*9);
      float32x4_t src3V2 = vld1q_f32(srcData + index*9 + 4);
      float src3Last = srcData[index*9 + 8];
      index += srcColW;

      float32x4_t r3V = vmulq_f32(src3V1,maskV1);
      r3V = vmlaq_f32(r3V,src3V2,maskV2);

      float result3 = vaddvq_f32(r3V) + srcLast * maskLast;
      dstData[i+2] = result3;
  }
   */

void blasTestS(){

    LOGD("blasTestS");
    int A[SRC_H][SRC_W]={ {1,2,3,4,5},
                          {1,2,3,4,5},
                          {1,2,3,4,5},
                          {1,2,3,4,5},
                          {1,2,3,4,5} };//test
    int B[MASK_H][MASK_W]={{1,2,3},{4,5,6},{7,8,9}};//test
    int C[DST_H][DST_W];

    blasConv((int **) A, SRC_H, SRC_W,
             (int **) B, MASK_H, MASK_W,
             (int **) C);
}


