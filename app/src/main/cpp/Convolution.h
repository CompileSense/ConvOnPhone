//
// Created by 神经元002 on 2017/2/14.
//

#ifndef CONVONPHONE_CONVOLUTION_H
#define CONVONPHONE_CONVOLUTION_H

#define SRC_H 5
#define SRC_W 5
#define MASK_H 3
#define MASK_W 3
#define DST_H SRC_H - MASK_H + 1
#define DST_W SRC_W - MASK_W + 1

void conv2 (int x[SRC_H][SRC_W], int y[MASK_H][MASK_W], int z[DST_H][DST_W]);
void getRandomSrcData (int data[SRC_H][SRC_W]);
void getRandomSrcData (float data[SRC_H*SRC_W]);
void getRandomMask(int mask[MASK_H][MASK_H]);
void getRandomMask(float mask[MASK_H*MASK_H]);

void blasConv(int** matSrc, int srcH, int srcW,
              int** matMask, int maskH, int maskW,
              int** matDst);

void blasConv(int* matSrc, int srcH, int srcW,
              float* matMask, int maskH, int maskW,
              float* matDst);

void blasConvColData(float * matColSrc, int colSrcH, int colSrcW,
                     float* matMask, int maskH, int maskW,
                     float * matDst);

void im2col (int* input, int i_h, int i_w, int i_elevation, int maskSize, float *result);

void blasTestS();

#endif //CONVONPHONE_CONVOLUTION_H
