//
// Created by 神经元002 on 2017/2/21.
//

#include "neonTest.h"

//neon 适合同时处理 8位向量。
void neon_convert (uint8_t * __restrict dest, uint8_t * __restrict src, int n)
{
    int i;
    uint8x8_t rfac = vdup_n_u8 (77);       // 转换权值  R
    uint8x8_t gfac = vdup_n_u8 (151);    // 转换权值  G
    uint8x8_t bfac = vdup_n_u8 (28);      // 转换权值  B
    n/=8;

    for (i=0; i<n; i++)
    {
        uint16x8_t  temp;
        uint8x8x3_t rgb  = vld3_u8 (src);
        uint8x8_t result;

        temp = vmull_u8 (rgb.val[0],      rfac);       // vmull_u8 每个字节（8bit）对应相乘，结果为每个单位2字节（16bit）
        temp = vmlal_u8 (temp,rgb.val[1], gfac);  // 每个比特对应相乘并加上
        temp = vmlal_u8 (temp,rgb.val[2], bfac);

        result = vshrn_n_u16 (temp, 8);  // 全部移位8位
        vst1_u8 (dest, result);   // 转存运算结果
        src  += 8*3;
        dest += 8;
    }
}

void im2col (float32_t* input, int i_h, int i_w, int i_elevation, int maskSize, float *result){
    int dst_h = (i_h - maskSize +1)*(i_w - maskSize + 1);
    int dst_w = maskSize*maskSize;
    int srcFrameSize = i_h * i_w;
    int dstFrameSize = dst_h * dst_w;

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
                int inputI = input[(indexY + offSetY)*i_w + (indexX + offSetX) + elevation * srcFrameSize];
                result[i*dst_w + j + elevation * dstFrameSize] = inputI;
            }
            ++indexX;
        }
    }
}


void testNeon(short *output, const short* input, const short* kernel, int width, int kernelSize) {

    int nn, offset = -kernelSize / 2;

    for (nn = 0; nn < width; nn++) {
        int mm, sum = 0;
        int32x4_t sum_vec = vdupq_n_s32(0);
        for (mm = 0; mm < kernelSize / 4; mm++) {
            int16x4_t kernel_vec = vld1_s16(kernel + mm * 4);
            int16x4_t input_vec = vld1_s16(input + (nn + offset + mm * 4));
            sum_vec = vmlal_s16(sum_vec, kernel_vec, input_vec);
        }

        sum += vgetq_lane_s32(sum_vec, 0);
        sum += vgetq_lane_s32(sum_vec, 1);
        sum += vgetq_lane_s32(sum_vec, 2);
        sum += vgetq_lane_s32(sum_vec, 3);

        if (kernelSize & 3) {
            for (mm = kernelSize - (kernelSize & 3); mm < kernelSize; mm++)
                sum += kernel[mm] * input[nn + offset + mm];
        }

        output[nn] = (short) ((sum + 0x8000) >> 16);
    }
}