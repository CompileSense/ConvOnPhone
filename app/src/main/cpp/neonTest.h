//
// Created by 神经元002 on 2017/2/21.
//

#ifndef CONVONPHONE_NEONTEST_H
#define CONVONPHONE_NEONTEST_H

#include <arm_neon.h>

void testNeon(short *output, const short* input, const short* kernel, int width, int kernelSize);

#endif //CONVONPHONE_NEONTEST_H
