// 实现OpenCL的卷积计算
// Created by liuyi(695183065@qq.com) on 2017/2/18.

kernel void convolution(
	global const float srcMat[9*(3*3)*100], 	//输入矩阵 已经 im2col, 9*9=81;
	global const float maskMat[9*10],	//卷积算子
	global float dstMat[9*100*10]       //输出矩阵
)
{
    int x = get_global_id(0);	//index (cols/3) 9*3 = 27
    int y = get_global_id(1);	//index srcMats 100
    int z = get_global_id(2);   //index (maskMats/3) 10*3 = 30


    int l = get_local_id(0);
    int m = get_local_id(1);
    int n = get_local_id(2);

    int groupSize = get_local_size(0);

    local float resultArray[3];

    int offSet1 = (x + y*9);  // , prt = offSet * n
    float3 colVector = vload3(offSet1, srcMat);

    int offSet2 = z;
    float3 maskVector = vload3(offSet2, maskMat);

    float result = dot(colVector, maskVector);

    resultArray[l] = result;

    barrier(CLK_GLOBAL_MEM_FENCE);

    if(l == 0){
        int indexCols = x/3;
        int indexMasks = z/3;
        dstMat[indexCols + y*9 + indexMasks*900] = resultArray[0] + resultArray[1] + resultArray[2];
    }
}


