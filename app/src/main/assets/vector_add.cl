kernel void addKernel(
	global const int* v1, 	//输入矩阵
	global const int* v2,	//卷积算子
	global int* out       	//输出矩阵
)
{
    int index = get_global_id(0);	//第一维度位置
    int size = get_global_size(0);

    out[index] = v1[index] + v2[index];

}