package com.compilesense.liuyi.convonphone.algorithm;

/**
 * Created by shenjingyuan002 on 2017/2/14.
 */

public class Convolution {

    public int[] conv(double[][] mask, MockImage src) {
        int mh = mask.length;
        int mw = mask[1].length;
        int dst_h = src.h - mh +1;
        int dst_w = src.w - mw +1;

        int[] d= new int[dst_h*dst_w];
        int[][] src_data = src.data;

        for(int i = 0; i < dst_h;i++){
            for(int j = 0;j < dst_w;j++){
                int s = 0;
                for(int m=0; m<mh ; m++){
                    for(int n=0;n<mw;n++){
                        s = s + (int)(mask[m][n]*src_data[m+i][n+j]);
                    }
                }
                d[i * dst_w + j] = s;
            }
        }

        return d;
    }
}
