package com.compilesense.liuyi.convonphone.algorithm;

/**
 * Created by shenjingyuan002 on 2017/2/14.
 */

public class Mask {
    public final static int MASK_SIZE = 3;
    public final static int TYPE_LAPLACIAN = 121;
    public final static int TYPE_TEST = 222;
    private double[][] mask;

    public Mask(){
        mask = new double[MASK_SIZE][MASK_SIZE];
        for (int i = 0; i < MASK_SIZE; i++){
            for (int j = 0; j < MASK_SIZE; j++){
                mask[i][j] = Math.random()*10;
            }
        }
    }

    public Mask(int type){
        if (type == TYPE_TEST){
            mask = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        }

        if (type == TYPE_LAPLACIAN){
            mask = new double[][]{{0, -1, 0}, {-1,4,-1}, {0,-1,0}};
        }
    }

    public double[][] getMask() {
        return mask;
    }
}
