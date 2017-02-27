package com.compilesense.liuyi.convonphone.algorithm;

import android.util.Log;

/**
 * Created by shenjingyuan002 on 2017/2/14.
 */

public class MockImage {
    public static final int SIZE_MOCK_IMAGE = 5;
    public int w=SIZE_MOCK_IMAGE,h=SIZE_MOCK_IMAGE;
//    public int[][] data = new int[SIZE_MOCK_IMAGE][SIZE_MOCK_IMAGE];
    public int[][] data = { {1,2,3,4,5},
        {1,2,3,4,5},
        {1,2,3,4,5},
        {1,2,3,4,5},
        {1,2,3,4,5} };
    public MockImage(){
        for (int i = 0; i < SIZE_MOCK_IMAGE; i++){
            for (int j = 0; j < SIZE_MOCK_IMAGE; j++){
                data[i][j] = (int)(Math.random()*100);
//                data[i][j] = i*SIZE_MOCK_IMAGE + j;

//                Log.d("src", "src[" + i + "][" + j + "]:" + (data[i][j]));
            }
        }
    }

    public MockImage(boolean test){
    }
}
