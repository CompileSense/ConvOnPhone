package com.compilesense.liuyi.convonphone;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.TextView;

import com.compilesense.liuyi.convonphone.algorithm.Convolution;
import com.compilesense.liuyi.convonphone.algorithm.Mask;
import com.compilesense.liuyi.convonphone.algorithm.MockImage;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";
    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("native-lib");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        findViewById(R.id.bt_conv_native).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                convTest();
//                testConv();
            }
        });
        findViewById(R.id.bt_conv_java).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                testConv();
            }
        });
        findViewById(R.id.bt_blas_test).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                blasTest();
            }
        });

    }

    @Override
    protected void onResume() {
        super.onResume();

//        double scale = 8d/24d;
//        if (result != null){
//            for (int i = 0; i<result.length; i++){
//                Log.d("result","result["+i+"]:"+(result[i]*scale));
//            }
//        }
    }

    void testConv(){
        int sizeMockImages = 100;
        MockImage[] mockImages = new MockImage[sizeMockImages];
        long st = System.currentTimeMillis();
        for (int i = 0; i<sizeMockImages; i++){
            mockImages[i] = new MockImage();
        }
        Log.d(TAG,"MockImage init time:" + (System.currentTimeMillis() - st));

        st = System.currentTimeMillis();
        Mask[] masks = new Mask[10];
        for (int i = 0; i < masks.length; i++){
            masks[i] = new Mask();
        }
        Log.d(TAG,"masks init time:" + (System.currentTimeMillis() - st));

        int[][] result = new int[1000][];
        Convolution convolution = new Convolution();

        st = System.currentTimeMillis();
        int index = 0;
        for (Mask mask : masks) {
            for (MockImage image : mockImages){
                result[index] = convolution.conv(mask.getMask(), image);
                index++;
            }
        }
        Log.d(TAG,"conv time:" + (System.currentTimeMillis() - st));
        Log.d(TAG,"conv result.length:" + result.length);
        Log.d(TAG,"conv result[0].length:" + result[0].length);
        Log.d(TAG,"conv result[0][0]:" + result[0][0]);
    }

    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
    public native String convTest();

    public native String blasTest();
}
