package com.compilesense.liuyi.convonphone;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;

import com.compilesense.liuyi.convonphone.algorithm.Convolution;
import com.compilesense.liuyi.convonphone.algorithm.Mask;
import com.compilesense.liuyi.convonphone.algorithm.MockImage;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

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

        findViewById(R.id.bt_conv_native_neon).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                convTestNeon();
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

        findViewById(R.id.bt_openCL_init).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                intiOpenCL(getOpenCLProgram());
            }
        });

        findViewById(R.id.bt_openCL_conv).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                testOpenCLConv();
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


//        Convolution convolution = new Convolution();
//        Mask mask = new Mask(Mask.TYPE_TEST);
//        MockImage image = new MockImage(true);
//        int[] result = new int[9];
//        result = convolution.conv(mask.getMask(), image);
//        for (int i = 0; i < 9; ++i) {
//            Log.d(TAG,"result:"+result[i]);
//
//        }

    }

    private String getOpenCLProgram ()
    {
        /* OpenCL program text is stored in a separate file in
         * assets directory. Here you need to load it as a single
         * string.
         *
         * In fact, the program may be directly built into
         * native source code where OpenCL API is used,
         * it is useful for short kernels (few lines) because it doesn't
         * involve loading code and you don't need to pass it from Java to
         * native side.
         */

        try
        {
            StringBuilder buffer = new StringBuilder();
            InputStream stream = getAssets().open("convolution.cl");
            BufferedReader reader = new BufferedReader(new InputStreamReader(stream));
            String s;

            while((s = reader.readLine()) != null)
            {
                buffer.append(s);
                buffer.append("\n");
            }

            reader.close();
            return buffer.toString();
        }
        catch (IOException e) {
            e.printStackTrace();
        }
        return "";

    }

    private String getOpenCLHeader(){
        try
        {
            StringBuilder buffer = new StringBuilder();
            InputStream stream = getAssets().open("header.h");
            BufferedReader reader = new BufferedReader(new InputStreamReader(stream));
            String s;

            while((s = reader.readLine()) != null)
            {
                buffer.append(s);
                buffer.append("\n");
            }

            reader.close();
            return buffer.toString();
        }
        catch (IOException e) {
            e.printStackTrace();
        }
        return "";
    }

    @Override
    protected void onStop() {
        super.onStop();
        finish();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        shutdownOpenCL();
    }

    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
    public native String convTest();

    public native void convTestNeon();

    public native String blasTest();

    public native void intiOpenCL(String openCLProgramText);

    public native void intiOpenCL2(String openCLProgramText, String openCLHeaderText);

    public native void shutdownOpenCL();

    native void testOpenCLConv();
}
