package com.feimatu.utils;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import org.apache.commons.collections4.Put;
import org.bytedeco.opencv.global.opencv_imgcodecs;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.nio.Buffer;
import java.nio.FloatBuffer;

import static org.bytedeco.opencv.global.opencv_core.CV_32FC3;

/**
 * @author mazepeng
 * @date 2024/2/6 16:59
 */
public class ImagePreprocessingUtils {


    /**
     * 将图片转换到OnnxTensor
     * @param environment 引擎
     * @param mat 图片
     * @return OnnxTensor
     * @throws OrtException 格式不匹配等相关报错
     */
    public static OnnxTensor img2OnnxTensor(OrtEnvironment environment, Mat mat) throws OrtException {
        BGR2RGB(mat);
        Mat mat1 = scaleImageInEqualProportions(640, 640, mat);
        normalization(mat1);
        FloatBuffer floatBuffer = convertHWCtoCHW(mat1);
        return OnnxTensor.createTensor(environment, floatBuffer, new long[]{1, 3, 640, 640});
    }

    /**
     * 读取图片
     *
     * @param file 文件地址
     * @return Mat图片类
     */
    public static Mat readPicture(String file) {
        return opencv_imgcodecs.imread(file);
    }

    /**
     * 将图片通道从BGR转换到RGB 直接改变源文件
     *
     * @param mat 图片文件
     */
    public static void BGR2RGB(Mat mat) {
        opencv_imgproc.cvtColor(mat, mat, opencv_imgproc.COLOR_BGR2RGB);
    }

    /**
     * 将图片进行等比例缩放
     *
     * @param width  缩放后的宽
     * @param height 缩放后的高
     * @param mat    图片
     * @return 等比例缩放后的图片
     */
    public static Mat scaleImageInEqualProportions(int width, int height, Mat mat) {
        opencv_imgproc.resize(mat, mat, new Size(width, height));
        int x = 0, y = 0;
        if (mat.cols() == 640) {
            y = (mat.cols() - mat.rows()) / 2;
        } else {
            x = (mat.rows() - mat.cols()) / 2;
        }
        Mat im = new Mat(width, height, mat.type(), new Scalar(0, 0, 0, 0));
        Mat apply = im.apply(new Rect(x, y, mat.cols(), mat.rows()));
        mat.copyTo(apply);
        return im;
    }

    /**
     * 归一化数据
     *
     * @param mat 图片文件
     */
    public static void normalization(Mat mat) {
        //将图片类型转换为浮点类型
        mat.convertTo(mat, CV_32FC3);
        //将数据归一化到0-1之间
        mat.convertTo(mat, CV_32FC3, 1.0 / 255.0, 0);
    }

    public static FloatBuffer convertHWCtoCHW(Mat mat) {
        FloatBuffer buffer = mat.createBuffer();
        // 获取FloatBuffer中的剩余元素数量
        int remainingElements = buffer.remaining();

        // 创建一个与剩余元素数量相等的float[]数组
        float[] floatArray = new float[remainingElements];

        buffer.get(floatArray);
        buffer.clear();
        FloatBuffer nioFloat;
        try (INDArray indArray = Nd4j.create(floatArray, new long[]{640, 640, 3})) {
            //交换维度，并将数据从多维展平，order 设置为t才能正确处理
            nioFloat = indArray.permute(2, 0, 1).reshape('t', 1, -1).data().asNioFloat();
        }
        return nioFloat;
    }
}
