package com.feimatu.utils;

import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import com.feimatu.entitys.BoundingBox;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import org.bytedeco.opencv.opencv_core.Mat;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * @author mazepeng
 * @date 2024/2/6 17:25
 */
public class PostProcessingOfPredictionResults {
    private static final Map<Integer,String> typeDictionary = new HashMap<>();


    public static void updateTheDictionary(String names){
        Gson gson = new Gson();
        Type type = TypeToken.getParameterized(Map.class, Integer.class, String.class).getType();
        Map<Integer,String> o = gson.fromJson(names, type);
        typeDictionary.putAll(o);
    }


    /**
     * 解析推理结果
     *
     * @param result 推理结果
     * @param mat    原始图像
     * @return 结果集合
     * @throws OrtException
     */
    public static List<BoundingBox> interpretTheReasoningResults(OrtSession.Result result, Mat mat) throws OrtException {
        float[][][] value = (float[][][]) result.get(0).getValue();
        INDArray indArray1 = Nd4j.create(value[0]);
        INDArray transpose = indArray1.transpose();
        INDArray indArray2 = transpose.get(NDArrayIndex.all(), NDArrayIndex.interval(4, 84));
        List<BoundingBox> boundingBoxes = new ArrayList<>();
        double wide = mat.cols() / 640.0;
        double high = mat.rows() / 640.0;
        for (int i = 0; i < indArray2.rows(); i++) {
            INDArray indArray3 = indArray2.getRow(i).argMax();
            int anInt = indArray3.getInt();
            indArray3.close();
            double aDouble = indArray2.getDouble(i, anInt);
            if (aDouble > 0.5) {
                double x1 = transpose.getDouble(i, 0);
                double y1 = transpose.getDouble(i, 1);
                double w = transpose.getDouble(i, 2);
                double h = transpose.getDouble(i, 3);
                int left = (int) ((x1 - w / 2) * wide);
                int top = (int) ((y1 - h / 2) * high);
                int width = (int) (w * wide);
                int height = (int) (h * high);
                BoundingBox boundingBox = new BoundingBox(left, top, width, height, aDouble, anInt,typeDictionary.getOrDefault(anInt,"unknown"));
                boundingBoxes.add(boundingBox);
            }
        }

        return NMSUtils.nmsWithCategories(boundingBoxes, 0.5);
    }
}
