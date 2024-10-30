package com.feimatu.utils;

import ai.onnxruntime.*;
import com.feimatu.entitys.BoundingBox;
import com.feimatu.entitys.KeyPoint;
import org.bytedeco.opencv.opencv_core.Mat;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * @author mazepeng
 * @date 2024/10/29 下午5:52
 */
public class KeypointPostProcessor {

    public static List<BoundingBox> getBoundingBoxes(OrtEnvironment environment, Mat mat, OrtSession session, Mat clone) throws OrtException {
        OnnxTensor onnxTensor = ImagePreprocessingUtils.img2OnnxTensor(environment, mat);
        Map<String, OnnxTensor> input = Map.of("images", onnxTensor);
        OrtSession.Result run = session.run(input);
        OnnxTensor onnxTensorResults = (OnnxTensor) run.get(0);


        float[][][] outputData = (float[][][]) onnxTensorResults.getValue();
        INDArray permute;
        List<BoundingBox> boundingBoxes;
        try (INDArray fromArray = Nd4j.createFromArray(outputData)) {
            permute = fromArray.permute(0, 2, 1);
            boundingBoxes = KeypointPostProcessor.processYoloPoseOutput(permute, 0.5, 0.8, clone);
        }
        return boundingBoxes;
    }

    private static List<BoundingBox> processYoloPoseOutput(INDArray transposedArray, double confThreshold, double iouThreshold, Mat mat) {
        List<BoundingBox> boundingBoxes = new ArrayList<>();
        int numBoxes = (int) transposedArray.shape()[1];
        int numAttributes = (int) transposedArray.shape()[2];
        double wide = mat.cols() / 640.0;
        double high = mat.rows() / 640.0;
        for (int i = 0; i < numBoxes; i++) {
            double confidence = transposedArray.getDouble(0, i, 4);
            if (confidence < confThreshold) {
                continue;
            }
            BoundingBox boundingBox = new BoundingBox();
            // 提取边界框
            double cx = transposedArray.getDouble(0, i, 0);
            double cy = transposedArray.getDouble(0, i, 1);
            double w = transposedArray.getDouble(0, i, 2);
            double h = transposedArray.getDouble(0, i, 3);
            double xMin = (cx - w / 2) * wide;
            double yMin = (cy - h / 2) * high;
            double xMax = (w) * wide;
            double yMax = (h) * high;
            boundingBox.setXCenter(xMin);
            boundingBox.setYCenter(yMin);
            boundingBox.setWidth(xMax);
            boundingBox.setHeight(yMax);
            boundingBox.setScore(confidence);

            // 提取关键点
            ArrayList<KeyPoint> keyPoints = new ArrayList<>();
            boundingBox.setKeyPoints(keyPoints);
            for (int j = 5; j < numAttributes; j += 3) {
                double keypointX = transposedArray.getDouble(0, i, j);
                double keypointY = transposedArray.getDouble(0, i, j + 1);
                double keypointConf = transposedArray.getDouble(0, i, j + 2);

                KeyPoint keypoint = new KeyPoint(keypointX * wide, keypointY * high, keypointConf);
                keyPoints.add(keypoint);
            }
            boundingBoxes.add(boundingBox);
        }
        return NMSUtils.nmsWithCategories(boundingBoxes, iouThreshold);
    }
}
