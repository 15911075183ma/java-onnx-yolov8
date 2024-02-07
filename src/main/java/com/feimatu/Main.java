package com.feimatu;

import ai.onnxruntime.*;
import com.feimatu.entitys.BoundingBox;
import com.feimatu.utils.ImagePreprocessingUtils;
import com.feimatu.utils.ImageUtils;
import com.feimatu.utils.PostProcessingOfPredictionResults;
import lombok.extern.slf4j.Slf4j;
import org.bytedeco.opencv.opencv_core.Mat;

import java.util.List;
import java.util.Map;

import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imwrite;


/**
 * @author mazepeng
 * @date 2024/1/11 10:20
 */
@Slf4j
public class Main {

    public static void main(String[] args) throws OrtException {
        OrtEnvironment environment = OrtEnvironment.getEnvironment();
        OrtSession session = environment.createSession("src/main/resources/yolov8n.onnx");
        PostProcessingOfPredictionResults.updateTheDictionary(session.getMetadata().getCustomMetadata().get("names"));
        try (Mat mat = ImagePreprocessingUtils.readPicture("src/main/resources/bus.jpg")) {
            Mat clone = mat.clone();
            OnnxTensor onnxTensor = ImagePreprocessingUtils.img2OnnxTensor(environment, mat);
            Map<String, OnnxTensor> input = Map.of("images", onnxTensor);
            OrtSession.Result run = session.run(input);
            List<BoundingBox> boundingBoxes = PostProcessingOfPredictionResults.interpretTheReasoningResults(run, clone);
            ImageUtils.pictureFrame(clone,boundingBoxes,"src/main/resources/bus1.jpg");
            clone.close();
        }


    }

}