package com.feimatu;

import ai.onnxruntime.*;
import com.feimatu.entitys.BoundingBox;
import com.feimatu.utils.*;
import org.bytedeco.opencv.opencv_core.Mat;
import java.util.List;

/**
 * @author mazepeng
 * @date 2024/10/29 下午4:52
 */
public class yolo11_pose {

    public static void main(String[] args) throws OrtException {

        try (OrtEnvironment environment = OrtEnvironment.getEnvironment();
             OrtSession session = environment.createSession("src/main/resources/yolo11x-pose.onnx");
             Mat mat = ImagePreprocessingUtils.readPicture("src/main/resources/img.png")
        ) {

            Mat clone = mat.clone();
            List<BoundingBox> boundingBoxes = KeypointPostProcessor.getBoundingBoxes(environment, mat, session, clone);
            ImageUtils.pictureFrame(clone, boundingBoxes, "src/main/resources/img1.png");
        }
    }


}
