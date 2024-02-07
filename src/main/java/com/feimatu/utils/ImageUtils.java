package com.feimatu.utils;

import com.feimatu.entitys.BoundingBox;
import org.bytedeco.opencv.global.opencv_imgcodecs;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Scalar;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.List;

/**
 * @author mazepeng
 * @date 2024/2/6 17:30
 */
public class ImageUtils {
    public static void pictureFrame(Mat mat, List<BoundingBox> boundingBoxes, String outFile) {
        for (BoundingBox boundingBox : boundingBoxes) {
            // 计算矩形的左上角和右下角坐标
            try (Point topLeft = new Point((int) (boundingBox.getXCenter()), (int) (boundingBox.getYCenter()));
                 Point bottomRight = new Point((int) (boundingBox.getXCenter() + boundingBox.getWidth()), (int) (boundingBox.getYCenter() + boundingBox.getHeight()));
                 Point org = new Point(topLeft.x(), topLeft.y() - 10);
            ) {
                opencv_imgproc.rectangle(mat, topLeft, bottomRight, new Scalar(255, 0, 0, 1), 1, opencv_imgproc.LINE_8, 0);
                // 文字起始点坐标
                BigDecimal bigDecimal = BigDecimal.valueOf(boundingBox.getScore());
                bigDecimal = bigDecimal.setScale(2, RoundingMode.HALF_UP);
                opencv_imgproc.putText(mat, boundingBox.getCategoryName()+": " + bigDecimal.doubleValue(), org, opencv_imgproc.FONT_HERSHEY_TRIPLEX, 0.5, new Scalar(255, 0, 0, 1));
            }
        }
        opencv_imgcodecs.imwrite(outFile, mat);


    }
}
