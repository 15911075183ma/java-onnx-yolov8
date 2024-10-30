package com.feimatu.utils;

import com.feimatu.entitys.BoundingBox;
import com.feimatu.entitys.KeyPoint;
import org.bytedeco.opencv.global.opencv_imgcodecs;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Scalar;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

/**
 * @author mazepeng
 * @date 2024/2/6 17:30
 */
public class ImageUtils {
    private static final double CONFIDENCE_THRESHOLD = 0.7; // 设置置信度阈值


    public static void pictureFrame(Mat mat, List<BoundingBox> boundingBoxes, String outFile) {
        for (BoundingBox boundingBox : boundingBoxes) {
            // 计算矩形的左上角和右下角坐标
            try (Point topLeft = new Point((int) (boundingBox.getXCenter()), (int) (boundingBox.getYCenter()));
                 Point bottomRight = new Point((int) (boundingBox.getXCenter() + boundingBox.getWidth()), (int) (boundingBox.getYCenter() + boundingBox.getHeight()));
                 Point org = new Point(topLeft.x(), topLeft.y() - 10);
            ) {
                opencv_imgproc.rectangle(mat, topLeft, bottomRight, getHighContrastRandomColor(), 1, opencv_imgproc.LINE_8, 0);
                // 文字起始点坐标
                BigDecimal bigDecimal = BigDecimal.valueOf(boundingBox.getScore());
                bigDecimal = bigDecimal.setScale(2, RoundingMode.HALF_UP);
                opencv_imgproc.putText(mat, boundingBox.getCategoryName()+": " + bigDecimal.doubleValue(), org, opencv_imgproc.FONT_HERSHEY_TRIPLEX, 0.5, new Scalar(255, 0, 0, 1));
                List<KeyPoint> keyPoints = boundingBox.getKeyPoints();
                if (keyPoints !=null && !keyPoints.isEmpty()) {
                    for (KeyPoint keypoint : keyPoints) {
                        if (keypoint.getKeypointConf() >= CONFIDENCE_THRESHOLD) {
                            opencv_imgproc.circle(mat, new Point((int) keypoint.getKeypointX(), (int) keypoint.getKeypointY()), 5, getHighContrastRandomColor(), -1, opencv_imgproc.LINE_AA, 0);
                        }
                    }
                    drawSkeleton(mat, keyPoints);
                }

            }
        }
        opencv_imgcodecs.imwrite(outFile, mat);
    }

    // 生成高对比度随机颜色的方法
    public static Scalar getHighContrastRandomColor() {
        // 随机生成较大范围的色相值（0-360）
        float hue = ThreadLocalRandom.current().nextFloat() * 360;
        float saturation = 0.9f; // 设置高饱和度
        float value = 0.9f;      // 设置高亮度

        // 将 HSV 转换为 RGB
        return hsvToBgr(hue, saturation, value);
    }

    // 将 HSV 转换为 OpenCV BGR 格式的 Scalar
    public static Scalar hsvToBgr(float hue, float saturation, float value) {
        int h = (int)(hue / 60) % 6;
        float f = (hue / 60) - h;
        float p = value * (1 - saturation);
        float q = value * (1 - f * saturation);
        float t = value * (1 - (1 - f) * saturation);

        float r = 0, g = 0, b = 0;
        switch (h) {
            case 0 -> { r = value; g = t; b = p; }
            case 1 -> { r = q; g = value; b = p; }
            case 2 -> { r = p; g = value; b = t; }
            case 3 -> { r = p; g = q; b = value; }
            case 4 -> { r = t; g = p; b = value; }
            case 5 -> { r = value; g = p; b = q; }
        }

        return new Scalar(b * 255, g * 255, r * 255, 1);
    }


    // 绘制骨架连接线
    public static void drawSkeleton(Mat mat, List<KeyPoint> keyPoints) {
        // 定义连接的骨架对
        int[][] skeleton = {
                {0, 1}, {0, 2}, {1, 3}, {2, 4},   // 头部和脸部
                {5, 6}, {5, 7}, {6, 8},           // 肩膀和手臂上段
                {7, 9}, {8, 10},                  // 手臂下段
                {5, 11}, {6, 12},                 // 肩膀到臀部
                {11, 12},                         // 腰部连接
                {11, 13}, {12, 14},               // 腰到膝盖
                {13, 15}, {14, 16}                // 膝盖到脚踝
        };

        // 遍历每对关键点并绘制连线
        for (int[] pair : skeleton) {
            KeyPoint pt1 = keyPoints.get(pair[0]);
            KeyPoint pt2 = keyPoints.get(pair[1]);

            // 仅当两个点的置信度都高于阈值时才绘制连线
            if (pt1.getKeypointConf() >= CONFIDENCE_THRESHOLD && pt2.getKeypointConf() >= CONFIDENCE_THRESHOLD) {
                opencv_imgproc.line(mat,
                        new Point((int) pt1.getKeypointX(), (int) pt1.getKeypointY()),
                        new Point((int) pt2.getKeypointX(), (int) pt2.getKeypointY()),
                        new Scalar(0, 255, 0, 1), 2, opencv_imgproc.LINE_AA, 0
                );
            }
        }
    }
}
