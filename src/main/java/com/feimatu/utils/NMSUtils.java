package com.feimatu.utils;

import com.feimatu.entitys.BoundingBox;
import com.feimatu.entitys.BoxCorners;

import java.util.*;

/**
 * @author mazepeng
 * @date 2024/2/6 16:43
 * 应用非极大值抑制以过滤掉重叠的边界框
 */
public class NMSUtils {

    public static List<BoundingBox> nmsWithCategories(List<BoundingBox> boxes, double threshold) {
        List<BoundingBox> result = new ArrayList<>();
        Map<Integer, List<BoundingBox>> boxesByCategory = new HashMap<>();

        // 将边界框按照类别分组
        for (BoundingBox box : boxes) {
            boxesByCategory.computeIfAbsent(box.getCategoryId(), k -> new ArrayList<>()).add(box);
        }

        // 对每个类别执行NMS
        for (List<BoundingBox> categoryBoxes : boxesByCategory.values()) {
            // 按照置信度降序排序边界框
            categoryBoxes.sort(Comparator.comparingDouble(BoundingBox::getScore).reversed());

            while (!categoryBoxes.isEmpty()) {
                // 选择置信度最高的边界框
                BoundingBox topBox = categoryBoxes.get(0);
                result.add(topBox);

                // 计算与其他边界框的IoU并移除重叠较多的边界框
                categoryBoxes.removeIf(box -> calculateIoU(topBox, box) > threshold);
            }
        }

        return result;
    }

    private static double calculateIoU(BoundingBox box1, BoundingBox box2) {
        // 转换为左上角和右下角坐标
        BoxCorners ltrb1 = BoxUtils.getLTRB(box1);
        BoxCorners ltrb2 = BoxUtils.getLTRB(box2);

        // 计算交集部分的左上角和右下角坐标
        double x_inter_left = Math.max(ltrb1.getX1(), ltrb2.getX1());
        double y_inter_top = Math.max(ltrb1.getY1(), ltrb2.getY1());
        double x_inter_right = Math.min(ltrb1.getX2(), ltrb2.getX2());
        double y_inter_bottom = Math.min(ltrb1.getY2(), ltrb2.getY2());

        // 计算并集与交集面积
        double intersectionArea = (x_inter_right - x_inter_left) * (y_inter_bottom - y_inter_top);
        double areaA = getArea(ltrb1);
        double areaB = getArea(ltrb2);
        double unionArea = areaA + areaB - intersectionArea;

        // 计算并处理边界情况
        return intersectionArea / (unionArea > 0 ? unionArea : 1);
    }

    private static double getArea(BoxCorners boxLtrb) {
        return (boxLtrb.getX2() - boxLtrb.getX1()) * (boxLtrb.getY2() - boxLtrb.getY1());
    }


}
