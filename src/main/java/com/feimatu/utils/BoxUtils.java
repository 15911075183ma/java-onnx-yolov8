package com.feimatu.utils;

import com.feimatu.entitys.BoundingBox;
import com.feimatu.entitys.BoxCorners;

/**
 * @author mazepeng
 * @date 2024/2/6 16:49
 */
public class BoxUtils {
    public static BoxCorners getLTRB(BoundingBox box) {
        double x1 = box.getXCenter() - box.getWidth() / 2;
        double y1 = box.getYCenter() - box.getHeight() / 2;
        double x2 = box.getXCenter() + box.getWidth() / 2;
        double y2 = box.getYCenter() + box.getHeight() / 2;
        return new BoxCorners(x1, y1, x2, y2);
    }
}



