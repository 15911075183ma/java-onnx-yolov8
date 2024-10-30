package com.feimatu.entitys;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

/**
 * @author mazepeng
 * @date 2024/2/6 15:04
 *
 */
@Data
@AllArgsConstructor
@NoArgsConstructor
public class BoundingBox {
    private double xCenter;
    private double yCenter;
    private double width;
    private double height;
    private double score;
    private int categoryId;
    private String categoryName;
    List<KeyPoint> keyPoints;
}
