package com.feimatu.entitys;

import lombok.AllArgsConstructor;
import lombok.Data;

/**
 * @author mazepeng
 * @date 2024/10/30 上午11:14
 */
@Data
@AllArgsConstructor
public class KeyPoint {
    private double keypointX;
    private double keypointY;
    private double keypointConf;

}
