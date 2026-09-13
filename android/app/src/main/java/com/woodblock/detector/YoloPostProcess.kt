package com.woodblock.detector

import kotlin.math.max
import kotlin.math.min

/** 一个木块检测框（图片坐标）。不依赖 Android API，便于在 JVM 上单测。 */
data class Detection(val x1: Float, val y1: Float, val x2: Float, val y2: Float, val conf: Float) {
    val width: Float get() = x2 - x1
    val height: Float get() = y2 - y1
    val area: Float get() = max(0f, width) * max(0f, height)
}

/** letterbox 参数：原图 -> 模型输入（640×640）的等比缩放与居中留白。 */
data class Letterbox(val scale: Float, val padX: Float, val padY: Float, val w: Int, val h: Int)

/**
 * YOLO11 检测头的后处理（纯 Kotlin，与桌面端 Python 实现一一对应）。
 *
 * 模型输出形状为 (1, 4+nc, 8400)；本项目 nc=1，即 (1, 5, 8400)：
 * 每列是 [cx, cy, w, h, score]（坐标在 640×640 的 letterbox 图上）。
 *
 * 这里做四件事：
 *  1) 取类别分数（单类别）并按置信度阈值筛选；
 *  2) letterbox 反变换回原图坐标，并裁剪到图片范围内；
 *  3) 形态过滤：丢弃极端狭长（长宽比 > 3.5）和过小（面积占比 < 2e-4）的框；
 *  4) 按 IoU 做 NMS。
 * 与桌面端 `woodblock/detector.py` 的 `_accept` + `evaluate._iou` 规则保持一致。
 */
object YoloPostProcess {

    const val MAX_ASPECT = 3.5f
    const val MIN_AREA_RATIO = 2e-4f

    fun letterboxOf(srcW: Int, srcH: Int, inputSize: Int): Letterbox {
        val scale = min(inputSize.toFloat() / srcW, inputSize.toFloat() / srcH)
        val newW = (srcW * scale).toInt()
        val newH = (srcH * scale).toInt()
        return Letterbox(scale, (inputSize - newW) / 2f, (inputSize - newH) / 2f, newW, newH)
    }

    fun iou(a: Detection, b: Detection): Float {
        val ix1 = max(a.x1, b.x1)
        val iy1 = max(a.y1, b.y1)
        val ix2 = min(a.x2, b.x2)
        val iy2 = min(a.y2, b.y2)
        val inter = max(0f, ix2 - ix1) * max(0f, iy2 - iy1)
        val union = a.area + b.area - inter
        return if (union > 0f) inter / union else 0f
    }

    fun postProcess(
        raw: Array<FloatArray>,
        lb: Letterbox,
        srcW: Int,
        srcH: Int,
        confThreshold: Float,
        iouThreshold: Float,
        maxDetections: Int = 20,
    ): List<Detection> {
        val rows = raw.size          // 4 + nc
        val n = raw[0].size          // 8400
        val boxes = ArrayList<Detection>(64)
        for (i in 0 until n) {
            var score = 0f
            for (r in 4 until rows) score = max(score, raw[r][i])
            if (score < confThreshold) continue

            val cx = raw[0][i]
            val cy = raw[1][i]
            val w = raw[2][i]
            val h = raw[3][i]
            val x1 = ((cx - w / 2f - lb.padX) / lb.scale).coerceIn(0f, srcW.toFloat())
            val y1 = ((cy - h / 2f - lb.padY) / lb.scale).coerceIn(0f, srcH.toFloat())
            val x2 = ((cx + w / 2f - lb.padX) / lb.scale).coerceIn(0f, srcW.toFloat())
            val y2 = ((cy + h / 2f - lb.padY) / lb.scale).coerceIn(0f, srcH.toFloat())
            val det = Detection(x1, y1, x2, y2, score)
            if (det.width <= 1f || det.height <= 1f) continue
            if (max(det.width / det.height, det.height / det.width) > MAX_ASPECT) continue
            if (det.area / (srcW.toFloat() * srcH) < MIN_AREA_RATIO) continue
            boxes.add(det)
        }
        return nms(boxes, iouThreshold, maxDetections)
    }

    fun nms(boxes: List<Detection>, iouThreshold: Float, maxDetections: Int = 20): List<Detection> {
        val sorted = boxes.sortedByDescending { it.conf }
        val keep = ArrayList<Detection>(sorted.size)
        for (d in sorted) {
            if (keep.all { iou(it, d) < iouThreshold }) keep.add(d)
            if (keep.size >= maxDetections) break
        }
        return keep
    }
}
