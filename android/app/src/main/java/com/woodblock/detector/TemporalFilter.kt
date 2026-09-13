package com.woodblock.detector

import kotlin.math.max
import kotlin.math.min

/**
 * N-of-M 时序确认 + 框平滑（与桌面端 detect_camera.py 的 TemporalFilter 等价）。
 *
 * 只有当同一位置在最近 window 帧中出现至少 minHits 次时才输出，
 * 用来抑制偶发误检，同时让画面里的框不抖动。
 */
class TemporalFilter(
    private val minHits: Int = 3,
    private val window: Int = 8,
    private val iouThreshold: Float = 0.3f,
    private val smoothing: Float = 0.6f,
) {
    private class Track(var det: Detection, var hits: Int, var missed: Int)

    private val tracks = ArrayList<Track>()

    fun update(detections: List<Detection>): List<Detection> {
        val matched = HashSet<Int>()

        for (tr in tracks) {
            var bestIdx = -1
            var bestIou = 0f
            for ((i, d) in detections.withIndex()) {
                if (i in matched) continue
                val v = iou(tr.det, d)
                if (v > bestIou) {
                    bestIou = v
                    bestIdx = i
                }
            }
            if (bestIdx >= 0 && bestIou >= iouThreshold) {
                matched.add(bestIdx)
                val d = detections[bestIdx]
                val a = smoothing
                tr.det = Detection(
                    a * tr.det.x1 + (1 - a) * d.x1,
                    a * tr.det.y1 + (1 - a) * d.y1,
                    a * tr.det.x2 + (1 - a) * d.x2,
                    a * tr.det.y2 + (1 - a) * d.y2,
                    max(tr.det.conf, d.conf),
                )
                tr.hits++
                tr.missed = 0
            } else {
                tr.missed++
            }
        }

        for ((i, d) in detections.withIndex()) {
            if (i !in matched) tracks.add(Track(d, 1, 0))
        }

        tracks.removeAll { it.missed > window }
        return tracks.filter { it.hits >= minHits }.map { it.det }
    }

    fun reset() = tracks.clear()

    private fun iou(a: Detection, b: Detection): Float {
        val ix1 = max(a.x1, b.x1)
        val iy1 = max(a.y1, b.y1)
        val ix2 = min(a.x2, b.x2)
        val iy2 = min(a.y2, b.y2)
        val inter = max(0f, ix2 - ix1) * max(0f, iy2 - iy1)
        val union = a.area + b.area - inter
        return if (union > 0f) inter / union else 0f
    }
}
