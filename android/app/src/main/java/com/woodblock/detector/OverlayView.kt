package com.woodblock.detector

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import kotlin.math.max
import kotlin.math.min

/**
 * 把检测框画在预览之上。
 *
 * 坐标换算：检测结果在"已旋转的分析帧"坐标系里，这里按 FILL_CENTER 的方式
 * 等比缩放并居中映射到控件尺寸，保证框与画面里的木块对齐。
 */
class OverlayView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null,
) : View(context, attrs) {

    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 6f
        color = Color.rgb(0, 220, 0)
    }
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.rgb(0, 255, 120)
        textSize = 42f
        isFakeBoldText = true
    }
    private val textBg = Paint().apply { color = Color.argb(160, 0, 0, 0) }

    private var detections: List<Detection> = emptyList()
    private var srcW = 0
    private var srcH = 0
    private var mirrored = false

    fun setResult(dets: List<Detection>, frameW: Int, frameH: Int, mirror: Boolean) {
        detections = dets
        srcW = frameW
        srcH = frameH
        mirrored = mirror
        postInvalidateOnAnimation()
    }

    fun clear() {
        detections = emptyList()
        postInvalidateOnAnimation()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (srcW <= 0 || srcH <= 0 || detections.isEmpty()) return

        val scale = max(width.toFloat() / srcW, height.toFloat() / srcH)
        val offX = (width - srcW * scale) / 2f
        val offY = (height - srcH * scale) / 2f

        for (d in detections) {
            var x1 = d.x1 * scale + offX
            var x2 = d.x2 * scale + offX
            if (mirrored) {                       // 前置摄像头预览是镜像的
                val a = width - x1
                val b = width - x2
                x1 = min(a, b)
                x2 = max(a, b)
            }
            val y1 = d.y1 * scale + offY
            val y2 = d.y2 * scale + offY
            canvas.drawRect(RectF(x1, y1, x2, y2), boxPaint)

            val label = String.format("wood_block %.2f", d.conf)
            val tw = textPaint.measureText(label)
            val ty = if (y1 > 60f) y1 - 12f else y2 + 46f
            canvas.drawRect(x1, ty - 40f, x1 + tw + 16f, ty + 8f, textBg)
            canvas.drawText(label, x1 + 8f, ty, textPaint)
        }
    }
}
