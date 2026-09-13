package com.woodblock.detector

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import java.nio.FloatBuffer

/**
 * 木块检测器：在手机上用 ONNX Runtime 跑与桌面端同一个模型。
 *
 * 与桌面端保持一致的三重把关（满足"不框到其他物体"）：
 *  1) 单类别模型：网络只输出 wood_block 一类；
 *  2) 置信度阈值：默认 0.35（在 BOP 官方负样本上标定，误检率 0.85%）；
 *  3) 时序确认：连续多帧命中同一位置才画框（见 TemporalFilter）。
 *
 * 后处理（解码 + 过滤 + NMS）放在纯 Kotlin 的 [YoloPostProcess] 里，
 * 这样可以脱离 Android 设备用 JVM 单测验证结果与桌面端一致。
 */
class WoodBlockDetector(
    context: Context,
    modelAsset: String = "wood_block_yolo11s.onnx",
    var confThreshold: Float = 0.35f,
    var iouThreshold: Float = 0.45f,
    val inputSize: Int = 640,
    private val maxDetections: Int = 20,
) {
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val inputName: String

    /** 预处理用的画布，复用避免每帧分配。 */
    private val canvas = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888)
    private val canvasPaint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val inputBuffer = FloatBuffer.allocate(3 * inputSize * inputSize)
    private val pixels = IntArray(inputSize * inputSize)

    init {
        val bytes = context.assets.open(modelAsset).use { it.readBytes() }
        val options = OrtSession.SessionOptions().apply {
            // 手机端限制线程数，兼顾延迟与发热（取可用核数的一半，至少 2）
            setIntraOpNumThreads(maxOf(2, Runtime.getRuntime().availableProcessors() / 2))
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        session = env.createSession(bytes, options)
        inputName = session.inputNames.first()
    }

    fun letterboxOf(srcW: Int, srcH: Int): Letterbox =
        YoloPostProcess.letterboxOf(srcW, srcH, inputSize)

    /** 对一张 Bitmap 推理，返回原图坐标下的框。 */
    fun detect(bitmap: Bitmap): List<Detection> {
        val lb = letterboxOf(bitmap.width, bitmap.height)

        // letterbox 到 640x640，空白填 114（与训练/导出时一致）
        val c = Canvas(canvas)
        c.drawColor(Color.rgb(114, 114, 114))
        c.drawBitmap(bitmap, null, RectF(lb.padX, lb.padY, lb.padX + lb.w, lb.padY + lb.h), canvasPaint)

        // 转成 NCHW float32（RGB，0~1）
        canvas.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)
        val ch = inputSize * inputSize
        for (i in pixels.indices) {
            val p = pixels[i]
            inputBuffer.put(i, ((p shr 16) and 0xFF) / 255f)
            inputBuffer.put(ch + i, ((p shr 8) and 0xFF) / 255f)
            inputBuffer.put(2 * ch + i, (p and 0xFF) / 255f)
        }

        val tensor = OnnxTensor.createTensor(
            env, inputBuffer, longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())
        )
        val raw = tensor.use { t ->
            session.run(mapOf(inputName to t)).use { out ->
                when (val value = out[0].value) {
                    is Array<*> -> (value[0] as Array<*>).map { (it as FloatArray).copyOf() }.toTypedArray()
                    else -> throw IllegalStateException("未预期的模型输出类型: ${value.javaClass}")
                }
            }
        }
        return YoloPostProcess.postProcess(
            raw, lb, bitmap.width, bitmap.height, confThreshold, iouThreshold, maxDetections
        )
    }

    fun close() {
        session.close()
    }
}
