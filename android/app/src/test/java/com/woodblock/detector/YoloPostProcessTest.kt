package com.woodblock.detector

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.abs

/**
 * 在桌面 JVM 上验证 Android 端的推理链路（ONNX 输出 -> 解码 -> 坐标反变换 -> NMS）。
 *
 * Android 的单测环境没有 java.awt，无法解码图片，因此测试资源由
 * `tools/make_android_test_assets.py` 预先算好：
 *   * letterbox_input.bin —— block.jpg(1080x2376) 经 letterbox 后的 640x640 RGB uint8
 *   * letterbox_meta.json —— 参数与桌面端（Python + onnxruntime）的基准检测结果
 *
 * 基准：conf>=0.35 时恰好 1 个框，置信度 0.8302，框 ≈ (-13.8,209.2,1087.4,1586.4)；
 * Android 端会把框裁剪到图片内，实测应为 (0,209.2,1080,1586.4)。
 *
 * 覆盖的高风险点：模型输出布局 (1,5,8400)、解码公式、letterbox 反变换、
 * 形态过滤（长宽比/面积）与 NMS。
 */
class YoloPostProcessTest {

    private val testRes = File("src/test/resources")
    private val modelFile = File("src/main/assets/wood_block_yolo11s.onnx")
    private val inputBin = File(testRes, "letterbox_input.bin")

    private val expectedConf = 0.8302f
    private val expectedBox = floatArrayOf(0f, 209.2f, 1080f, 1586.4f)

    @Test
    fun `letterbox 参数与桌面端一致`() {
        val lb = YoloPostProcess.letterboxOf(1080, 2376, 640)
        assertEquals("scale", 0.269360f, lb.scale, 1e-4f)
        assertEquals("padX", 174f, lb.padX, 1f)
        assertEquals("padY", 0f, lb.padY, 1f)
    }

    @Test
    fun `输出布局符合单类别 YOLO11 检测头`() {
        val raw = runOnnx(readInput())
        assertEquals("应为 (1, 4+1, 8400) 即 5 行", 5, raw.size)
        assertEquals("每行 8400 个候选框", 8400, raw[0].size)
    }

    @Test
    fun `block_jpg 上的端到端结果与桌面端一致`() {
        assertTrue("缺少模型: ${modelFile.absolutePath}", modelFile.exists())
        assertTrue("缺少测试输入: ${inputBin.absolutePath}（请先运行 tools/make_android_test_assets.py）",
            inputBin.exists())

        val lb = YoloPostProcess.letterboxOf(1080, 2376, 640)
        val raw = runOnnx(readInput())
        val dets = YoloPostProcess.postProcess(
            raw, lb, srcW = 1080, srcH = 2376, confThreshold = 0.35f, iouThreshold = 0.45f
        )
        println("检测结果: " + dets.joinToString { "${it.conf}(${it.x1},${it.y1},${it.x2},${it.y2})" })

        assertEquals("conf>=0.35 时应当只有 1 个木块框", 1, dets.size)
        val d = dets[0]
        assertTrue("置信度应接近桌面端 $expectedConf，实际 ${d.conf}", abs(d.conf - expectedConf) < 0.05f)
        val ref = Detection(expectedBox[0], expectedBox[1], expectedBox[2], expectedBox[3], d.conf)
        val iou = YoloPostProcess.iou(d, ref)
        assertTrue("框应与桌面端高度重合（IoU=$iou）", iou > 0.9f)
    }

    @Test
    fun `阈值提高到 095 后不再输出框`() {
        val lb = YoloPostProcess.letterboxOf(1080, 2376, 640)
        val raw = runOnnx(readInput())
        val dets = YoloPostProcess.postProcess(
            raw, lb, srcW = 1080, srcH = 2376, confThreshold = 0.95f, iouThreshold = 0.45f
        )
        assertEquals("极高阈值下应无检测框", 0, dets.size)
    }

    @Test
    fun `形态过滤会丢弃极端狭长的框`() {
        val lb = Letterbox(1f, 0f, 0f, 640, 640)
        // 构造 (1,5,2)：第一个是正常方块，第二个是 500x10 的狭长框
        val raw = arrayOf(
            floatArrayOf(320f, 320f),
            floatArrayOf(320f, 320f),
            floatArrayOf(100f, 500f),
            floatArrayOf(100f, 10f),
            floatArrayOf(0.9f, 0.9f),
        )
        val dets = YoloPostProcess.postProcess(raw, lb, 640, 640, 0.35f, 0.45f)
        assertEquals("狭长框应被形态过滤丢弃", 1, dets.size)
        assertTrue("保留的应是正常的那个框", dets[0].width < 200f)
    }

    // ---------------------------------------------------------------- 工具
    private fun readInput(): FloatBuffer {
        val bytes = inputBin.readBytes()
        val px = 640 * 640
        require(bytes.size == px * 3) { "输入文件大小异常: ${bytes.size}" }
        val buf = FloatBuffer.allocate(px * 3)
        for (i in 0 until px) {
            buf.put(i, (bytes[i * 3].toInt() and 0xFF) / 255f)              // R
            buf.put(px + i, (bytes[i * 3 + 1].toInt() and 0xFF) / 255f)     // G
            buf.put(2 * px + i, (bytes[i * 3 + 2].toInt() and 0xFF) / 255f) // B
        }
        return buf
    }

    private fun runOnnx(input: FloatBuffer): Array<FloatArray> {
        val env = OrtEnvironment.getEnvironment()
        env.createSession(modelFile.absolutePath, OrtSession.SessionOptions()).use { session ->
            OnnxTensor.createTensor(env, input, longArrayOf(1, 3, 640, 640)).use { tensor ->
                session.run(mapOf(session.inputNames.first() to tensor)).use { out ->
                    val value = out[0].value
                    check(value is Array<*>) { "未预期的输出类型: ${value.javaClass}" }
                    @Suppress("UNCHECKED_CAST")
                    return (value[0] as Array<*>).map { (it as FloatArray).copyOf() }.toTypedArray()
                }
            }
        }
    }
}
