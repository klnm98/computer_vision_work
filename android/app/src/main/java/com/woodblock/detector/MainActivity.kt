package com.woodblock.detector

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.SeekBar
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.woodblock.detector.databinding.ActivityMainBinding
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.max

/**
 * 木块识别（手机端）。
 *
 * 功能：
 *  - 实时摄像头识别：CameraX 取帧 -> ONNX Runtime 推理 -> 画框（含 N-of-M 时序确认）
 *  - 相册选图识别：对已有照片跑一次检测
 *  - 内置示例图（block.jpg）一键测试
 *  - 置信度阈值可调（误检多时调高；漏检时调低）
 */
class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var detector: WoodBlockDetector
    private lateinit var overlay: OverlayView
    private val smoother = TemporalFilter(minHits = 3, window = 8)
    private var analysisExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private var lensFacing = CameraSelector.LENS_FACING_BACK
    private var busy = false
    private var frameCount = 0
    private var lastFpsTime = System.currentTimeMillis()
    private var running = true

    private val requestPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startCamera() else {
                Toast.makeText(this, "没有相机权限，可先用「相册选图」测试", Toast.LENGTH_LONG).show()
            }
        }

    private val pickImage =
        registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
            if (uri != null) detectStillImage(uri)
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        overlay = binding.overlay
        binding.btnSwitch.setOnClickListener { switchCamera() }
        binding.btnGallery.setOnClickListener { pickImage.launch("image/*") }
        binding.btnDemo.setOnClickListener { detectDemoImage() }

        binding.seekConf.max = 60
        binding.seekConf.progress = (0.35f * 100).toInt() - 5
        binding.tvConf.text = getString(R.string.conf_value, 0.35f)
        binding.seekConf.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar?, progress: Int, fromUser: Boolean) {
                val v = (progress + 5) / 100f
                detector.confThreshold = v
                binding.tvConf.text = getString(R.string.conf_value, v)
            }
            override fun onStartTrackingTouch(sb: SeekBar?) {}
            override fun onStopTrackingTouch(sb: SeekBar?) {}
        })

        binding.tvStatus.text = getString(R.string.loading_model)
        Thread {
            try {
                val t0 = System.currentTimeMillis()
                detector = WoodBlockDetector(applicationContext)
                runOnUiThread {
                    binding.tvStatus.text = getString(R.string.model_ready, System.currentTimeMillis() - t0)
                }
                if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                    == PackageManager.PERMISSION_GRANTED
                ) {
                    runOnUiThread { startCamera() }
                } else {
                    requestPermission.launch(Manifest.permission.CAMERA)
                }
            } catch (e: Exception) {
                Log.e(TAG, "模型加载失败", e)
                runOnUiThread {
                    binding.tvStatus.text = getString(R.string.model_failed, e.message ?: "")
                }
            }
        }.start()
    }

    // ------------------------------------------------------------ 摄像头
    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            try {
                bindUseCases(future.get())
            } catch (e: Exception) {
                Log.e(TAG, "相机启动失败", e)
                Toast.makeText(this, "相机启动失败：${e.message}", Toast.LENGTH_LONG).show()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun bindUseCases(provider: ProcessCameraProvider) {
        provider.unbindAll()
        val preview = Preview.Builder().build().also {
            it.setSurfaceProvider(binding.previewView.surfaceProvider)
        }
        val analysis = ImageAnalysis.Builder()
            .setTargetResolution(android.util.Size(640, 480))
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
        analysis.setAnalyzer(analysisExecutor) { proxy -> analyze(proxy) }

        val selector = CameraSelector.Builder().requireLensFacing(lensFacing).build()
        provider.bindToLifecycle(this, selector, preview, analysis)
        smoother.reset()
        binding.tvStatus.text = getString(R.string.status_live)
    }

    private fun switchCamera() {
        lensFacing = if (lensFacing == CameraSelector.LENS_FACING_BACK)
            CameraSelector.LENS_FACING_FRONT else CameraSelector.LENS_FACING_BACK
        startCamera()
    }

    private fun analyze(proxy: ImageProxy) {
        if (!::detector.isInitialized || !running) {
            proxy.close()
            return
        }
        try {
            var bitmap = proxy.toBitmap()
            val rot = proxy.imageInfo.rotationDegrees
            if (rot != 0) bitmap = rotate(bitmap, rot.toFloat())

            val raw = detector.detect(bitmap)
            val shown = smoother.update(raw)

            frameCount++
            val now = System.currentTimeMillis()
            var fps = 0f
            if (now - lastFpsTime >= 1000) {
                fps = frameCount * 1000f / (now - lastFpsTime)
                frameCount = 0
                lastFpsTime = now
            }
            val mirror = lensFacing == CameraSelector.LENS_FACING_FRONT
            runOnUiThread {
                overlay.setResult(shown, bitmap.width, bitmap.height, mirror)
                binding.tvCount.text = getString(R.string.count_fmt, shown.size)
                if (fps > 0f) binding.tvFps.text = getString(R.string.fps_fmt, fps)
            }
        } catch (e: Exception) {
            Log.w(TAG, "分析帧失败", e)
        } finally {
            proxy.close()
        }
    }

    private fun rotate(src: Bitmap, degrees: Float): Bitmap {
        val m = Matrix().apply { postRotate(degrees) }
        return Bitmap.createBitmap(src, 0, 0, src.width, src.height, m, true)
    }

    // ------------------------------------------------------------ 静态图片
    private fun detectStillImage(uri: Uri) {
        Thread {
            try {
                val bmp = contentResolver.openInputStream(uri)?.use {
                    android.graphics.BitmapFactory.decodeStream(it)
                } ?: return@Thread
                val dets = detector.detect(bmp)
                runOnUiThread {
                    showStill(bmp, dets)
                }
            } catch (e: Exception) {
                runOnUiThread { Toast.makeText(this, "读取图片失败：${e.message}", Toast.LENGTH_LONG).show() }
            }
        }.start()
    }

    private fun detectDemoImage() {
        Thread {
            try {
                val bmp = assets.open("block.jpg").use { android.graphics.BitmapFactory.decodeStream(it) }
                val dets = detector.detect(bmp)
                runOnUiThread {
                    showStill(bmp, dets)
                    Toast.makeText(this, getString(R.string.demo_done, dets.size), Toast.LENGTH_SHORT).show()
                }
            } catch (e: Exception) {
                runOnUiThread { Toast.makeText(this, "示例图加载失败：${e.message}", Toast.LENGTH_LONG).show() }
            }
        }.start()
    }

    private fun showStill(bmp: Bitmap, dets: List<Detection>) {
        running = false
        binding.previewView.visibility = View.GONE
        binding.imgStill.visibility = View.VISIBLE
        binding.imgStill.setImageBitmap(bmp)
        binding.btnBackToCamera.visibility = View.VISIBLE
        binding.tvCount.text = getString(R.string.count_fmt, dets.size)
        binding.tvStatus.text = getString(R.string.status_still)
        // 静态图直接展示检测结果（不做时序确认）
        binding.overlay.setResult(dets, bmp.width, bmp.height, false)
        binding.btnBackToCamera.setOnClickListener {
            binding.imgStill.visibility = View.GONE
            binding.btnBackToCamera.visibility = View.GONE
            binding.previewView.visibility = View.VISIBLE
            overlay.clear()
            running = true
            smoother.reset()
            startCamera()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        running = false
        analysisExecutor.shutdown()
        if (::detector.isInitialized) detector.close()
    }

    companion object {
        private const val TAG = "WoodBlock"
    }
}
