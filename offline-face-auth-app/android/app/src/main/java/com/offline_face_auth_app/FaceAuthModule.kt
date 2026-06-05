package com.offline_face_auth_app

import android.content.Context
import android.content.res.AssetFileDescriptor
import android.database.sqlite.SQLiteDatabase
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Base64
import android.util.Log
import com.facebook.react.bridge.*
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker.FaceLandmarkerOptions
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.security.SecureRandom
import java.util.concurrent.Executors
import javax.crypto.Cipher
import javax.crypto.SecretKeyFactory
import javax.crypto.spec.GCMParameterSpec
import javax.crypto.spec.PBEKeySpec
import javax.crypto.spec.SecretKeySpec
import kotlin.math.*

class FaceAuthModule(private val reactContext: ReactApplicationContext) :
    ReactContextBaseJavaModule(reactContext) {

    private var landmarker: FaceLandmarker? = null
    private var interpreter: Interpreter? = null
    private var database: SQLiteDatabase? = null
    private val executor = Executors.newSingleThreadExecutor()

    // Liveness state variables
    private val challenges = listOf("BLINK", "SMILE", "TURN_LEFT", "TURN_RIGHT")
    private var currentChallengeIndex = 0
    private var isLivenessPassed = false

    // State machine trackers
    private var blinkCounter = 0
    private var eyeState = "OPEN"
    private var tBlinkStart = 0.0

    private var smoothedYaw = 0.0f
    private var hasInitEma = false
    private var turnFrameCounter = 0
    private var centerFrameCounter = 0
    private var leftVerified = false
    private var rightVerified = false
    private var baselineYaw = 0.0f
    private var isBaselineCalibrated = false

    private var smileFrameCounter = 0

    init {
        if (!OpenCVLoader.initDebug()) {
            Log.e("FaceAuthModule", "OpenCV initialization failed.")
        } else {
            Log.d("FaceAuthModule", "OpenCV initialization succeeded.")
        }
    }

    override fun getName(): String {
        return "FaceAuthModule"
    }

    @ReactMethod
    fun initialize(dbName: String, promise: Promise) {
        executor.execute {
            try {
                // 1. Initialize MediaPipe FaceLandmarker
                val baseOptions = BaseOptions.builder()
                    .setModelAssetPath("face_landmarker.task")
                    .build()
                val options = FaceLandmarkerOptions.builder()
                    .setBaseOptions(baseOptions)
                    .setMinFaceDetectionConfidence(0.5f)
                    .setMinTrackingConfidence(0.5f)
                    .setRunningMode(RunningMode.IMAGE)
                    .build()
                landmarker = FaceLandmarker.createFromOptions(reactContext, options)

                // 2. Initialize TFLite MobileFaceNet
                val modelBuffer = loadModelFile(reactContext, "mobilefacenet.tflite")
                interpreter = Interpreter(modelBuffer)

                // 3. Initialize SQLite Database
                database = reactContext.openOrCreateDatabase(dbName, Context.MODE_PRIVATE, null)
                createTablesIfNotExist()

                resetLivenessState()

                promise.resolve(true)
            } catch (e: Exception) {
                Log.e("FaceAuthModule", "Initialization failed", e)
                promise.reject("INIT_ERROR", e.message)
            }
        }
    }

    private fun createTablesIfNotExist() {
        database?.execSQL(
            """
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                name TEXT,
                sync_status TEXT DEFAULT 'PENDING',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """.trimIndent()
        )
        database?.execSQL(
            """
            CREATE TABLE IF NOT EXISTS face_embeddings (
                user_id TEXT PRIMARY KEY,
                encrypted_embedding TEXT,
                salt TEXT,
                iv TEXT
            )
            """.trimIndent()
        )
        database?.execSQL(
            """
            CREATE TABLE IF NOT EXISTS auth_logs (
                log_id TEXT PRIMARY KEY,
                user_id TEXT,
                similarity_score REAL,
                liveness_score REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT,
                sync_status TEXT DEFAULT 'PENDING'
            )
            """.trimIndent()
        )
        database?.execSQL(
            """
            CREATE TABLE IF NOT EXISTS sync_queue (
                sync_id TEXT PRIMARY KEY,
                action_type TEXT,
                payload TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """.trimIndent()
        )
    }

    @ReactMethod
    fun resetLiveness(promise: Promise) {
        executor.execute {
            resetLivenessState()
            promise.resolve(true)
        }
    }

    private fun resetLivenessState() {
        currentChallengeIndex = 0
        isLivenessPassed = false
        blinkCounter = 0
        eyeState = "OPEN"
        tBlinkStart = 0.0
        smoothedYaw = 0.0f
        hasInitEma = false
        turnFrameCounter = 0
        centerFrameCounter = 0
        leftVerified = false
        rightVerified = false
        baselineYaw = 0.0f
        isBaselineCalibrated = false
        smileFrameCounter = 0
    }

    @ReactMethod
    fun enrollUser(userId: String, name: String, imageBase64: String, pin: String, promise: Promise) {
        executor.execute {
            try {
                if (database == null || interpreter == null || landmarker == null) {
                    promise.reject("NOT_INITIALIZED", "Biometric engine not initialized.")
                    return@execute
                }

                // 1. Decode base64 image to OpenCV BGR Mat
                val imageBytes = Base64.decode(imageBase64, Base64.DEFAULT)
                val matOfByte = MatOfByte(*imageBytes)
                val frame = Imgcodecs.imdecode(matOfByte, Imgcodecs.IMREAD_COLOR)
                if (frame.empty()) {
                    promise.reject("INVALID_IMAGE", "Could not decode camera image.")
                    return@execute
                }

                // 2. Run Face Mesh Landmarks
                val width = frame.cols()
                val height = frame.rows()
                val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
                Utils.matToBitmap(frame, bitmap)
                val mpImage = BitmapImageBuilder(bitmap).build()
                val mpResult = landmarker?.detect(mpImage)

                if (mpResult == null || mpResult.faceLandmarks().isEmpty()) {
                    promise.reject("FACE_NOT_DETECTED", "No face detected in the image.")
                    return@execute
                }

                val landmarks = mpResult.faceLandmarks()[0]

                // 3. Coordinate Eye Locations
                val ptAx = (landmarks[33].x() + landmarks[133].x()) / 2.0f * width
                val ptAy = (landmarks[33].y() + landmarks[133].y()) / 2.0f * height
                val ptBx = (landmarks[362].x() + landmarks[263].x()) / 2.0f * width
                val ptBy = (landmarks[362].y() + landmarks[263].y()) / 2.0f * height
                val (leftEye, rightEye) = if (ptAx < ptBx) {
                    Pair(Point(ptAx.toDouble(), ptAy.toDouble()), Point(ptBx.toDouble(), ptBy.toDouble()))
                } else {
                    Pair(Point(ptBx.toDouble(), ptBy.toDouble()), Point(ptAx.toDouble(), ptAy.toDouble()))
                }

                // 4. Alignment feasibility checks
                val dx = rightEye.x - leftEye.x
                val dy = rightEye.y - leftEye.y
                val eyeDistance = sqrt(dx * dx + dy * dy)
                if (eyeDistance < 20.0) {
                    promise.reject("FACE_POORLY_POSITIONED", "Face is too far from the camera.")
                    return@execute
                }

                val angle = Math.toDegrees(atan2(dy, dx))
                if (abs(angle) > 45.0) {
                    promise.reject("FACE_POORLY_POSITIONED", "Face tilt angle exceeds 45 degrees.")
                    return@execute
                }

                // Calculate bounding box
                var minX = Float.MAX_VALUE
                var maxX = Float.MIN_VALUE
                var minY = Float.MAX_VALUE
                var maxY = Float.MIN_VALUE
                for (lm in landmarks) {
                    val lx = lm.x() * width
                    val ly = lm.y() * height
                    if (lx < minX) minX = lx
                    if (lx > maxX) maxX = lx
                    if (ly < minY) minY = ly
                    if (ly > maxY) maxY = ly
                }
                val faceW = maxX - minX
                val faceH = maxY - minY
                val faceSize = min(faceW, faceH)
                if (faceSize < 50.0f) {
                    promise.reject("FACE_TOO_SMALL", "Face size is too small.")
                    return@execute
                }

                // 5. Affine Similarity Transform alignment to 112x112
                val dTarget = 112.0 * (1.0 - 2.0 * 0.35)
                val scale = dTarget / eyeDistance
                val midpointInput = Point((leftEye.x + rightEye.x) / 2.0, (leftEye.y + rightEye.y) / 2.0)
                val midpointTarget = Point(112.0 / 2.0, 112.0 * 0.35)

                val R = Imgproc.getRotationMatrix2D(midpointInput, angle, scale)
                val tx = midpointTarget.x - (R.get(0, 0)[0] * midpointInput.x + R.get(0, 1)[0] * midpointInput.y)
                val ty = midpointTarget.y - (R.get(1, 0)[0] * midpointInput.x + R.get(1, 1)[0] * midpointInput.y)
                R.put(0, 2, tx)
                R.put(1, 2, ty)

                val alignedFace = Mat()
                Imgproc.warpAffine(
                    frame,
                    alignedFace,
                    R,
                    Size(112.0, 112.0),
                    Imgproc.INTER_LINEAR,
                    Core.BORDER_CONSTANT,
                    Scalar(0.0, 0.0, 0.0)
                )

                // 6. CLAHE Enhancement
                val grayFace = Mat()
                Imgproc.cvtColor(alignedFace, grayFace, Imgproc.COLOR_BGR2GRAY)
                val clahe = Imgproc.createCLAHE(2.0, Size(8.0, 8.0))
                val enhancedFace = Mat()
                clahe.apply(grayFace, enhancedFace)

                val meanVal = Core.mean(grayFace).`val`[0]
                val stdDevMat = MatOfDouble()
                val meanMat = MatOfDouble()
                Core.meanStdDev(grayFace, meanMat, stdDevMat)
                val stdVal = stdDevMat.get(0, 0)[0]

                // Piecewise Quality thresholds
                if (meanVal < 40.0 || meanVal > 220.0 || stdVal < 15.0) {
                    promise.reject("CHALLENGE_NOT_COMPLETED", "Image contrast/exposure checks failed.")
                    return@execute
                }

                // 7. Blur Detection
                val laplacian = Mat()
                Imgproc.Laplacian(enhancedFace, laplacian, CvType.CV_64F)
                val lMeanMat = MatOfDouble()
                val lStdDevMat = MatOfDouble()
                Core.meanStdDev(laplacian, lMeanMat, lStdDevMat)
                val blurScore = lStdDevMat.get(0, 0)[0] * lStdDevMat.get(0, 0)[0]

                if (blurScore < 50.0) {
                    promise.reject("CHALLENGE_NOT_COMPLETED", "Image is blurry.")
                    return@execute
                }

                // 8. Generate 128D Embedding from MobileFaceNet
                val embedding = runMobileFaceNet(alignedFace)

                // 9. Encrypt Embedding using AES-256 derived from user PIN
                val (encryptedHex, saltHex, ivHex) = encryptEmbedding(embedding, pin)

                // 10. Write to local database tables
                database?.beginTransaction()
                try {
                    database?.execSQL(
                        "INSERT OR REPLACE INTO user_profiles (user_id, name) VALUES (?, ?)",
                        arrayOf(userId, name)
                    )
                    database?.execSQL(
                        "INSERT OR REPLACE INTO face_embeddings (user_id, encrypted_embedding, salt, iv) VALUES (?, ?, ?, ?)",
                        arrayOf(userId, encryptedHex, saltHex, ivHex)
                    )
                    
                    // Queue sync event
                    val payload = """{"user_id":"$userId","name":"$name","encrypted_embedding":"$encryptedHex","salt":"$saltHex","iv":"$ivHex"}"""
                    database?.execSQL(
                        "INSERT INTO sync_queue (sync_id, action_type, payload) VALUES (?, ?, ?)",
                        arrayOf(java.util.UUID.randomUUID().toString(), "ENROLL", payload)
                    )
                    
                    database?.setTransactionSuccessful()
                } finally {
                    database?.endTransaction()
                }

                val result = Arguments.createMap()
                result.putBoolean("success", true)
                result.putString("userId", userId)
                promise.resolve(result)
            } catch (e: Exception) {
                Log.e("FaceAuthModule", "Enrollment failed", e)
                promise.reject("ENROLL_ERROR", e.message)
            }
        }
    }

    @ReactMethod
    fun authenticate(imageBase64: String, pin: String, promise: Promise) {
        executor.execute {
            try {
                if (database == null || interpreter == null || landmarker == null) {
                    promise.reject("NOT_INITIALIZED", "Biometric engine not initialized.")
                    return@execute
                }

                // 1. Decode Base64 Image
                val imageBytes = Base64.decode(imageBase64, Base64.DEFAULT)
                val matOfByte = MatOfByte(*imageBytes)
                val frame = Imgcodecs.imdecode(matOfByte, Imgcodecs.IMREAD_COLOR)
                if (frame.empty()) {
                    promise.reject("INVALID_IMAGE", "Could not decode camera image.")
                    return@execute
                }

                // 2. Extract Landmarks & Align Face
                val width = frame.cols()
                val height = frame.rows()
                val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
                Utils.matToBitmap(frame, bitmap)
                val mpImage = BitmapImageBuilder(bitmap).build()
                val mpResult = landmarker?.detect(mpImage)

                if (mpResult == null || mpResult.faceLandmarks().isEmpty()) {
                    promise.reject("FACE_NOT_DETECTED", "No face detected in the image.")
                    return@execute
                }

                val landmarks = mpResult.faceLandmarks()[0]

                // Eye locations
                val ptAx = (landmarks[33].x() + landmarks[133].x()) / 2.0f * width
                val ptAy = (landmarks[33].y() + landmarks[133].y()) / 2.0f * height
                val ptBx = (landmarks[362].x() + landmarks[263].x()) / 2.0f * width
                val ptBy = (landmarks[362].y() + landmarks[263].y()) / 2.0f * height
                val (leftEye, rightEye) = if (ptAx < ptBx) {
                    Pair(Point(ptAx.toDouble(), ptAy.toDouble()), Point(ptBx.toDouble(), ptBy.toDouble()))
                } else {
                    Pair(Point(ptBx.toDouble(), ptBy.toDouble()), Point(ptAx.toDouble(), ptAy.toDouble()))
                }

                val dx = rightEye.x - leftEye.x
                val dy = rightEye.y - leftEye.y
                val eyeDistance = sqrt(dx * dx + dy * dy)
                val angle = Math.toDegrees(atan2(dy, dx))

                val dTarget = 112.0 * (1.0 - 2.0 * 0.35)
                val scale = dTarget / eyeDistance
                val midpointInput = Point((leftEye.x + rightEye.x) / 2.0, (leftEye.y + rightEye.y) / 2.0)
                val midpointTarget = Point(112.0 / 2.0, 112.0 * 0.35)

                val R = Imgproc.getRotationMatrix2D(midpointInput, angle, scale)
                val tx = midpointTarget.x - (R.get(0, 0)[0] * midpointInput.x + R.get(0, 1)[0] * midpointInput.y)
                val ty = midpointTarget.y - (R.get(1, 0)[0] * midpointInput.x + R.get(1, 1)[0] * midpointInput.y)
                R.put(0, 2, tx)
                R.put(1, 2, ty)

                val alignedFace = Mat()
                Imgproc.warpAffine(
                    frame,
                    alignedFace,
                    R,
                    Size(112.0, 112.0),
                    Imgproc.INTER_LINEAR,
                    Core.BORDER_CONSTANT,
                    Scalar(0.0, 0.0, 0.0)
                )

                // 3. Generate query embedding from current face
                val queryEmbedding = runMobileFaceNet(alignedFace)

                // 4. Retrieve and Decrypt all enrolled templates
                val enrolledEmbeddings = mutableListOf<Pair<String, FloatArray>>()
                val cursor = database?.rawQuery("SELECT user_id, encrypted_embedding, salt, iv FROM face_embeddings", null)
                if (cursor != null) {
                    while (cursor.moveToNext()) {
                        val uId = cursor.getString(0)
                        val encryptedHex = cursor.getString(1)
                        val saltHex = cursor.getString(2)
                        val ivHex = cursor.getString(3)

                        val emb = decryptEmbedding(encryptedHex, saltHex, ivHex, pin)
                        if (emb != null) {
                            enrolledEmbeddings.add(Pair(uId, emb))
                        }
                    }
                    cursor.close()
                }

                if (enrolledEmbeddings.isEmpty()) {
                    promise.reject("NO_USERS_FOUND", "Wrong PIN or no registered users in database.")
                    return@execute
                }

                // 5. Compute Cosine Similarity (Dot Product match)
                var matchedUserId: String? = null
                var maxScore = 0.0f
                for ((uId, emb) in enrolledEmbeddings) {
                    var dotProduct = 0.0f
                    for (i in 0 until 128) {
                        dotProduct += emb[i] * queryEmbedding[i]
                    }
                    if (dotProduct > maxScore) {
                        maxScore = dotProduct
                        matchedUserId = uId
                    }
                }

                val threshold = 0.60f
                val isMatched = matchedUserId != null && maxScore >= threshold

                // 6. Log results & sync logs
                val finalStatus = if (isMatched) "SUCCESS" else "FAILED"
                val logId = java.util.UUID.randomUUID().toString()
                database?.execSQL(
                    "INSERT INTO auth_logs (log_id, user_id, similarity_score, liveness_score, status) VALUES (?, ?, ?, ?, ?)",
                    arrayOf(logId, matchedUserId ?: "UNKNOWN", maxScore, 1.0f, finalStatus)
                )

                // Queue sync event for auth log
                val payload = """{"log_id":"$logId","user_id":"${matchedUserId ?: "UNKNOWN"}","similarity_score":$maxScore,"liveness_score":1.0,"status":"$finalStatus","timestamp":${System.currentTimeMillis()}}"""
                database?.execSQL(
                    "INSERT INTO sync_queue (sync_id, action_type, payload) VALUES (?, ?, ?)",
                    arrayOf(java.util.UUID.randomUUID().toString(), "LOG", payload)
                )

                val result = Arguments.createMap()
                result.putBoolean("success", isMatched)
                result.putString("userId", matchedUserId)
                result.putDouble("similarityScore", maxScore.toDouble())
                if (isMatched) {
                    val pCursor = database?.rawQuery("SELECT name FROM user_profiles WHERE user_id = ?", arrayOf(matchedUserId))
                    if (pCursor != null && pCursor.moveToFirst()) {
                        result.putString("name", pCursor.getString(0))
                        pCursor.close()
                    }
                }
                promise.resolve(result)
            } catch (e: Exception) {
                Log.e("FaceAuthModule", "Authentication failed", e)
                promise.reject("AUTH_ERROR", e.message)
            }
        }
    }

    @ReactMethod
    fun processFrame(imageBase64: String, timestampMs: Double, promise: Promise) {
        executor.execute {
            try {
                if (database == null || interpreter == null || landmarker == null) {
                    promise.reject("NOT_INITIALIZED", "Biometric engine not initialized.")
                    return@execute
                }

                val timestamp = timestampMs / 1000.0 // convert to seconds

                // 1. Decode image
                val imageBytes = Base64.decode(imageBase64, Base64.DEFAULT)
                val matOfByte = MatOfByte(*imageBytes)
                val frame = Imgcodecs.imdecode(matOfByte, Imgcodecs.IMREAD_COLOR)
                if (frame.empty()) {
                    val resp = buildErrorResponse("FACE_NOT_DETECTED")
                    promise.resolve(resp)
                    return@execute
                }

                val width = frame.cols()
                val height = frame.rows()
                Log.e("FaceAuthModule", "processFrame - Decoded image size: ${width}x${height}")

                // 2. MediaPipe Landmarks
                val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
                Utils.matToBitmap(frame, bitmap)
                val mpImage = BitmapImageBuilder(bitmap).build()
                val mpResult = landmarker?.detect(mpImage)

                if (mpResult == null || mpResult.faceLandmarks().isEmpty()) {
                    Log.e("FaceAuthModule", "processFrame - FACE_NOT_DETECTED by MediaPipe")
                    val resp = buildErrorResponse("FACE_NOT_DETECTED")
                    promise.resolve(resp)
                    return@execute
                }

                val landmarks = mpResult.faceLandmarks()[0]
                Log.e("FaceAuthModule", "processFrame - Face detected, landmarks count = ${landmarks.size}")

                // 3. Face size FQA gate
                var minX = Float.MAX_VALUE
                var maxX = Float.MIN_VALUE
                var minY = Float.MAX_VALUE
                var maxY = Float.MIN_VALUE
                for (lm in landmarks) {
                    val lx = lm.x() * width
                    val ly = lm.y() * height
                    if (lx < minX) minX = lx
                    if (lx > maxX) maxX = lx
                    if (ly < minY) minY = ly
                    if (ly > maxY) maxY = ly
                }
                val faceW = maxX - minX
                val faceH = maxY - minY
                val faceSize = min(faceW, faceH)
                Log.e("FaceAuthModule", "processFrame - Face size: faceW=$faceW, faceH=$faceH, faceSize=$faceSize")
                if (faceSize < 80.0f) {
                    Log.e("FaceAuthModule", "processFrame - FACE_TOO_SMALL: faceSize=$faceSize < 80.0")
                    val resp = buildErrorResponse("FACE_TOO_SMALL")
                    promise.resolve(resp)
                    return@execute
                }

                // 4. Liveness State Machine checks
                val currentChallenge = if (currentChallengeIndex < challenges.size) challenges[currentChallengeIndex] else "PASSED"
                var challengeVerified = false

                val rawYaw = calculateYaw(landmarks, width)
                if (!isBaselineCalibrated) {
                    baselineYaw = rawYaw
                    isBaselineCalibrated = true
                    Log.e("FaceAuthModule", "Calibrated initial baselineYaw: $baselineYaw")
                }

                if (currentChallenge == "BLINK") {
                    baselineYaw = (0.95f * baselineYaw) + (0.05f * rawYaw)
                    val ear = calculateEar(landmarks, width, height)
                    Log.e("FaceAuthModule", "BLINK: ear=$ear, rawYaw=$rawYaw, baselineYaw=$baselineYaw, blinkCounter=$blinkCounter, eyeState=$eyeState")
                    if (ear < 0.24f) {
                        if (blinkCounter == 0) {
                            tBlinkStart = timestamp
                        }
                        blinkCounter++
                        if (blinkCounter >= 2) {
                            eyeState = "CLOSED"
                        }
                    } else {
                        if (eyeState == "CLOSED") {
                            val duration = timestamp - tBlinkStart
                            Log.e("FaceAuthModule", "BLINK closed-to-open: duration=$duration")
                            if (duration in 0.05..0.85) {
                                challengeVerified = true
                                currentChallengeIndex++
                            }
                        }
                        blinkCounter = 0
                        tBlinkStart = 0.0
                        eyeState = "OPEN"
                    }
                } else if (currentChallenge == "TURN_LEFT" || currentChallenge == "TURN_RIGHT") {
                    val relativeYaw = rawYaw - baselineYaw
                    if (!hasInitEma) {
                        smoothedYaw = relativeYaw
                        hasInitEma = true
                    } else {
                        smoothedYaw = (0.85f * relativeYaw) + (0.15f * smoothedYaw)
                    }
                    Log.e("FaceAuthModule", "TURN: challenge=$currentChallenge, rawYaw=$rawYaw, baselineYaw=$baselineYaw, relativeYaw=$relativeYaw, smoothedYaw=$smoothedYaw, turnCount=$turnFrameCounter, centerCount=$centerFrameCounter")

                    if (currentChallenge == "TURN_LEFT") {
                        if (!leftVerified) {
                            if (smoothedYaw <= -8.0f) {
                                turnFrameCounter++
                                if (turnFrameCounter >= 1) {
                                    leftVerified = true
                                    centerFrameCounter = 0
                                }
                            } else {
                                turnFrameCounter = 0
                            }
                        } else {
                            if (abs(smoothedYaw) < 5.0f) {
                                centerFrameCounter++
                                if (centerFrameCounter >= 1) {
                                    challengeVerified = true
                                    currentChallengeIndex++
                                    // reset counters for next
                                    turnFrameCounter = 0
                                    centerFrameCounter = 0
                                    hasInitEma = false
                                }
                            } else {
                                centerFrameCounter = 0
                            }
                        }
                    } else { // TURN_RIGHT
                        if (!rightVerified) {
                            if (smoothedYaw >= 8.0f) {
                                turnFrameCounter++
                                if (turnFrameCounter >= 1) {
                                    rightVerified = true
                                    centerFrameCounter = 0
                                }
                            } else {
                                turnFrameCounter = 0
                            }
                        } else {
                            if (abs(smoothedYaw) < 5.0f) {
                                centerFrameCounter++
                                if (centerFrameCounter >= 1) {
                                    challengeVerified = true
                                    currentChallengeIndex++
                                    turnFrameCounter = 0
                                    centerFrameCounter = 0
                                    hasInitEma = false
                                }
                            } else {
                                centerFrameCounter = 0
                            }
                        }
                    }
                } else if (currentChallenge == "SMILE") {
                    val smileScore = calculateSmileScore(landmarks, width, height)
                    if (smileScore >= 0.45f) {
                        smileFrameCounter++
                        if (smileFrameCounter >= 3) {
                            challengeVerified = true
                            currentChallengeIndex++
                        }
                    } else {
                        smileFrameCounter = 0
                    }
                }

                if (currentChallengeIndex >= challenges.size) {
                    isLivenessPassed = true
                }

                val progress = Arguments.createArray()
                for (i in challenges.indices) {
                    progress.pushBoolean(i < currentChallengeIndex)
                }

                val resp = Arguments.createMap()
                resp.putBoolean("success", challengeVerified)
                resp.putString("currentChallenge", if (currentChallengeIndex < challenges.size) challenges[currentChallengeIndex] else "PASSED")
                resp.putArray("challengeProgress", progress)
                resp.putBoolean("livenessPassed", isLivenessPassed)
                resp.putNull("error")
                promise.resolve(resp)
            } catch (e: Exception) {
                Log.e("FaceAuthModule", "processFrame failed", e)
                promise.resolve(buildErrorResponse("LANDMARKS_UNSTABLE"))
            }
        }
    }

    private fun buildErrorResponse(errorType: String): WritableMap {
        val resp = Arguments.createMap()
        resp.putBoolean("success", false)
        resp.putString("currentChallenge", if (currentChallengeIndex < challenges.size) challenges[currentChallengeIndex] else "PASSED")
        val progress = Arguments.createArray()
        for (i in challenges.indices) {
            progress.pushBoolean(i < currentChallengeIndex)
        }
        resp.putArray("challengeProgress", progress)
        resp.putBoolean("livenessPassed", isLivenessPassed)
        resp.putString("error", errorType)
        return resp
    }

    private fun calculateEar(landmarks: List<com.google.mediapipe.tasks.components.containers.NormalizedLandmark>, width: Int, height: Int): Float {
        fun dist(i1: Int, i2: Int): Float {
            val dx = (landmarks[i1].x() - landmarks[i2].x()) * width
            val dy = (landmarks[i1].y() - landmarks[i2].y()) * height
            return sqrt(dx * dx + dy * dy)
        }
        val leftEar = (dist(160, 144) + dist(158, 145)) / (2.0f * dist(33, 133))
        val rightEar = (dist(385, 380) + dist(387, 373)) / (2.0f * dist(362, 263))
        return (leftEar + rightEar) / 2.0f
    }

    private fun calculateYaw(landmarks: List<com.google.mediapipe.tasks.components.containers.NormalizedLandmark>, width: Int): Float {
        val xLe = ((landmarks[33].x() + landmarks[133].x()) / 2.0f) * width
        val xRe = ((landmarks[362].x() + landmarks[263].x()) / 2.0f) * width
        val xN = landmarks[1].x() * width

        val dxL = abs(xLe - xN)
        val dxR = abs(xRe - xN)
        val denom = dxL + dxR
        if (denom < 1e-5f) return 0.0f
        val R = (dxL - dxR) / denom
        return R * 80.0f
    }

    private fun calculateSmileScore(landmarks: List<com.google.mediapipe.tasks.components.containers.NormalizedLandmark>, width: Int, height: Int): Float {
        fun dist(i1: Int, i2: Int): Float {
            val dx = (landmarks[i1].x() - landmarks[i2].x()) * width
            val dy = (landmarks[i1].y() - landmarks[i2].y()) * height
            return sqrt(dx * dx + dy * dy)
        }
        val mouthW = dist(61, 291)
        val mouthH = dist(13, 14)

        val xLe = (landmarks[33].x() + landmarks[133].x()) / 2.0f
        val yLe = (landmarks[33].y() + landmarks[133].y()) / 2.0f
        val xRe = (landmarks[362].x() + landmarks[263].x()) / 2.0f
        val yRe = (landmarks[362].y() + landmarks[263].y()) / 2.0f

        val dx = (xLe - xRe) * width
        val dy = (yLe - yRe) * height
        val dEyes = sqrt(dx * dx + dy * dy)

        if (dEyes < 1e-5f) return 0.0f
        val wNorm = mouthW / dEyes
        val hNorm = mouthH / dEyes
        return 0.7f * wNorm + 0.3f * hNorm
    }

    private fun runMobileFaceNet(alignedFace: Mat): FloatArray {
        // Convert to RGB
        val rgbFace = Mat()
        Imgproc.cvtColor(alignedFace, rgbFace, Imgproc.COLOR_BGR2RGB)

        // Read pixels into FloatArray
        val imgData = ByteBuffer.allocateDirect(1 * 112 * 112 * 3 * 4)
        imgData.order(ByteOrder.nativeOrder())

        val tempBytes = ByteArray(112 * 112 * 3)
        rgbFace.get(0, 0, tempBytes)

        for (i in 0 until 112 * 112 * 3) {
            val pixelVal = tempBytes[i].toInt() and 0xFF
            val normalizedVal = (pixelVal.toFloat() - 127.5f) / 128.0f
            imgData.putFloat(normalizedVal)
        }

        val outData = Array(1) { FloatArray(128) }
        interpreter?.run(imgData, outData)

        val embedding = outData[0]
        var norm = 0.0f
        for (f in embedding) {
            norm += f * f
        }
        norm = sqrt(norm)
        if (norm > 1e-5f) {
            for (i in 0 until 128) {
                embedding[i] /= norm
            }
        }
        return embedding
    }

    // Database helper API endpoints for UI Screens (React Native layer)
    @ReactMethod
    fun getUsers(promise: Promise) {
        executor.execute {
            try {
                val array = Arguments.createArray()
                val cursor = database?.rawQuery("SELECT user_id, name, created_at FROM user_profiles ORDER BY created_at DESC", null)
                if (cursor != null) {
                    while (cursor.moveToNext()) {
                        val map = Arguments.createMap()
                        map.putString("userId", cursor.getString(0))
                        map.putString("name", cursor.getString(1))
                        map.putString("createdAt", cursor.getString(2))
                        array.pushMap(map)
                    }
                    cursor.close()
                }
                promise.resolve(array)
            } catch (e: java.lang.Exception) {
                promise.reject("DB_ERROR", e.message)
            }
        }
    }

    @ReactMethod
    fun deleteUser(userId: String, promise: Promise) {
        executor.execute {
            try {
                database?.beginTransaction()
                try {
                    database?.execSQL("DELETE FROM user_profiles WHERE user_id = ?", arrayOf(userId))
                    database?.execSQL("DELETE FROM face_embeddings WHERE user_id = ?", arrayOf(userId))
                    
                    val payload = """{"user_id":"$userId"}"""
                    database?.execSQL(
                        "INSERT INTO sync_queue (sync_id, action_type, payload) VALUES (?, ?, ?)",
                        arrayOf(java.util.UUID.randomUUID().toString(), "DELETE", payload)
                    )
                    database?.setTransactionSuccessful()
                } finally {
                    database?.endTransaction()
                }
                promise.resolve(true)
            } catch (e: java.lang.Exception) {
                promise.reject("DB_ERROR", e.message)
            }
        }
    }

    @ReactMethod
    fun getSyncQueue(promise: Promise) {
        executor.execute {
            try {
                val array = Arguments.createArray()
                val cursor = database?.rawQuery("SELECT sync_id, action_type, payload, created_at FROM sync_queue ORDER BY created_at ASC", null)
                if (cursor != null) {
                    while (cursor.moveToNext()) {
                        val map = Arguments.createMap()
                        map.putString("syncId", cursor.getString(0))
                        map.putString("actionType", cursor.getString(1))
                        map.putString("payload", cursor.getString(2))
                        map.putString("createdAt", cursor.getString(3))
                        array.pushMap(map)
                    }
                    cursor.close()
                }
                promise.resolve(array)
            } catch (e: java.lang.Exception) {
                promise.reject("DB_ERROR", e.message)
            }
        }
    }

    @ReactMethod
    fun deleteSyncItem(syncId: String, promise: Promise) {
        executor.execute {
            try {
                database?.execSQL("DELETE FROM sync_queue WHERE sync_id = ?", arrayOf(syncId))
                promise.resolve(true)
            } catch (e: java.lang.Exception) {
                promise.reject("DB_ERROR", e.message)
            }
        }
    }

    @ReactMethod
    fun restoreUser(userId: String, name: String, encryptedEmbedding: String, salt: String, iv: String, promise: Promise) {
        executor.execute {
            try {
                database?.beginTransaction()
                try {
                    database?.execSQL(
                        "INSERT OR REPLACE INTO user_profiles (user_id, name, sync_status) VALUES (?, ?, 'SYNCED')",
                        arrayOf(userId, name)
                    )
                    database?.execSQL(
                        "INSERT OR REPLACE INTO face_embeddings (user_id, encrypted_embedding, salt, iv) VALUES (?, ?, ?, ?)",
                        arrayOf(userId, encryptedEmbedding, salt, iv)
                    )
                    database?.setTransactionSuccessful()
                } finally {
                    database?.endTransaction()
                }
                promise.resolve(true)
            } catch (e: java.lang.Exception) {
                promise.reject("DB_ERROR", e.message)
            }
        }
    }

    @ReactMethod
    fun getAuthLogs(promise: Promise) {
        executor.execute {
            try {
                val array = Arguments.createArray()
                val cursor = database?.rawQuery("SELECT log_id, user_id, similarity_score, liveness_score, timestamp, status FROM auth_logs ORDER BY timestamp DESC", null)
                if (cursor != null) {
                    while (cursor.moveToNext()) {
                        val map = Arguments.createMap()
                        map.putString("logId", cursor.getString(0))
                        map.putString("userId", cursor.getString(1))
                        map.putDouble("similarityScore", cursor.getDouble(2))
                        map.putDouble("livenessScore", cursor.getDouble(3))
                        map.putString("timestamp", cursor.getString(4))
                        map.putString("status", cursor.getString(5))
                        array.pushMap(map)
                    }
                    cursor.close()
                }
                promise.resolve(array)
            } catch (e: java.lang.Exception) {
                promise.reject("DB_ERROR", e.message)
            }
        }
    }

    // Cryptography Helpers
    private fun deriveKey(pin: String, salt: ByteArray): SecretKeySpec {
        val spec = PBEKeySpec(pin.toCharArray(), salt, 10000, 256)
        val factory = SecretKeyFactory.getInstance("PBKDF2WithHmacSHA256")
        val keyBytes = factory.generateSecret(spec).encoded
        return SecretKeySpec(keyBytes, "AES")
    }

    private fun encryptEmbedding(embedding: FloatArray, pin: String): Triple<String, String, String> {
        val salt = ByteArray(16)
        SecureRandom().nextBytes(salt)
        val iv = ByteArray(12)
        SecureRandom().nextBytes(iv)

        val secretKey = deriveKey(pin, salt)
        val cipher = Cipher.getInstance("AES/GCM/NoPadding")
        val spec = GCMParameterSpec(128, iv)
        cipher.init(Cipher.ENCRYPT_MODE, secretKey, spec)

        val byteBuffer = ByteBuffer.allocate(128 * 4)
        byteBuffer.order(ByteOrder.LITTLE_ENDIAN)
        for (f in embedding) {
            byteBuffer.putFloat(f)
        }
        val encryptedBytes = cipher.doFinal(byteBuffer.array())

        return Triple(
            toHex(encryptedBytes),
            toHex(salt),
            toHex(iv)
        )
    }

    private fun decryptEmbedding(encryptedHex: String, saltHex: String, ivHex: String, pin: String): FloatArray? {
        try {
            val encrypted = fromHex(encryptedHex)
            val salt = fromHex(saltHex)
            val iv = fromHex(ivHex)

            val secretKey = deriveKey(pin, salt)
            val cipher = Cipher.getInstance("AES/GCM/NoPadding")
            val spec = GCMParameterSpec(128, iv)
            cipher.init(Cipher.DECRYPT_MODE, secretKey, spec)

            val decryptedBytes = cipher.doFinal(encrypted)
            val byteBuffer = ByteBuffer.wrap(decryptedBytes)
            byteBuffer.order(ByteOrder.LITTLE_ENDIAN)
            val embedding = FloatArray(128)
            for (i in 0 until 128) {
                embedding[i] = byteBuffer.float
            }
            return embedding
        } catch (e: Exception) {
            return null
        }
    }

    private fun toHex(bytes: ByteArray): String {
        return bytes.joinToString("") { "%02x".format(it) }
    }

    private fun fromHex(hex: String): ByteArray {
        val len = hex.length
        val data = ByteArray(len / 2)
        var i = 0
        while (i < len) {
            data[i / 2] = ((Character.digit(hex[i], 16) shl 4) + Character.digit(hex[i + 1], 16)).toByte()
            i += 2
        }
        return data
    }

    private fun loadModelFile(context: Context, modelPath: String): MappedByteBuffer {
        val fileDescriptor: AssetFileDescriptor = context.assets.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel: FileChannel = inputStream.channel
        val startOffset: Long = fileDescriptor.startOffset
        val declaredLength: Long = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
}
