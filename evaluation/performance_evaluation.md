# Performance Evaluation Report: Face Recognition Subsystem

This report provides evidence-backed engineering performance benchmarks for the offline Face Recognition subsystem. Timings were measured on the local host CPU environment.

---

## 1. MobileFaceNet Inference Latency

Model inference was benchmarked by running `FaceRecognizer.extract_embedding()` 100 times using a pre-aligned BGR face crop.

* **Total Inference Runs**: `100`
* **Mean Latency**: `6.88 ms`
* **Median Latency**: `6.39 ms`
* **95th Percentile (P95)**: `10.10 ms`
* **99th Percentile (P99)**: `10.80 ms`
* **Minimum Latency**: `5.80 ms`
* **Maximum Latency**: `13.24 ms`

> [!TIP]
> **Real-time Viability**: With a mean latency of `6.88 ms`, the MobileFaceNet model runs at a throughput of approximately **145 frames per second** on a single CPU thread. This latency is well within real-world constraints for offline mobile apps, leaving ample CPU overhead for UI rendering and liveness challenge detectors.

---

## 2. Memory Usage Metrics

Process resident set size (RSS) memory was monitored using `psutil` at key execution milestones:

1. **Before Model Init**: `333.82 MB` (Baseline workspace runtime)
2. **After Model Init**: `379.82 MB` (TFLite interpreter loaded)
3. **After First Inference**: `382.28 MB` (Tensors allocated and executed)
4. **After 100 Inferences**: `380.35 MB` (Stable loop execution)

* **Model Initialization Delta**: `+46.00 MB` (Interpreter and weights load overhead)
* **First Inference Delta**: `+2.46 MB` (Activation tensor buffer allocation)
* **Subsequent Execution Delta**: `-1.93 MB` (Garbage collection / stabilizing)
* **Peak Observed Memory**: `382.28 MB`

> [!NOTE]
> **Memory Footprint**: The MobileFaceNet model file requires a very low resident memory footprint of ~46 MB, which is ideal for low-end mobile devices and resource-constrained embedded systems. Memory consumption stabilizes immediately after the first run, proving there are no persistent memory leaks in the TFLite wrapper.

---

## 3. SQLite Registry Scalability Benchmarks

Search scalability was benchmarked using `SQLiteFaceRegistry.verify()` against vector databases populated with simulated unit-normalized 128D embeddings. The search operates in-memory on cached NumPy arrays.

| Database Size | Mean Latency | Median Latency | P95 Latency | P99 Latency | Min / Max Latency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **10 Users** | 0.0027 ms | 0.0020 ms | 0.0038 ms | 0.0162 ms | 0.0019 / 0.0438 ms |
| **100 Users** | 0.0030 ms | 0.0027 ms | 0.0030 ms | 0.0063 ms | 0.0026 / 0.0276 ms |
| **1,000 Users** | 0.0098 ms | 0.0084 ms | 0.0121 ms | 0.0408 ms | 0.0075 / 0.0430 ms |
| **10,000 Users**| 0.1217 ms | 0.0749 ms | 0.2142 ms | 0.6857 ms | 0.0588 / 1.9386 ms |

> [!TIP]
> **Vector Search Performance**: Even at a scale of 10,000 users, the cached database lookup takes a mean of **0.12 ms** and has a P99 latency of **0.68 ms**. This sub-millisecond search time is achieved through vectorized NumPy dot product matrix multiplication (`np.dot`), bypassing the slow SQLite disk read during active verification.

---

## 4. Enrollment Performance Benchmarks

The enrollment time, including SQLite database write (`INSERT OR REPLACE`) and in-memory cache synchronization, was measured separately.

* **100 Enrollments**:
  * **Mean**: `5.12 ms`
  * **Median**: `4.91 ms`
  * **P95**: `6.67 ms`
  * **P99**: `7.28 ms`
* **1,000 Enrollments**:
  * **Mean**: `4.72 ms`
  * **Median**: `4.44 ms`
  * **P95**: `6.37 ms`
  * **P99**: `7.90 ms`

> [!NOTE]
> **Enrollment Throughput**: SQLite database updates take under 5 ms on average. Since enrollment is an infrequent, single-write event during user registration, a latency of < 8 ms at p99 guarantees instant, responsive feedback to the client application.

---

## 5. Concurrent Verification Benchmarks (Lock Contention)

To evaluate multi-threaded lock contention, we spawned multiple threads performing 1,000 concurrent verification queries on a 1,000-user database.

* **1 Thread**: Total Duration: `8.1 ms` | Throughput: `123,568 ops/sec` | Avg Latency: `0.0073 ms` | Max: `0.1 ms`
* **5 Threads**: Total Duration: `60.8 ms` | Throughput: `82,205 ops/sec` | Avg Latency: `0.0571 ms` | Max: `1.8 ms`
* **10 Threads**: Total Duration: `129.3 ms` | Throughput: `77,341 ops/sec` | Avg Latency: `0.1208 ms` | Max: `1.7 ms`
* **20 Threads**: Total Duration: `256.6 ms` | Throughput: `77,942 ops/sec` | Avg Latency: `0.2431 ms` | Max: `1.9 ms`

> [!WARNING]
> **Lock Contention**: The average latency scales linearly with thread count (`0.007 ms` → `0.243 ms`), while the aggregate throughput peaks and slightly declines. This behavior is caused by the thread lock (`self.lock`) wrapped around `SQLiteFaceRegistry.verify()`, which forces threads to execute vector search sequentially. This design is necessary to protect the cache from corruption during parallel database edits, and is fully acceptable given that client attendance punch-ins occur sequentially.

---

## 6. Cold Start Benchmarks

 Timings for starting up the components from scratch (first import/initialization):

* **TFLite Module Import**: `0.29 ms`
* **Model File Load**: `2.71 ms`
* **Tensor Allocation**: `5.95 ms`
* **SQLite Fresh DB Init**: `16.84 ms`
* **Cache Load (100 Users)**: `1.55 ms`
* **FaceAuthSDK Startup**: `20.39 ms`

> [!NOTE]
> **Startup Time**: The entire `FaceAuthSDK` initializes in **20.39 ms**, which is virtually instantaneous. Cold start is completely imperceptible to the end user, enabling rapid app launch times.
