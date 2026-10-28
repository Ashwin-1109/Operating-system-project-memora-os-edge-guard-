"""
MemoraOS EdgeGuard — Automated Test Suite
Unit tests for OS memory allocation, Banker's Algorithm, PASMA+ admission,
MemoraBrain advisor, telemetry, and metrics calculation.
"""

import time
import pytest
from edgeguard import (
    LogicalMemoryPool,
    PULSEFit,
    SafeGuard,
    MemoraBrain,
    MemoraExplain,
    PASMAAdmissionController,
)
from inference import (
    LlamaCppClient,
    ExperimentResult,
    RequestTelemetry,
    compare_benchmarks,
)


# ==============================================================================
# 1. LOGICAL MEMORY POOL & PLACEMENT ALGORITHMS
# ==============================================================================

def test_pool_initialization():
    pool = LogicalMemoryPool(capacity_mb=1024)
    assert pool.total_free_mb == 1024
    assert pool.total_allocated_mb == 0
    assert pool.external_fragmentation == 0.0
    assert len(pool.blocks) == 1
    assert pool.blocks[0].status == "FREE"


def test_first_fit_allocation():
    pool = LogicalMemoryPool(capacity_mb=1024)
    block = pool.allocate_first_fit(size_mb=256, request_id="req_01")
    assert block is not None
    assert block.size_mb == 256
    assert block.start_mb == 0
    assert block.status == "ALLOCATED"
    assert pool.total_allocated_mb == 256
    assert pool.total_free_mb == 768


def test_best_fit_allocation():
    """Best Fit should pick the smallest free block that can satisfy the request."""
    pool = LogicalMemoryPool(capacity_mb=1000)
    # Create fragments: allocate 3 blocks
    b1 = pool.allocate_first_fit(200, "r1")  # [0, 200]
    b2 = pool.allocate_first_fit(300, "r2")  # [200, 500]
    b3 = pool.allocate_first_fit(200, "r3")  # [500, 700]
    # Free r1 (200 MiB free) and remainder at end is 300 MiB free
    pool.release("r1")

    # Now we have free blocks of 200 MiB and 300 MiB
    # A request of 180 MiB should choose the 200 MiB block (Best Fit)
    allocated = pool.allocate_best_fit(180, "r_target")
    assert allocated is not None
    assert allocated.start_mb == 0  # picked the 200 MiB block, not the 300 MiB one


def test_worst_fit_allocation():
    """Worst Fit should pick the largest free block to leave the biggest remainder."""
    pool = LogicalMemoryPool(capacity_mb=1000)
    b1 = pool.allocate_first_fit(200, "r1")  # [0, 200]
    b2 = pool.allocate_first_fit(300, "r2")  # [200, 500]
    pool.release("r1")  # 200 MiB free at start; 500 MiB free at end

    # Request of 150 MiB with Worst Fit should pick the 500 MiB block at end (start_mb = 500)
    allocated = pool.allocate_worst_fit(150, "r_worst")
    assert allocated is not None
    assert allocated.start_mb == 500


def test_next_fit_allocation():
    """Next Fit continues from where the last allocation occurred."""
    pool = LogicalMemoryPool(capacity_mb=1000)
    b1 = pool.allocate_next_fit(200, "r1")
    assert b1.start_mb == 0
    b2 = pool.allocate_next_fit(200, "r2")
    assert b2.start_mb == 200


def test_memory_coalescing_on_release():
    """Releasing adjacent blocks must coalesce them into a single free block."""
    pool = LogicalMemoryPool(capacity_mb=1024)
    pool.allocate_first_fit(256, "r1")
    pool.allocate_first_fit(256, "r2")
    pool.allocate_first_fit(256, "r3")

    # Release r2 first (creates a hole)
    pool.release("r2")
    assert len(pool.blocks) == 4

    # Release r1 (r1 and r2 are contiguous and free -> coalesces to 512 MiB)
    pool.release("r1")
    free_blocks = [b for b in pool.blocks if b.status == "FREE"]
    # First free block should now be 512 MiB
    assert free_blocks[0].size_mb == 512
    assert free_blocks[0].start_mb == 0

    # Release r3 -> all contiguous free blocks coalesce into one 1024 MiB block
    pool.release("r3")
    assert len(pool.blocks) == 1
    assert pool.blocks[0].size_mb == 1024
    assert pool.external_fragmentation == 0.0


def test_external_fragmentation_metric():
    pool = LogicalMemoryPool(capacity_mb=1000)
    # Create disjoint free blocks: 100 free, 100 alloc, 100 free, 700 alloc
    pool.allocate_first_fit(100, "r1")
    pool.allocate_first_fit(100, "r2")
    pool.allocate_first_fit(100, "r3")
    pool.allocate_first_fit(700, "r4")

    pool.release("r1")
    pool.release("r3")
    # Total free = 200 MiB, largest free block = 100 MiB
    # EF = 1.0 - (100 / 200) = 0.50
    assert pool.external_fragmentation == 0.50


# ==============================================================================
# 2. SAFE-GUARD: BANKER'S ALGORITHM
# ==============================================================================

def test_banker_safe_state_detection():
    """Verifies that Banker's algorithm correctly identifies a safe state and sequence."""
    safeguard = SafeGuard(total_slots=4, total_memory_mb=1000)

    # Process P1 requests [1, 200] with max [1, 300]
    granted, reason, seq = safeguard.evaluate_request(
        request_id="P1",
        request_vector=[1, 200],
        max_claim_vector=[1, 300]
    )
    assert granted is True
    assert "P1" in seq

    # Process P2 requests [1, 300] with max [1, 400]
    granted2, reason2, seq2 = safeguard.evaluate_request(
        request_id="P2",
        request_vector=[1, 300],
        max_claim_vector=[1, 400]
    )
    assert granted2 is True
    assert len(seq2) == 2


def test_banker_unsafe_state_prevention():
    """Verifies that an unsafe allocation is rejected and rolled back."""
    # System with total 2 slots and 500 MiB
    safeguard = SafeGuard(total_slots=2, total_memory_mb=500)

    # P1 claims max [1, 400], allocated [1, 300]. Available now: [1, 200]
    safeguard.evaluate_request("P1", [1, 300], [1, 400])

    # P2 claims max [1, 300], requests [1, 200].
    # If granted, Available would be [0, 0].
    # But P1 needs 100 more, P2 needs 100 more. Neither can finish!
    # This is UNSAFE. SafeGuard should reject and defer.
    granted, reason, seq = safeguard.evaluate_request("P2", [1, 200], [1, 300])
    assert granted is False
    assert "UNSAFE" in reason or "deferred" in reason
    # Verify rollback: Available must still be [1, 200]
    assert safeguard.available == [1, 200]


def test_banker_resource_release():
    safeguard = SafeGuard(total_slots=4, total_memory_mb=1000)
    safeguard.evaluate_request("P1", [1, 250], [1, 250])
    assert safeguard.available == [3, 750]

    freed = safeguard.release_request("P1")
    assert freed == [1, 250]
    assert safeguard.available == [4, 1000]


# ==============================================================================
# 3. MEMORABRAIN & MEMORAEXPLAIN
# ==============================================================================

def test_memorabrain_critical_headroom():
    brain = MemoraBrain(memory_headroom_percent=15.0)
    # Host RAM usage at 92% (> 85% threshold)
    advice = brain.evaluate(
        system_ram_percent=92.0,
        system_available_mb=400.0,
        process_rss_mb=500.0,
        active_requests=1,
        max_concurrent=4,
        pool_fragmentation=0.1,
        logical_free_mb=500,
        queue_len=0
    )
    assert advice["recommended_action"] == "DEFER"
    assert advice["severity"] == "CRITICAL"


def test_memora_explain_logging():
    explain = MemoraExplain(max_history=5)
    explain.record_decision("req_1", "ADMITTED", "Safe state verified", {"est_mb": 150})
    logs = explain.get_latest_logs(10)
    assert len(logs) == 1
    assert logs[0]["request_id"] == "req_1"
    assert logs[0]["decision"] == "ADMITTED"


# ==============================================================================
# 4. PASMA+ ADMISSION CONTROLLER
# ==============================================================================

def test_pasma_admission_and_release_lifecycle():
    controller = PASMAAdmissionController(pool_capacity_mb=1024, max_workers=2)

    admitted, reason, block = controller.evaluate_admission(
        request_id="test_req_01",
        prompt="Explain virtual memory paging.",
        max_tokens=64,
        system_ram_percent=40.0,
        system_available_mb=4000.0
    )
    assert admitted is True
    assert block is not None
    assert block.status == "ALLOCATED"
    assert controller.safeguard.available[0] == 1  # 1 slot used

    # Release
    rel = controller.release_request("test_req_01")
    assert rel["freed_resources"][0] == 1
    assert controller.safeguard.available[0] == 2
    assert controller.pool.total_free_mb == 1024


# ==============================================================================
# 5. METRICS & COMPARISON BENCHMARKING
# ==============================================================================

def test_compare_benchmarks():
    norm = ExperimentResult(
        mode="NORMAL",
        timestamp=time.time(),
        total_requests=4,
        completed_requests=4,
        failed_requests=0,
        deferred_requests=0,
        wall_time_sec=10.0,
        avg_latency_sec=2.5,
        median_latency_sec=2.5,
        p95_latency_sec=2.9,
        avg_waiting_sec=0.1,
        peak_rss_mb=450.0,
        total_tokens=200,
        aggregate_tokens_per_sec=20.0,
        requests_data=[]
    )

    edge = ExperimentResult(
        mode="EDGEGUARD",
        timestamp=time.time(),
        total_requests=4,
        completed_requests=4,
        failed_requests=0,
        deferred_requests=1,
        wall_time_sec=9.5,
        avg_latency_sec=2.3,
        median_latency_sec=2.3,
        p95_latency_sec=2.6,
        avg_waiting_sec=0.2,
        peak_rss_mb=420.0,
        total_tokens=200,
        aggregate_tokens_per_sec=21.05,
        requests_data=[]
    )

    comp = compare_benchmarks(norm, edge)
    rows = comp["rows"]
    # Check Wall Time delta: ((9.5 - 10.0) / 10.0) * 100 = -5.0%
    wall_row = next(r for r in rows if "Wall Time" in r["metric"])
    assert wall_row["delta_pct"] == -5.0


# ==============================================================================
# 6. LLAMA CLIENT OFFLINE HANDLING (INTEGRATION HELPER)
# ==============================================================================

def test_llama_client_offline_graceful():
    """Client must handle unreachable server cleanly without unhandled exceptions."""
    client = LlamaCppClient(base_url="http://127.0.0.1:59999")
    healthy, info, msg = client.check_health(timeout_sec=0.5)
    assert healthy is False
    assert "refused" in msg.lower() or "unreachable" in msg.lower() or "timed out" in msg.lower()


# ==============================================================================
# 7. HETEROGENEOUS MULTIMODAL & VISION ADMISSION TESTS
# ==============================================================================

def test_heterogeneous_memory_estimation():
    controller = PASMAAdmissionController(pool_capacity_mb=1024, max_workers=4)
    text_est = controller.estimate_request_memory("Describe system load", max_tokens=64, task_type="TEXT")
    vision_est = controller.estimate_request_memory("Inspect high-res frame", max_tokens=64, task_type="VISION_MULTIMODAL")
    
    assert 128 <= text_est <= 256
    assert 384 <= vision_est <= 512
    assert vision_est > text_est


def test_heterogeneous_banker_admission_and_interleaving():
    """
    In a 1024 MiB pool with 2 worker slots:
    Admitting two 512 MiB Vision requests consumes 100% capacity.
    A third request must be safely deferred until resources are released.
    """
    controller = PASMAAdmissionController(pool_capacity_mb=1024, max_workers=2)

    # 1. Admit first vision task (512 MiB)
    adm1, r1, b1 = controller.evaluate_admission("v1", "Camera frame 1", max_tokens=128, task_type="VISION_MULTIMODAL")
    assert adm1 is True
    assert b1.size_mb >= 384

    # 2. Admit second vision task (512 MiB) -> consumes remaining pool
    adm2, r2, b2 = controller.evaluate_admission("v2", "Camera frame 2", max_tokens=128, task_type="VISION_MULTIMODAL")
    assert adm2 is True

    # 3. Third request arrives -> pool is full / slots full -> must DEFER safely
    adm3, r3, b3 = controller.evaluate_admission("t1", "Status text", max_tokens=64, task_type="TEXT")
    assert adm3 is False
    assert b3 is None

    # 4. Release v1 -> restores slot and memory
    controller.release_request("v1")

    # 5. Now t1 can be admitted safely!
    adm3_retry, r3_retry, b3_retry = controller.evaluate_admission("t1", "Status text", max_tokens=64, task_type="TEXT")
    assert adm3_retry is True
    assert b3_retry is not None


def test_edge_vision_yolo_detection():
    """
    Verifies that EdgeVisionEngine runs real YOLOv8 object detection on camera frames.
    Validates that:
    1. Both Mode A and Mode B detect objects accurately.
    2. Mode B with PASMA+ reservation is significantly faster than Mode A with unmanaged memory delay.
    """
    from vision import EdgeVisionEngine
    import os
    if not os.path.exists("sample_edge_frame.jpg"):
        return  # Skip if sample image missing

    engine = EdgeVisionEngine("yolov8n.pt")
    controller = PASMAAdmissionController(pool_capacity_mb=1024, max_workers=2)

    res_edgeguard = engine.run_detection("sample_edge_frame.jpg", mode="EDGEGUARD", controller=controller)
    assert res_edgeguard["success"] is True
    assert res_edgeguard["objects_count"] >= 1
    assert res_edgeguard["annotated_image"] is not None
    assert res_edgeguard["total_time_sec"] < 2.0  # Ultra-fast execution

    res_normal = engine.run_detection("sample_edge_frame.jpg", mode="NORMAL")
    assert res_normal["success"] is True
    assert res_normal["objects_count"] == res_edgeguard["objects_count"]  # Accuracy parity
    assert res_normal["total_time_sec"] > res_edgeguard["total_time_sec"]  # Mode A suffers OS contention delay


