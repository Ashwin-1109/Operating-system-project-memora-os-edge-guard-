"""
MemoraOS EdgeGuard — Real-Time Edge AI Memory Management
Core Resource Admission & Algorithmic Memory Management Subsystem

Components:
1. LogicalMemoryPool: Simulated memory pool with contiguous block management and coalescing.
2. PULSE-Fit: Adaptive allocation policy selector (First Fit, Best Fit, Worst Fit, Next Fit).
3. SAFE-Guard: Multi-resource Banker's Algorithm implementation for provable safe-state admission.
4. MemoraBrain: Lightweight deterministic rule-based resource advisor.
5. MemoraExplain: Transparent mathematical decision auditing and explanation generator.
6. PASMAAdmissionController: Master placement-aware safe memory admission controller.
"""

from __future__ import annotations
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class MemoryBlock:
    """Represents a logical memory partition within the managed pool."""
    block_id: int
    start_mb: int
    size_mb: int
    status: str  # 'FREE', 'ALLOCATED', 'RESERVED'
    request_id: Optional[str] = None
    allocated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "block_id": self.block_id,
            "start_mb": self.start_mb,
            "size_mb": self.size_mb,
            "status": self.status,
            "request_id": self.request_id,
            "allocated_at": self.allocated_at,
        }


# ==============================================================================
# 1. LOGICAL MEMORY POOL (CONTIGUOUS PARTITIONING & COALESCING)
# ==============================================================================

class LogicalMemoryPool:
    """
    Simulates a contiguous logical memory pool partitioned into blocks.
    Tracks internal/external fragmentation, allocates using classic OS placement
    strategies, and coalesces adjacent free blocks upon release.
    """

    def __init__(self, capacity_mb: int = 1024):
        self.capacity_mb = capacity_mb
        self._next_block_id = 1
        # Initialize pool with a single free block encompassing full capacity
        self.blocks: List[MemoryBlock] = [
            MemoryBlock(
                block_id=self._get_next_id(),
                start_mb=0,
                size_mb=capacity_mb,
                status="FREE",
                request_id=None
            )
        ]
        self.last_allocation_index: int = 0
        self._lock = threading.Lock()

    def _get_next_id(self) -> int:
        bid = self._next_block_id
        self._next_block_id += 1
        return bid

    def reset(self):
        """Reset pool to pristine state."""
        with self._lock:
            self._next_block_id = 1
            self.blocks = [
                MemoryBlock(
                    block_id=self._get_next_id(),
                    start_mb=0,
                    size_mb=self.capacity_mb,
                    status="FREE",
                    request_id=None
                )
            ]
            self.last_allocation_index = 0

    @property
    def total_free_mb(self) -> int:
        return sum(b.size_mb for b in self.blocks if b.status == "FREE")

    @property
    def total_allocated_mb(self) -> int:
        return sum(b.size_mb for b in self.blocks if b.status == "ALLOCATED")

    @property
    def total_reserved_mb(self) -> int:
        return sum(b.size_mb for b in self.blocks if b.status == "RESERVED")

    @property
    def external_fragmentation(self) -> float:
        """
        External Fragmentation metric EF:
        EF = 1.0 - (Largest_Free_Block / Total_Free_Memory)
        When total free is 0 or all free is contiguous, EF is 0.0.
        When free memory is chopped into many small disjoint blocks, EF approaches 1.0.
        """
        free_blocks = [b.size_mb for b in self.blocks if b.status == "FREE"]
        if not free_blocks or sum(free_blocks) == 0:
            return 0.0
        largest = max(free_blocks)
        total_free = sum(free_blocks)
        return round(1.0 - (largest / total_free), 4)

    def get_blocks_snapshot(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [b.to_dict() for b in self.blocks]

    def _split_and_allocate(self, block_index: int, size_mb: int, request_id: str) -> MemoryBlock:
        """Helper to split a free block if larger than requested, or occupy it."""
        target = self.blocks[block_index]
        if target.size_mb == size_mb:
            target.status = "ALLOCATED"
            target.request_id = request_id
            target.allocated_at = time.time()
            return target

        # Split block: allocate first part, leave remainder as FREE
        rem_size = target.size_mb - size_mb
        allocated_block = MemoryBlock(
            block_id=self._get_next_id(),
            start_mb=target.start_mb,
            size_mb=size_mb,
            status="ALLOCATED",
            request_id=request_id,
            allocated_at=time.time()
        )
        remainder_block = MemoryBlock(
            block_id=self._get_next_id(),
            start_mb=target.start_mb + size_mb,
            size_mb=rem_size,
            status="FREE",
            request_id=None
        )
        self.blocks[block_index] = allocated_block
        self.blocks.insert(block_index + 1, remainder_block)
        return allocated_block

    def allocate_first_fit(self, size_mb: int, request_id: str) -> Optional[MemoryBlock]:
        """First Fit: Finds the first free block with size >= size_mb."""
        with self._lock:
            for i, b in enumerate(self.blocks):
                if b.status == "FREE" and b.size_mb >= size_mb:
                    self.last_allocation_index = i
                    return self._split_and_allocate(i, size_mb, request_id)
            return None

    def allocate_best_fit(self, size_mb: int, request_id: str) -> Optional[MemoryBlock]:
        """Best Fit: Finds the free block with the minimum size >= size_mb (leaves smallest remnant)."""
        with self._lock:
            best_idx = -1
            best_diff = float("inf")
            for i, b in enumerate(self.blocks):
                if b.status == "FREE" and b.size_mb >= size_mb:
                    diff = b.size_mb - size_mb
                    if diff < best_diff:
                        best_diff = diff
                        best_idx = i
            if best_idx != -1:
                self.last_allocation_index = best_idx
                return self._split_and_allocate(best_idx, size_mb, request_id)
            return None

    def allocate_worst_fit(self, size_mb: int, request_id: str) -> Optional[MemoryBlock]:
        """Worst Fit: Finds the free block with the maximum size >= size_mb (leaves largest remnant)."""
        with self._lock:
            worst_idx = -1
            max_size = -1
            for i, b in enumerate(self.blocks):
                if b.status == "FREE" and b.size_mb >= size_mb:
                    if b.size_mb > max_size:
                        max_size = b.size_mb
                        worst_idx = i
            if worst_idx != -1:
                self.last_allocation_index = worst_idx
                return self._split_and_allocate(worst_idx, size_mb, request_id)
            return None

    def allocate_next_fit(self, size_mb: int, request_id: str) -> Optional[MemoryBlock]:
        """Next Fit: Circular search starting from the last allocation index."""
        with self._lock:
            n = len(self.blocks)
            if n == 0:
                return None
            start = self.last_allocation_index % n
            for offset in range(n):
                idx = (start + offset) % n
                b = self.blocks[idx]
                if b.status == "FREE" and b.size_mb >= size_mb:
                    self.last_allocation_index = idx
                    return self._split_and_allocate(idx, size_mb, request_id)
            return None

    def release(self, request_id: str) -> int:
        """
        Releases all blocks allocated to request_id and coalesces adjacent FREE blocks.
        Returns the number of MiB freed.
        """
        with self._lock:
            freed_mb = 0
            for b in self.blocks:
                if b.request_id == request_id:
                    b.status = "FREE"
                    b.request_id = None
                    freed_mb += b.size_mb

            if freed_mb > 0:
                self._coalesce_free_blocks()
            return freed_mb

    def _coalesce_free_blocks(self):
        """Merges contiguous FREE blocks into single unified blocks."""
        if len(self.blocks) <= 1:
            return
        coalesced: List[MemoryBlock] = []
        current = self.blocks[0]

        for nxt in self.blocks[1:]:
            if current.status == "FREE" and nxt.status == "FREE":
                # Merge into current
                current = MemoryBlock(
                    block_id=current.block_id,
                    start_mb=current.start_mb,
                    size_mb=current.size_mb + nxt.size_mb,
                    status="FREE",
                    request_id=None
                )
            else:
                coalesced.append(current)
                current = nxt
        coalesced.append(current)
        self.blocks = coalesced
        if self.last_allocation_index >= len(self.blocks):
            self.last_allocation_index = 0


# ==============================================================================
# 2. PULSE-FIT: ADAPTIVE ALLOCATION SELECTION
# ==============================================================================

class PULSEFit:
    """
    PULSE-Fit evaluates the logical pool's external fragmentation, request size,
    and free block distribution to select the optimal placement strategy:
    First Fit, Best Fit, Worst Fit, or Next Fit.
    """

    def __init__(self, pool: LogicalMemoryPool):
        self.pool = pool
        self.recent_allocations: List[Dict[str, Any]] = []

    def select_strategy(self, request_size_mb: int, manual_override: Optional[str] = None) -> str:
        """
        Adaptive policy selector:
        - If manual_override is provided and valid, respects user choice.
        - Otherwise evaluates:
            1. Pool fragmentation: High fragmentation (EF > 0.30) requires Best Fit
               to preserve remaining large blocks and fit into existing gaps.
            2. Large requests (> 40% of total free): Best Fit to avoid splitting huge blocks.
            3. Low fragmentation (EF < 0.15) & small requests: First Fit for minimal search overhead.
            4. Moderate fragmentation with dispersed blocks: Next Fit for uniform wear.
        """
        if manual_override in ("FIRST_FIT", "BEST_FIT", "WORST_FIT", "NEXT_FIT"):
            return manual_override

        frag = self.pool.external_fragmentation
        total_free = self.pool.total_free_mb

        if frag >= 0.30 or (total_free > 0 and (request_size_mb / total_free) > 0.40):
            return "BEST_FIT"
        elif frag < 0.15:
            return "FIRST_FIT"
        elif len([b for b in self.pool.blocks if b.status == "FREE"]) > 3:
            return "NEXT_FIT"
        else:
            return "BEST_FIT"

    def allocate(self, size_mb: int, request_id: str, manual_override: Optional[str] = None) -> Tuple[Optional[MemoryBlock], Dict[str, Any]]:
        """
        Executes placement using the chosen strategy and captures detailed telemetry.
        """
        frag_before = self.pool.external_fragmentation
        free_before = self.pool.total_free_mb
        strategy = self.select_strategy(size_mb, manual_override)

        block: Optional[MemoryBlock] = None
        if strategy == "BEST_FIT":
            block = self.pool.allocate_best_fit(size_mb, request_id)
        elif strategy == "WORST_FIT":
            block = self.pool.allocate_worst_fit(size_mb, request_id)
        elif strategy == "NEXT_FIT":
            block = self.pool.allocate_next_fit(size_mb, request_id)
        else:
            strategy = "FIRST_FIT"
            block = self.pool.allocate_first_fit(size_mb, request_id)

        frag_after = self.pool.external_fragmentation
        free_after = self.pool.total_free_mb

        trace = {
            "strategy": strategy,
            "success": block is not None,
            "block_id": block.block_id if block else None,
            "start_mb": block.start_mb if block else None,
            "size_mb": size_mb,
            "fragmentation_before": frag_before,
            "fragmentation_after": frag_after,
            "free_space_before": free_before,
            "remaining_free": free_after,
            "request_id": request_id,
            "timestamp": time.time(),
        }
        self.recent_allocations.append(trace)
        if len(self.recent_allocations) > 50:
            self.recent_allocations.pop(0)

        return block, trace


# ==============================================================================
# 3. SAFE-GUARD: RESOURCE SAFETY (BANKER'S ALGORITHM)
# ==============================================================================

class SafeGuard:
    """
    Implements Dijkstra's classic Banker's Algorithm for multiple managed resource types:
    Resource 0: Active Inference Worker Slots (e.g., max concurrent slots = 4)
    Resource 1: Logical Memory Reservations (e.g., max pool = 1024 MiB)

    Maintains:
    - Available Vector: A = [available_slots, available_memory_mb]
    - Max Matrix: M[pid] = [max_slots, max_memory_mb]
    - Allocation Matrix: Alloc[pid] = [alloc_slots, alloc_memory_mb]
    - Need Matrix: Need[pid] = Max[pid] - Alloc[pid]
    """

    def __init__(self, total_slots: int = 4, total_memory_mb: int = 1024):
        self.total_resources = [total_slots, total_memory_mb]
        self.available = [total_slots, total_memory_mb]
        self.max_matrix: Dict[str, List[int]] = {}
        self.allocation_matrix: Dict[str, List[int]] = {}
        self.need_matrix: Dict[str, List[int]] = {}
        self._lock = threading.Lock()

    def reset(self, total_slots: int = 4, total_memory_mb: int = 1024):
        with self._lock:
            self.total_resources = [total_slots, total_memory_mb]
            self.available = [total_slots, total_memory_mb]
            self.max_matrix.clear()
            self.allocation_matrix.clear()
            self.need_matrix.clear()

    def is_safe_state(self) -> Tuple[bool, List[str], str]:
        """
        Executes Banker's Safety Algorithm:
        Let Work = Available[:]
        Let Finish = {p: False for p in active_processes}
        Find p such that Finish[p] == False and Need[p] <= Work
        If found: Work += Alloc[p], Finish[p] = True, append p to safe sequence.
        Returns: (is_safe, safe_sequence, explanation)
        """
        work = self.available[:]
        active_processes = list(self.max_matrix.keys())
        finish = {p: False for p in active_processes}
        safe_sequence: List[str] = []

        while True:
            found = False
            for p in active_processes:
                if not finish[p]:
                    # Check if Need[p] <= Work across all resource dimensions
                    need_p = self.need_matrix[p]
                    if all(need_p[j] <= work[j] for j in range(len(work))):
                        # Process can finish and release its allocated resources
                        alloc_p = self.allocation_matrix[p]
                        for j in range(len(work)):
                            work[j] += alloc_p[j]
                        finish[p] = True
                        safe_sequence.append(p)
                        found = True
                        break  # restart search
            if not found:
                break

        if all(finish.values()):
            seq_str = " -> ".join(safe_sequence) if safe_sequence else "Empty (No active processes)"
            return True, safe_sequence, f"State is SAFE. Safe Sequence: <{seq_str}>"
        else:
            unfinished = [p for p, f in finish.items() if not f]
            return False, [], f"State is UNSAFE. Processes unable to satisfy maximum need: {unfinished}"

    def evaluate_request(
        self,
        request_id: str,
        request_vector: List[int],
        max_claim_vector: Optional[List[int]] = None
    ) -> Tuple[bool, str, Optional[List[str]]]:
        """
        Tests whether granting request_vector = [slots, memory_mb] leaves the system in a safe state.
        If yes, commits allocation and returns (True, reason, safe_sequence).
        If no, rolls back and returns (False, reason, None).
        """
        with self._lock:
            # If new process, register its max claim
            if request_id not in self.max_matrix:
                if max_claim_vector is None:
                    max_claim_vector = request_vector[:]
                # Max claim cannot exceed total system resources
                for j in range(len(self.total_resources)):
                    if max_claim_vector[j] > self.total_resources[j]:
                        return False, f"Process max claim {max_claim_vector} exceeds total system capacity {self.total_resources}.", None
                self.max_matrix[request_id] = max_claim_vector[:]
                self.allocation_matrix[request_id] = [0] * len(self.total_resources)
                self.need_matrix[request_id] = max_claim_vector[:]

            need = self.need_matrix[request_id]
            # Condition 1: Request <= Need
            for j in range(len(request_vector)):
                if request_vector[j] > need[j]:
                    return False, f"Request {request_vector} exceeds process remaining Need {need}.", None

            # Condition 2: Request <= Available
            for j in range(len(request_vector)):
                if request_vector[j] > self.available[j]:
                    return False, f"Request {request_vector} exceeds currently Available resources {self.available}.", None

            # Provisional allocation
            for j in range(len(request_vector)):
                self.available[j] -= request_vector[j]
                self.allocation_matrix[request_id][j] += request_vector[j]
                self.need_matrix[request_id][j] -= request_vector[j]

            # Check safety of provisional state
            safe, seq, reason = self.is_safe_state()

            if safe:
                # Commit allocation
                return True, f"SAFE: Grant approved. {reason}", seq
            else:
                # Rollback provisional grant
                for j in range(len(request_vector)):
                    self.available[j] += request_vector[j]
                    self.allocation_matrix[request_id][j] -= request_vector[j]
                    self.need_matrix[request_id][j] += request_vector[j]
                # If newly added process with 0 allocation, remove it
                if all(v == 0 for v in self.allocation_matrix[request_id]):
                    del self.max_matrix[request_id]
                    del self.allocation_matrix[request_id]
                    del self.need_matrix[request_id]
                return False, f"UNSAFE: Grant deferred. Provisional state leads to potential deadlock: {reason}", None

    def release_request(self, request_id: str) -> List[int]:
        """Reclaims allocated resources for a completed request."""
        with self._lock:
            if request_id not in self.allocation_matrix:
                return [0, 0]
            freed = self.allocation_matrix[request_id][:]
            for j in range(len(self.available)):
                self.available[j] += freed[j]
            del self.max_matrix[request_id]
            del self.allocation_matrix[request_id]
            del self.need_matrix[request_id]
            return freed

    def get_matrices_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "available": self.available[:],
                "total": self.total_resources[:],
                "max_matrix": {k: v[:] for k, v in self.max_matrix.items()},
                "allocation_matrix": {k: v[:] for k, v in self.allocation_matrix.items()},
                "need_matrix": {k: v[:] for k, v in self.need_matrix.items()},
            }


# ==============================================================================
# 4. MEMORABRAIN: LIGHTWEIGHT RESOURCE ADVISOR
# ==============================================================================

class MemoraBrain:
    """
    Lightweight, deterministic rule-based resource advisor.
    Evaluates:
    - Real measured RAM usage & available system RAM (via psutil)
    - Active inference workers & queue length
    - Logical pool fragmentation & available capacity
    - Safety headroom constraints
    Produces actionable recommendations for admission policy without bypassing SAFE-Guard.
    """

    def __init__(self, memory_headroom_percent: float = 15.0):
        self.memory_headroom_percent = memory_headroom_percent

    def evaluate(
        self,
        system_ram_percent: float,
        system_available_mb: float,
        process_rss_mb: float,
        active_requests: int,
        max_concurrent: int,
        pool_fragmentation: float,
        logical_free_mb: int,
        queue_len: int
    ) -> Dict[str, Any]:
        """
        Deterministic rule engine:
        1. CRITICAL_HEADROOM: If system RAM > (100 - headroom), system is near thrashing.
        2. CONCURRENCY_SATURATED: If active requests >= max_concurrent, throttle new admissions.
        3. HIGH_FRAGMENTATION: If pool fragmentation > 0.35, advise Best Fit compaction.
        4. CAPACITY_EXHAUSTION: If logical free < 100 MiB, advise deferring.
        5. HEALTHY: Safe operating envelope.
        """
        advice = []
        action = "ADMIT"
        severity = "NORMAL"

        if system_ram_percent >= (100.0 - self.memory_headroom_percent) or system_available_mb < 600.0:
            action = "DEFER"
            severity = "CRITICAL"
            advice.append(f"Host RAM pressure high ({system_ram_percent:.1f}% used, {system_available_mb:.0f} MiB free). Throttle to avoid OS paging.")
        elif active_requests >= max_concurrent:
            action = "DEFER"
            severity = "WARNING"
            advice.append(f"Inference concurrency saturated ({active_requests}/{max_concurrent} active slots). Queue incoming requests.")
        elif logical_free_mb < 128:
            action = "DEFER"
            severity = "WARNING"
            advice.append(f"Logical memory pool nearly depleted ({logical_free_mb} MiB remaining). Await workload completion.")

        if pool_fragmentation >= 0.35:
            advice.append(f"Pool external fragmentation elevated ({pool_fragmentation * 100:.1f}%). Recommending Best Fit placement.")

        if not advice:
            advice.append("Resource margins healthy. Request admission approved.")

        return {
            "recommended_action": action,
            "severity": severity,
            "reasons": advice,
            "system_ram_percent": system_ram_percent,
            "active_requests": active_requests,
            "logical_free_mb": logical_free_mb,
            "fragmentation": pool_fragmentation,
        }


# ==============================================================================
# 5. MEMORAEXPLAIN: DECISION AUDITOR & EXPLANATION GENERATOR
# ==============================================================================

class MemoraExplain:
    """
    Maintains a verifiable audit log of every allocation, deferral, and release
    event with human-readable rationale grounded in actual mathematical telemetry.
    """

    def __init__(self, max_history: int = 100):
        self.logs: List[Dict[str, Any]] = []
        self.max_history = max_history
        self._lock = threading.Lock()

    def record_decision(
        self,
        request_id: str,
        decision: str,  # 'ADMITTED', 'DEFERRED', 'REJECTED', 'RELEASED'
        reason: str,
        details: Dict[str, Any]
    ):
        with self._lock:
            entry = {
                "timestamp": time.time(),
                "time_str": time.strftime("%H:%M:%S"),
                "request_id": request_id,
                "decision": decision,
                "reason": reason,
                "details": details,
            }
            self.logs.append(entry)
            if len(self.logs) > self.max_history:
                self.logs.pop(0)

    def get_latest_logs(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            return list(reversed(self.logs[-limit:]))

    def clear(self):
        with self._lock:
            self.logs.clear()


# ==============================================================================
# 6. PASMA+: PLACEMENT-AWARE SAFE MEMORY ADMISSION CONTROLLER
# ==============================================================================

class PASMAAdmissionController:
    """
    Central admission controller integrating:
    - LogicalMemoryPool (Contiguous block tracking & fragmentation)
    - PULSE-Fit (Adaptive allocation policy)
    - SAFE-Guard (Banker's Algorithm safe-state verification)
    - MemoraBrain (Host telemetry advisor)
    - MemoraExplain (Decision tracing)
    """

    def __init__(self, pool_capacity_mb: int = 1024, max_workers: int = 4):
        self.pool_capacity_mb = pool_capacity_mb
        self.max_workers = max_workers
        self.pool = LogicalMemoryPool(capacity_mb=pool_capacity_mb)
        self.pulse = PULSEFit(self.pool)
        self.safeguard = SafeGuard(total_slots=max_workers, total_memory_mb=pool_capacity_mb)
        self.brain = MemoraBrain(memory_headroom_percent=15.0)
        self.explain = MemoraExplain(max_history=100)
        self.peak_snapshot: Optional[Dict[str, Any]] = None
        self._lock = threading.Lock()

    def reset(self):
        with self._lock:
            self.pool.reset()
            self.safeguard.reset(total_slots=self.max_workers, total_memory_mb=self.pool_capacity_mb)
            self.explain.clear()

    def estimate_request_memory(self, prompt: str, max_tokens: int, task_type: str = "TEXT") -> int:
        """
        Estimates logical memory reservation required for the Edge AI request.
        - TEXT: Base KV cache overhead (96 MiB) + token buffer -> 128 to 256 MiB.
        - VISION_MULTIMODAL: High-resolution image/vision frame buffer (patch embeddings,
          vision encoder projection, KV context) -> 384 to 512 MiB.
        """
        if task_type == "VISION_MULTIMODAL":
            approx_tokens = len(prompt.split()) * 1.3 + max_tokens
            estimated = 384 + int(approx_tokens * 0.20)
            return max(384, min(512, estimated))
        else:
            approx_tokens = len(prompt.split()) * 1.3 + max_tokens
            estimated = 96 + int(approx_tokens * 0.15)
            return max(128, min(256, estimated))

    def evaluate_admission(
        self,
        request_id: str,
        prompt: str,
        max_tokens: int,
        task_type: str = "TEXT",
        system_ram_percent: float = 50.0,
        system_available_mb: float = 4096.0,
        process_rss_mb: float = 300.0,
        manual_placement_override: Optional[str] = None
    ) -> Tuple[bool, str, Optional[MemoryBlock]]:
        """
        PASMA+ Admission Pipeline:
        1. Estimate request memory reservation based on task type (TEXT vs VISION_MULTIMODAL).
        2. Consult MemoraBrain advisor on host telemetry.
        3. Check SafeGuard Banker's Algorithm with resource vector [1 slot, estimated_mb].
        4. If safe, execute PULSE-Fit placement into LogicalMemoryPool.
        5. Record complete audit trail in MemoraExplain.
        """
        with self._lock:
            est_mb = self.estimate_request_memory(prompt, max_tokens, task_type)
            active_workers = self.max_workers - self.safeguard.available[0]

            # Step 1: MemoraBrain Advisory Check
            brain_advice = self.brain.evaluate(
                system_ram_percent=system_ram_percent,
                system_available_mb=system_available_mb,
                process_rss_mb=process_rss_mb,
                active_requests=active_workers,
                max_concurrent=self.max_workers,
                pool_fragmentation=self.pool.external_fragmentation,
                logical_free_mb=self.pool.total_free_mb,
                queue_len=0
            )

            # If host RAM is critically exhausted, defer immediately
            if brain_advice["recommended_action"] == "DEFER" and brain_advice["severity"] == "CRITICAL":
                reason = f"[{task_type}] MemoraBrain deferred request: {'; '.join(brain_advice['reasons'])}"
                self.explain.record_decision(
                    request_id=request_id,
                    decision="DEFERRED",
                    reason=reason,
                    details={"task_type": task_type, "brain": brain_advice, "estimated_mb": est_mb}
                )
                return False, reason, None

            # Step 2: SAFE-Guard Banker's Algorithm Check
            # Resource vector: [1 slot, est_mb memory]
            request_vec = [1, est_mb]
            max_claim_vec = [1, est_mb]
            safe_granted, banker_reason, safe_seq = self.safeguard.evaluate_request(
                request_id=request_id,
                request_vector=request_vec,
                max_claim_vector=max_claim_vec
            )

            if not safe_granted:
                reason = f"[{task_type}] SAFE-Guard deferred request: {banker_reason}"
                self.explain.record_decision(
                    request_id=request_id,
                    decision="DEFERRED",
                    reason=reason,
                    details={"task_type": task_type, "safeguard_reason": banker_reason, "estimated_mb": est_mb}
                )
                return False, reason, None

            # Step 3: PULSE-Fit Logical Placement
            block, pulse_trace = self.pulse.allocate(
                size_mb=est_mb,
                request_id=request_id,
                manual_override=manual_placement_override
            )

            if block is None:
                # Pool placement failed despite Banker's vector (due to fragmentation)
                # Rollback Banker's allocation
                self.safeguard.release_request(request_id)
                reason = (
                    f"[{task_type}] PULSE-Fit could not place {est_mb} MiB block due to logical memory fragmentation "
                    f"({self.pool.external_fragmentation * 100:.1f}% external fragmentation, "
                    f"{self.pool.total_free_mb} MiB total free)."
                )
                self.explain.record_decision(
                    request_id=request_id,
                    decision="DEFERRED",
                    reason=reason,
                    details={"task_type": task_type, "pulse_trace": pulse_trace, "estimated_mb": est_mb}
                )
                return False, reason, None

            # Step 4: Admission Successful
            reason = (
                f"[{task_type}] Admitted via {pulse_trace['strategy']}: Reserved block #{block.block_id} "
                f"({est_mb} MiB @ offset {block.start_mb} MiB). Safe Sequence verified."
            )
            self.explain.record_decision(
                request_id=request_id,
                decision="ADMITTED",
                reason=reason,
                details={
                    "task_type": task_type,
                    "pulse_trace": pulse_trace,
                    "safe_sequence": safe_seq,
                    "brain": brain_advice,
                    "estimated_mb": est_mb,
                    "block_id": block.block_id,
                }
            )

            # Preserve peak allocation snapshot for Section 3 visualization
            cur_alloc = self.pool.total_allocated_mb
            if self.peak_snapshot is None or cur_alloc >= self.peak_snapshot.get("pool_allocated_mb", 0):
                self.peak_snapshot = {
                    "pool_capacity_mb": self.pool.capacity_mb,
                    "pool_free_mb": self.pool.total_free_mb,
                    "pool_allocated_mb": self.pool.total_allocated_mb,
                    "pool_fragmentation": self.pool.external_fragmentation,
                    "blocks": self.pool.get_blocks_snapshot(),
                    "safeguard": self.safeguard.get_matrices_snapshot(),
                    "available_slots": self.safeguard.available[0],
                    "active_slots": self.max_workers - self.safeguard.available[0],
                    "max_workers": self.max_workers,
                }

            return True, reason, block

    def release_request(self, request_id: str) -> Dict[str, Any]:
        """Reclaims logical memory block and Banker's worker slot."""
        with self._lock:
            freed_pool_mb = self.pool.release(request_id)
            freed_resources = self.safeguard.release_request(request_id)
            reason = f"Released request {request_id}. Restored {freed_pool_mb} MiB logical pool & {freed_resources[0]} worker slot."
            self.explain.record_decision(
                request_id=request_id,
                decision="RELEASED",
                reason=reason,
                details={"freed_pool_mb": freed_pool_mb, "freed_resources": freed_resources}
            )
            return {"freed_pool_mb": freed_pool_mb, "freed_resources": freed_resources}

    def get_telemetry_snapshot(self) -> Dict[str, Any]:
        """Provides full snapshot of admission subsystem state for visualization."""
        with self._lock:
            if self.pool.total_allocated_mb > 0 or not self.peak_snapshot:
                return {
                    "pool_capacity_mb": self.pool.capacity_mb,
                    "pool_free_mb": self.pool.total_free_mb,
                    "pool_allocated_mb": self.pool.total_allocated_mb,
                    "pool_fragmentation": self.pool.external_fragmentation,
                    "blocks": self.pool.get_blocks_snapshot(),
                    "safeguard": self.safeguard.get_matrices_snapshot(),
                    "available_slots": self.safeguard.available[0],
                    "active_slots": self.max_workers - self.safeguard.available[0],
                    "max_workers": self.max_workers,
                    "recent_logs": self.explain.get_latest_logs(15),
                }
            else:
                snap = dict(self.peak_snapshot)
                snap["recent_logs"] = self.explain.get_latest_logs(15)
                return snap
