from __future__ import annotations

import os
import threading
import time
from typing import List, Optional, Tuple

try:
    from codecarbon import EmissionsTracker
    HAVE_CODECARBON = True
except ImportError:
    HAVE_CODECARBON = False

try:
    import psutil
    HAVE_PSUTIL = True
except ImportError:
    HAVE_PSUTIL = False

try:
    import pynvml
    pynvml.nvmlInit()
    HAVE_PYNVML = True
except Exception:
    HAVE_PYNVML = False

from MLAgentBench.schema import EmissionsMetrics, UtilizationMetrics


def _extract_emissions(tracker) -> EmissionsMetrics:
    d = tracker.final_emissions_data
    return EmissionsMetrics(
        emissions_kg=float(getattr(d, "emissions", 0.0) or 0.0),
        energy_kwh=float(getattr(d, "energy_consumed", 0.0) or 0.0),
        cpu_energy_kwh=float(getattr(d, "cpu_energy", 0.0) or 0.0),
        gpu_energy_kwh=float(getattr(d, "gpu_energy", 0.0) or 0.0),
        ram_energy_kwh=float(getattr(d, "ram_energy", 0.0) or 0.0),
        duration_s=float(getattr(d, "duration", 0.0) or 0.0),
    )


class _UtilizationSampler:
    """Background thread that polls CPU, RAM, GPU utilization at fixed intervals."""

    def __init__(self, device_index: int = 0, poll_interval: float = 5.0):
        self._poll_interval = poll_interval
        self._snapshots: List[dict] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._gpu_handle = None
        if HAVE_PYNVML:
            try:
                self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            except Exception:
                pass

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> Optional[UtilizationMetrics]:
        if self._thread:
            self._stop_event.set()
            self._thread.join(timeout=self._poll_interval + 1.0)
        return self._summarize()

    def _run(self) -> None:
        while not self._stop_event.wait(self._poll_interval):
            snap = self._sample()
            if snap:
                self._snapshots.append(snap)

    def _sample(self) -> Optional[dict]:
        if not HAVE_PSUTIL:
            return None

        mem = psutil.virtual_memory()
        snap = {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "ram_used_gb": mem.used / (1024 ** 3),
            "ram_percent": mem.percent,
            "gpu_util_percent": None,
            "vram_used_gb": None,
            "vram_percent": None,
        }

        if self._gpu_handle:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                snap["gpu_util_percent"] = float(util.gpu)
                snap["vram_used_gb"] = mem_info.used / (1024 ** 3)
                snap["vram_percent"] = (mem_info.used / mem_info.total) * 100.0
            except Exception:
                pass

        return snap

    def _summarize(self) -> Optional[UtilizationMetrics]:
        if not self._snapshots:
            return None

        def _mean(vals): return sum(vals) / len(vals)

        cpu = [s["cpu_percent"] for s in self._snapshots]
        ram = [s["ram_used_gb"] for s in self._snapshots]
        gpu = [s["gpu_util_percent"] for s in self._snapshots if s["gpu_util_percent"] is not None]
        vram = [s["vram_used_gb"] for s in self._snapshots if s["vram_used_gb"] is not None]

        return UtilizationMetrics(
            cpu_mean=_mean(cpu),
            cpu_max=max(cpu),
            ram_mean_gb=_mean(ram),
            ram_max_gb=max(ram),
            gpu_mean=_mean(gpu) if gpu else None,
            gpu_max=max(gpu) if gpu else None,
            vram_mean_gb=_mean(vram) if vram else None,
            vram_max_gb=max(vram) if vram else None,
            sample_count=len(self._snapshots),
        )


class AgentCarbonTracker:
    """Per-worker tracker: CodeCarbon energy + psutil/pynvml utilization sampling."""

    def __init__(
        self,
        worker_id: str,
        log_dir: str,
        enabled: bool = True,
        device_index: int = 0,
        poll_interval: float = 5.0,
    ):
        self._emissions_tracker = None
        self._sampler = _UtilizationSampler(device_index=device_index, poll_interval=poll_interval)
        self._enabled = enabled

        if enabled and HAVE_CODECARBON:
            os.makedirs(log_dir, exist_ok=True)
            self._emissions_tracker = EmissionsTracker(
                project_name=f"worker_{worker_id}",
                output_dir=log_dir,
                save_to_file=True,
                log_level="error",
            )

    def start(self) -> None:
        if self._emissions_tracker:
            self._emissions_tracker.start()
        if self._enabled:
            self._sampler.start()

    def stop(self) -> Tuple[Optional[EmissionsMetrics], Optional[UtilizationMetrics]]:
        emissions = None
        if self._emissions_tracker:
            try:
                self._emissions_tracker.stop()
                emissions = _extract_emissions(self._emissions_tracker)
            except Exception:
                pass

        utilization = self._sampler.stop() if self._enabled else None
        return emissions, utilization


class SystemCarbonTracker:
    """Experiment-wide tracker: CodeCarbon energy + psutil/pynvml utilization sampling."""

    def __init__(
        self,
        log_dir: str,
        task: str,
        enabled: bool = True,
        device_index: int = 0,
        poll_interval: float = 10.0,
    ):
        self._emissions_tracker = None
        self._sampler = _UtilizationSampler(device_index=device_index, poll_interval=poll_interval)
        self._enabled = enabled

        if enabled and HAVE_CODECARBON:
            os.makedirs(log_dir, exist_ok=True)
            self._emissions_tracker = EmissionsTracker(
                project_name=f"system_{task}",
                output_dir=log_dir,
                save_to_file=True,
                log_level="error",
            )

    def start(self) -> None:
        if self._emissions_tracker:
            self._emissions_tracker.start()
        if self._enabled:
            self._sampler.start()

    def stop(self) -> Tuple[Optional[EmissionsMetrics], Optional[UtilizationMetrics]]:
        emissions = None
        if self._emissions_tracker:
            try:
                self._emissions_tracker.stop()
                emissions = _extract_emissions(self._emissions_tracker)
            except Exception:
                pass

        utilization = self._sampler.stop() if self._enabled else None
        return emissions, utilization
