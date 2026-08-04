#!/usr/bin/env python3
"""ESPHome Multi-Run Script - Batch compile and upload ESPHome configurations.

This script provides a unified tool for batch processing ESPHome configurations
with support for both serial and parallel execution modes. All functionality is
contained in a single file for easy deployment.

Features:
  - Serial and parallel execution modes
  - Automatic retry for failed builds
  - Color-coded output and progress tracking
  - Detailed execution summary with timing statistics
  - Graceful interrupt handling (Ctrl+C)
  - Real-time progress display in parallel mode

Usage:
    esphome-multi-run.py file1.yaml file2.yaml          Run specific files
    esphome-multi-run.py *.yaml                         Run files matching pattern
    esphome-multi-run.py -j 4 -p "*.yaml"               Run with 4 parallel workers
    esphome-multi-run.py examples/*/*.yaml              Run all files in subdirectories
    esphome-multi-run.py examples/Brand/*/*.yaml        Run all Brand configurations
    esphome-multi-run.py --help                         Show detailed help

"""

import argparse
import fnmatch
import glob
import json
import logging
import os
import pty
import re
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, NamedTuple, TypedDict

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# Core Data Structures, Enums, and Exceptions
# =============================================================================


class ExecutionStatus(str, Enum):
    """Status enumeration for execution results."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    TIMEOUT = "timeout"


class FailureType(str, Enum):
    """Failure type enumeration for retry decision.

    Used to distinguish between permanent errors (configuration issues)
    and transient errors (network/resource issues) to optimize retry strategy.
    """

    PERMANENT = "permanent"
    TRANSIENT = "transient"
    UNKNOWN = "unknown"


class Color(str, Enum):
    """ANSI color codes for terminal output."""

    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    BLUE = "\033[0;36m"
    RESET = "\033[0m"


class ExecutionResult(TypedDict):
    """Type-safe structure for execution results."""

    status: ExecutionStatus
    start_time: float | None
    end_time: float | None
    compile_time: float
    upload_time: float
    retry_count: int
    failure_type: FailureType


class ESPHomeRunnerError(Exception):
    """Base exception for all ESPHomeRunner errors.

    All custom exceptions in this application should inherit from this base.
    """

    pass


class ConfigurationError(ESPHomeRunnerError):
    """Raised when configuration is invalid.

    Used for validation errors in RunnerConfig and other configuration issues.
    """

    pass


@dataclass(frozen=True)
class RunnerConfig:
    """Immutable configuration for ESPHomeRunner.

    This class uses frozen=True to ensure immutability, preventing
    accidental modification of configuration after initialization.
    """

    files_to_run: list[str]
    exclude_file: Path = Path(".esphome-run-exclude")
    no_logs: bool = True
    parallel_workers: int = 0
    compile_only: bool = False
    log_dir: Path = Path("logs")
    max_retries: int = 3  # Configurable via CLI
    enable_failure_analysis: bool = True  # Enable smart failure detection (skip retry on config errors)

    # Warmup configuration
    warmup_enabled: bool = True
    warmup_cache_dir: Path = field(default_factory=lambda: _default_cache_dir())
    esphome_version: str = ""
    # Fingerprint of the installed toolchains (PlatformIO penv state + the
    # native ESP-IDF cache used by ESPHome >= 2026.7), sampled at startup.
    # Compared against the warmup stamp so a toolchain change (cold cache)
    # invalidates the stamp even when the ESPHome version is unchanged.
    toolchain_fingerprint: str = ""

    # Slow-start: enforces minimum gap between task starts to mitigate
    # cold-cache toolchain install races (pioarduino install_esptool on shared
    # penv/, native ESP-IDF framework extraction). The runner zeroes it at
    # startup when the warmup stamp proves the caches are already warm.
    slow_start_interval: float = 10.0  # seconds; 0 disables

    # Constants
    RETRY_BASE_DELAY: float = 3.0  # Base delay for exponential backoff
    RETRY_MAX_DELAY: float = 60.0  # Maximum retry delay (cap for exponential backoff)
    RETRY_EXPONENTIAL_BASE: float = 2.0  # Exponential base (delay = base_delay * base^retry_count)

    PROCESS_TERM_TIMEOUT: float = 5.0
    PROCESS_CLEANUP_TIMEOUT: float = 2.0
    PROCESS_WAIT_TIMEOUT: float = 3600.0  # 1 hour
    PROGRESS_UPDATE_INTERVAL: float = 0.5
    PROGRESS_BAR_LENGTH: int = 40
    MAX_FILENAME_DISPLAY: int = 60
    MAX_PARALLEL_WORKERS_WARNING: int = 16

    # Display and timing constants
    DISPLAY_THREAD_TIMEOUT: float = 2.0
    DISPLAY_INITIAL_DELAY: float = 0.2
    INTERRUPT_POLL_INTERVAL: float = 0.1
    PROCESS_POLL_INTERVAL: float = 0.1
    EXECUTOR_SHUTDOWN_DELAY: float = 0.1

    def __post_init__(self) -> None:
        """Validate configuration after initialization.

        Raises ConfigurationError if configuration is invalid.
        """
        if self.parallel_workers < 0:
            raise ConfigurationError("parallel_workers must be non-negative")
        if self.max_retries < 0:
            raise ConfigurationError("max_retries must be non-negative")
        if self.slow_start_interval < 0:
            raise ConfigurationError("slow_start_interval must be non-negative")

    @property
    def no_logs_arg(self) -> str:
        """Generate no-logs argument for ESPHome command.

        Returns "--no-logs" if no_logs is True, empty string otherwise.
        """
        return "--no-logs" if self.no_logs else ""

    @property
    def warmup_cache_path(self) -> Path:
        """Full path to the version-keyed warmup stamp file."""
        version = self.esphome_version or "unknown"
        return self.warmup_cache_dir / f"warmed-{version}"

    def calculate_retry_delay(self, retry_count: int) -> float:
        """Calculate retry delay using exponential backoff.

        Formula: delay = min(RETRY_BASE_DELAY * (RETRY_EXPONENTIAL_BASE ^ retry_count), RETRY_MAX_DELAY)

        Returns Delay in seconds, capped at RETRY_MAX_DELAY.
        """
        if retry_count <= 0:
            return self.RETRY_BASE_DELAY

        # Calculate exponential backoff: base_delay * (base ^ retry_count)
        delay = self.RETRY_BASE_DELAY * (self.RETRY_EXPONENTIAL_BASE ** retry_count)

        # Cap at maximum delay
        return min(delay, self.RETRY_MAX_DELAY)


@dataclass
class ExecutionStats:
    """Execution statistics for progress tracking.

    This class provides efficient statistical calculations using Counter
    for aggregating execution results.
    """

    completed: int = 0
    in_progress: int = 0
    pending: int = 0
    failed: int = 0
    success: int = 0
    retrying: int = 0
    total: int = 0

    @property
    def progress_pct(self) -> float:
        """Calculate progress percentage.

        Returns Percentage of completed executions (0-100).
        """
        return (self.completed / self.total * 100) if self.total > 0 else 0.0

    @classmethod
    def from_results(cls, results: dict[str, ExecutionResult]) -> "ExecutionStats":
        """Create stats from execution results.

        Uses Counter for efficient aggregation of status counts.

        Returns ExecutionStats instance with aggregated statistics.
        """
        status_counts = Counter(r["status"] for r in results.values())

        completed_statuses = {
            ExecutionStatus.SUCCESS,
            ExecutionStatus.FAILED,
            ExecutionStatus.INTERRUPTED,
            ExecutionStatus.TIMEOUT,
        }
        completed = sum(status_counts[status] for status in completed_statuses)

        # Count retrying files (in_progress with retry_count > 0)
        retrying = sum(
            1
            for r in results.values()
            if r["status"] == ExecutionStatus.IN_PROGRESS and r.get("retry_count", 0) > 0
        )

        return cls(
            completed=completed,
            in_progress=status_counts[ExecutionStatus.IN_PROGRESS],
            pending=status_counts[ExecutionStatus.PENDING],
            failed=status_counts[ExecutionStatus.FAILED],
            success=status_counts[ExecutionStatus.SUCCESS],
            retrying=retrying,
            total=len(results),
        )


class RegexPatterns:
    """Pre-compiled regex patterns for performance.

    Compiling patterns once at module load time improves performance
    when patterns are used repeatedly.
    """

    COMPILE_TIME: re.Pattern[str] = re.compile(r"Took (\d+\.\d+) seconds")
    UPLOAD_TIME: re.Pattern[str] = re.compile(r"Upload took (\d+\.\d+) seconds")
    ANSI_ESCAPE: re.Pattern[str] = re.compile(
        r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])"
    )


def create_execution_result(
    status: ExecutionStatus,
    start_time: float | None = None,
    end_time: float | None = None,
    compile_time: float = 0.0,
    upload_time: float = 0.0,
    retry_count: int = 0,
    failure_type: FailureType = FailureType.UNKNOWN,
) -> ExecutionResult:
    """Create an ExecutionResult with proper typing.

    Factory function to ensure ExecutionResult instances are created
    with correct types, improving type safety.

    Returns Properly typed ExecutionResult instance.
    """
    return ExecutionResult(
        status=status,
        start_time=start_time,
        end_time=end_time,
        compile_time=compile_time,
        upload_time=upload_time,
        retry_count=retry_count,
        failure_type=failure_type,
    )


def print_color(color: Color, message: str) -> None:
    """Print a message in a given color."""
    print(f"{color.value}{message}{Color.RESET.value}")


def _default_cache_dir() -> Path:
    """Return the OS-native user cache directory for esphome-multi-run.

    Linux / *BSD: respects XDG_CACHE_HOME, falls back to ~/.cache.
    macOS: ~/Library/Caches.
    Windows: %LOCALAPPDATA%, falls back to ~/AppData/Local.
    """
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "esphome-multi-run"
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        if base:
            return Path(base) / "esphome-multi-run" / "Cache"
        return Path.home() / "AppData" / "Local" / "esphome-multi-run" / "Cache"
    # Linux / other Unix: XDG Base Directory Spec
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "esphome-multi-run"


def get_esphome_version() -> str:
    """Probe ESPHome CLI for its version string.

    Returns the version number (e.g. '2026.4.0') or 'unknown' if the
    esphome binary cannot be invoked or its output is unparseable.
    Used to key the warmup cache stamp so a stamp from one ESPHome
    version is not reused across an upgrade.
    """
    try:
        result = subprocess.run(
            ["esphome", "version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    output = result.stdout.strip()
    # Accept "Version: 2026.4.0" or "2026.4.0"
    prefix = "Version:"
    if output.lower().startswith(prefix.lower()):
        output = output[len(prefix):].strip()
    return output or "unknown"


def get_toolchain_fingerprint() -> str:
    """Fingerprint the installed toolchain state (cheap, no subprocess).

    Two independent toolchain ecosystems can race during parallel batch
    builds; the fingerprint covers both so a change in either invalidates the
    warmup stamp even when the ESPHome version -- and therefore the stamp
    filename -- is unchanged (the gap the version-only key missed and that
    slow-start could not deterministically cover):

    - PlatformIO (~/.platformio): pioarduino re-runs `install_esptool()
      --force-reinstall` into the shared penv at the start of every compile
      while the penv esptool differs from the version pinned by the active
      platform package. Still exercised by esp8266 / LibreTiny builds and by
      esp32 with `toolchain: platformio` (ESPHome >= 2026.7 builds esp32
      natively by default).
    - Native ESP-IDF cache (ESPHome >= 2026.7): the installer has no
      inter-process locking -- two cold parallel builds both rmdir and
      re-extract the same frameworks/<version> tree, corrupting it.

    All components degrade to "unknown"/"absent" on read errors, which reads
    as cold and forces a fresh serial warmup first.
    """
    return f"{_platformio_fingerprint()}|{_espidf_fingerprint()}"


def _platformio_fingerprint() -> str:
    """PlatformIO toolchain state under ~/.platformio.

    Components:
      - active espressif platform package versions   -> general toolchain bumps
      - tool-esptoolpy package version ("required")  -> what the platform wants
      - esptool version installed in the penv ("have") -> what is actually there

    When required != have, a force-reinstall is pending and parallel builds
    will race; the differing fingerprint forces a fresh serial warmup first.
    """
    pio = Path.home() / ".platformio"

    plats: list[str] = []
    plat_dir = pio / "platforms"
    if plat_dir.is_dir():
        for pj in sorted(plat_dir.glob("*/platform.json")):
            try:
                data = json.loads(pj.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            plats.append(f"{data.get('name', '?')}@{data.get('version', '?')}")
    platforms = ",".join(sorted(set(plats))) if plats else "unknown"

    pkg_ver = "unknown"
    pkg_json = pio / "packages" / "tool-esptoolpy" / "package.json"
    try:
        pkg_ver = json.loads(pkg_json.read_text(encoding="utf-8")).get(
            "version", "unknown"
        )
    except (OSError, ValueError):
        pass

    penv_ver = "unknown"
    penv_lib = pio / "penv" / "lib"
    if penv_lib.is_dir():
        dists = sorted(penv_lib.glob("python*/site-packages/esptool-*.dist-info"))
        if dists:
            name = dists[0].name  # e.g. "esptool-5.2.0.dist-info"
            penv_ver = name[len("esptool-"):-len(".dist-info")]

    return f"platforms={platforms}|esptool_pkg={pkg_ver}|esptool_penv={penv_ver}"


def _espidf_cache_root() -> Path:
    """Root of ESPHome's native ESP-IDF toolchain cache (>= 2026.7).

    Mirrors esphome's own resolution -- ESPHOME_ESP_IDF_PREFIX override, else
    platformdirs.user_cache_dir("esphome", appauthor=False) / "idf" -- without
    taking on the platformdirs dependency.
    """
    if prefix := os.environ.get("ESPHOME_ESP_IDF_PREFIX", "").strip():
        return Path(prefix).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "esphome" / "idf"
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA")
        base_path = Path(base) if base else Path.home() / "AppData" / "Local"
        return base_path / "esphome" / "Cache" / "idf"
    xdg = os.environ.get("XDG_CACHE_HOME")
    base_path = Path(xdg) if xdg else Path.home() / ".cache"
    return base_path / "esphome" / "idf"


def _espidf_fingerprint() -> str:
    """Native ESP-IDF cache state (frameworks + python envs).

    Captured per frameworks/<ver>: the extraction-complete marker and
    ESPHome's install stamp, which records the installed targets/tools sets.
    A single build installs toolchains for all chip targets (targets=["all"]),
    so one warm representative covers every esp32 variant. penvs/<ver> tracks
    the per-IDF-version python envs.
    """
    root = _espidf_cache_root()

    frameworks: list[str] = []
    fw_dir = root / "frameworks"
    if fw_dir.is_dir():
        for d in sorted(fw_dir.iterdir()):
            if not d.is_dir():
                continue
            state = (
                "extracted" if (d / ".esphome_extracted").is_file() else "partial"
            )
            stamp = "no-stamp"
            try:
                data = json.loads(
                    (d / ".esphome.stamp.json").read_text(encoding="utf-8")
                )
                targets = ",".join(data.get("targets", []))
                tools = ",".join(data.get("tools", []))
                stamp = f"targets={targets};tools={tools}"
            except (OSError, ValueError):
                pass
            frameworks.append(f"{d.name}:{state}:{stamp}")

    penvs: list[str] = []
    penv_dir = root / "penvs"
    if penv_dir.is_dir():
        penvs = sorted(p.name for p in penv_dir.iterdir() if p.is_dir())

    fw_part = ",".join(frameworks) if frameworks else "absent"
    penv_part = ",".join(penvs) if penvs else "absent"
    return f"idf_frameworks={fw_part}|idf_penvs={penv_part}"


class BucketKey(NamedTuple):
    """Toolchain bucket identity for grouping yamls during warmup.

    Two yamls with the same BucketKey exercise the same toolchain +
    framework installation path (PlatformIO packages or the native ESP-IDF
    cache), so compiling one of them warms the cache for all.
    """
    platform: str          # "esp32" / "esp8266" / "rp2" / "ln882x" / ... / "default"
    chip_variant: str      # "ESP32" / "ESP32S3" / "ESP32C3" / ... or platform fallback
    framework_type: str    # "esp-idf" / "arduino" / "default"
    framework_version: str # "recommended" / "latest" / URL / "default"
    # ESPHome >= 2026.7 esp32 `toolchain:` key ("platformio" opts out of the
    # native ESP-IDF toolchain). Different toolchains install into different
    # caches, so they must warm separately. Defaulted so 4-field construction
    # sites (and sentinels) keep working.
    toolchain: str = "default"


_KNOWN_PLATFORMS = ("esp32", "esp8266", "rp2040", "rp2", "bk72xx", "rtl87xx",
                    "libretiny", "ln882x", "nrf52", "host")

# Sentinel buckets. PROBE_FAILED groups yamls whose `esphome config` failed —
# warmup skips them since they will fail again and surface the real error in
# the parallel phase. NO_PLATFORM groups yamls with no recognized platform
# key (rare); they still get warmed as their own bucket.
PROBE_FAILED_BUCKET = BucketKey("<probe-failed>", "", "", "")
NO_PLATFORM_BUCKET = BucketKey("<no-platform>", "", "", "")


def format_bucket_label(bk: BucketKey) -> str:
    """Render a BucketKey for display. Sentinels show only their tag."""
    if bk in (PROBE_FAILED_BUCKET, NO_PLATFORM_BUCKET):
        return bk.platform
    parts = list(bk)
    if bk.toolchain == "default":
        parts.pop()  # keep labels stable for the common no-override case
    return "/".join(parts)


def _find_field(lines: list[str], path: tuple[str, ...]) -> str | None:
    """Walk a YAML text by 2-space indentation to retrieve a scalar leaf.

    `path` is a tuple of keys, e.g. ('esp32', 'framework', 'type').
    Returns the string value at that path, or None if not present.
    """
    depth = 0
    i = 0
    current_indent = 0
    while i < len(lines) and depth < len(path):
        line = lines[i]
        if not line.strip() or line.lstrip().startswith("#"):
            i += 1
            continue
        indent = len(line) - len(line.lstrip())
        expected_indent = depth * 2
        stripped = line.strip()
        if indent < current_indent and depth > 0:
            # Left the sub-tree we were descending into without finding the key.
            return None
        if indent == expected_indent and stripped.startswith(f"{path[depth]}:"):
            if depth == len(path) - 1:
                value = stripped.split(":", 1)[1].strip()
                return value if value else None
            depth += 1
            current_indent = (depth) * 2
            i += 1
            continue
        i += 1
    return None


def extract_bucket_fields(resolved_config: str) -> BucketKey:
    """Extract the toolchain bucket key from `esphome config` output.

    Only reads user-facing top-level keys (esp32, esp8266, rp2040, ...)
    and their immediate framework / variant sub-fields. No internal
    ESPHome tables are consulted.
    """
    lines = resolved_config.splitlines()
    platform = "default"
    for candidate in _KNOWN_PLATFORMS:
        for line in lines:
            if line.startswith(f"{candidate}:"):
                platform = candidate
                break
        if platform != "default":
            break
    if platform == "default":
        return NO_PLATFORM_BUCKET

    variant = _find_field(lines, (platform, "variant")) or platform
    framework_type = _find_field(lines, (platform, "framework", "type"))
    framework_version = _find_field(lines, (platform, "framework", "version"))

    # esp8266 has no framework.type in config; it's always arduino.
    if framework_type is None and platform == "esp8266":
        framework_type = "arduino"
    if framework_type is None:
        framework_type = "default"
    if framework_version is None:
        # rp2040 uses framework.platform_version instead of .version
        framework_version = _find_field(lines, (platform, "framework", "platform_version")) or "default"

    toolchain = _find_field(lines, (platform, "toolchain")) or "default"
    return BucketKey(platform, variant, framework_type, framework_version, toolchain)


def probe_bucket(file_path: str, timeout: float = 60.0) -> BucketKey:
    """Resolve a yaml's toolchain bucket by invoking `esphome config`.

    Returns the default 'unknown' bucket on any failure; failures are
    intentionally swallowed here so warmup can still proceed with a
    best-effort grouping. The parallel phase will surface real config
    errors on its own pass.
    """
    try:
        result = subprocess.run(
            ["esphome", "config", file_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return PROBE_FAILED_BUCKET
    if result.returncode != 0:
        return PROBE_FAILED_BUCKET
    return extract_bucket_fields(result.stdout)


def read_warmup_stamp(stamp_path: Path, expected_fingerprint: str) -> bool:
    """Return True only if a valid, toolchain-matching warmup stamp exists.

    A stamp is honored (warmup skipped) only when its recorded
    `toolchain_fingerprint` matches the current one. A missing file, an
    unparseable stamp, a legacy stamp without the fingerprint field, or a
    fingerprint mismatch all read as "cold" -> return False so a fresh serial
    warmup runs before parallel dispatch. This is what stops a toolchain upgrade
    (e.g. an esptool pinned-version bump, or a wiped native ESP-IDF cache)
    from racing when the ESPHome version -- and therefore the stamp filename --
    has not changed.
    """
    try:
        data = json.loads(stamp_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return data.get("toolchain_fingerprint") == expected_fingerprint


def write_warmup_stamp(stamp_path: Path, version: str, buckets: list[str]) -> bool:
    """Write the warmup stamp. Returns True on success, False on IO error.

    The toolchain fingerprint is sampled here, AFTER the serial warmup build
    has installed and aligned the toolchains (PlatformIO penv esptool, native
    ESP-IDF framework + tools), so a freshly written stamp records the "warm"
    (consistent) fingerprint.
    """
    payload = {
        "version": version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "toolchain_fingerprint": get_toolchain_fingerprint(),
        "buckets": buckets,
    }
    try:
        stamp_path.parent.mkdir(parents=True, exist_ok=True)
        stamp_path.write_text(json.dumps(payload), encoding="utf-8")
        return True
    except OSError:
        return False


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from a string.

    Returns String with ANSI codes removed.
    """
    return RegexPatterns.ANSI_ESCAPE.sub("", text)


def append_failure_analysis_note(log_path: Path, failure_type: FailureType) -> None:
    """Append failure analysis note to log file.

    Writes a detailed explanation to the log file when a permanent failure
    is detected, helping users understand why retry was skipped.
    """
    if failure_type != FailureType.PERMANENT:
        return

    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write("\n\n" + "=" * 80 + "\n")
            log_file.write("=== FAILURE ANALYSIS ===\n")
            log_file.write(f"Time: {timestamp}\n")
            log_file.write("Status: Permanent configuration error detected\n")
            log_file.write("Retry: Skipped to save time\n")
            log_file.write("=" * 80 + "\n")
            log_file.write("This failure appears to be a permanent configuration issue that\n")
            log_file.write("cannot be fixed by retrying. Common causes:\n")
            log_file.write("  - Invalid YAML syntax\n")
            log_file.write("  - Missing files (e.g., secrets.yaml)\n")
            log_file.write("  - Component configuration errors\n")
            log_file.write("  - Duplicate IDs or conflicting configurations\n")
            log_file.write("\nRetry attempts have been skipped to save time.\n")
            log_file.write("Please fix the configuration error and run again.\n")
            log_file.write("\nTo force retry on all errors, use: --disable-failure-analysis\n")
            log_file.write("=" * 80 + "\n")
    except (OSError, IOError) as e:
        # Log failure but continue execution
        logger.debug(f"Failed to write failure analysis note to {log_path}: {e}")


def calculate_common_prefix(file_paths: list[str]) -> tuple[str, int]:
    """Calculate the common directory prefix for a list of file paths.

    Returns (common_prefix_path, depth): the common directory prefix (empty
    if none) and the number of directory levels it contains, e.g.
    ["examples/Brand/A/x.yaml", "examples/Brand/B/y.yaml"] -> ("examples/Brand", 2).
    """
    if not file_paths:
        return "", 0

    # Convert all paths to Path objects and get their parent directories
    parent_parts_list = [Path(p).parent.parts for p in file_paths]

    # If any file is in the current directory, no common prefix
    if any(len(parts) == 0 for parts in parent_parts_list):
        return "", 0

    # Find common prefix by comparing parts
    common_parts = []
    min_length = min(len(parts) for parts in parent_parts_list)

    for i in range(min_length):
        current_part = parent_parts_list[0][i]
        if all(parts[i] == current_part for parts in parent_parts_list):
            common_parts.append(current_part)
        else:
            break

    if not common_parts:
        return "", 0

    common_prefix = "/".join(common_parts)
    depth = len(common_parts)

    return common_prefix, depth


# =============================================================================
# Failure Analysis (SOLID Principle: SRP, OCP, DIP)
# =============================================================================


class ESPHomeFailureAnalyzer:
    """Analyzes ESPHome execution logs to identify permanent failures.

    This implementation detects common configuration errors that won't be
    fixed by retry, such as YAML syntax errors or missing files. Follows
    Single Responsibility and Open/Closed principles.
    """

    # Patterns for permanent errors (Open for extension)
    PERMANENT_ERROR_PATTERNS: list[str] = [
        r"Invalid YAML syntax",
        r"Failed config",
        # Native ESP-IDF builds (ESPHome >= 2026.7): a CMake configure
        # failure is deterministic -- retrying just repeats a long build.
        r"CMake Error",
    ]

    def analyze(self, log_path: Path) -> FailureType:
        """Analyze ESPHome log file for permanent errors.

        Reads the first 300 lines of the log file (configuration errors appear
        within the first few lines; CMake configure errors follow the esphome
        preamble but still precede the bulk of compile output) and checks for
        known permanent error patterns.

        Returns FailureType.PERMANENT if configuration error detected, FailureType.UNKNOWN otherwise (conservative retry).
        """
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as log_file:
                # Read first 300 lines (config + CMake errors appear early)
                lines = []
                for _ in range(300):
                    line = log_file.readline()
                    if not line:
                        break
                    lines.append(line)

                # Join lines for multi-line error detection
                log_content = "".join(lines)

                # Check for permanent error patterns
                for pattern in self.PERMANENT_ERROR_PATTERNS:
                    if re.search(pattern, log_content, re.IGNORECASE):
                        return FailureType.PERMANENT

                return FailureType.UNKNOWN

        except (OSError, IOError) as e:
            # If we can't read the log, assume unknown (conservative)
            logger.debug(f"Failed to read log file {log_path} for failure analysis: {e}")
            return FailureType.UNKNOWN


# =============================================================================
# File Filtering Logic
# =============================================================================


class FileFilter:
    """Filters files based on exclusion patterns.

    This class is responsible for reading exclusion patterns from a file
    and filtering a list of files based on those patterns.
    """

    # Default exclusion patterns (used when no exclude file)
    DEFAULT_PATTERNS = [
        "secrets.yaml",
        "secrets.yml",
        ".*.yaml",  # Hidden YAML files
        ".*.yml",   # Hidden YML files
    ]

    def __init__(self, exclude_file: Path):
        """Initialize the file filter."""
        self.exclude_file = exclude_file
        self.patterns: list[str] = []

    def load_patterns(self) -> None:
        """Load exclusion patterns from file or use defaults.

        If exclude_file doesn't exist, use default patterns to exclude
        common non-executable files like secrets.yaml.

        If exclude_file exists, use ONLY the patterns from the file,
        giving users complete control over what gets excluded.
        """
        if not self.exclude_file.exists():
            # No exclude file -> use default patterns
            self.patterns = self.DEFAULT_PATTERNS.copy()
            print("Using default exclusion patterns (no exclude file found):")
            for pattern in self.patterns:
                print(f"  - {pattern}")
            return

        # Exclude file exists -> use ONLY file patterns (user has full control)
        print(f"Active exclusion patterns from {self.exclude_file}:")
        self.patterns = [
            line.strip()
            for line in self.exclude_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

        for pattern in self.patterns:
            print(f"  - {pattern}")

    def filter_files(self, files: list[str]) -> tuple[list[str], list[str]]:
        """Filter files based on loaded exclusion patterns.

        Returns Tuple of (included_files, excluded_files).
        """
        if not self.patterns:
            return files, []

        excluded_files: list[str] = []
        included_files: list[str] = []

        for file_path in files:
            basename = Path(file_path).name
            is_excluded = any(
                fnmatch.fnmatch(basename, pattern) for pattern in self.patterns
            )

            if is_excluded:
                excluded_files.append(file_path)
                print_color(
                    Color.YELLOW,
                    f"Excluding: {file_path} (matched exclusion pattern)",
                )
            else:
                included_files.append(file_path)

        return included_files, excluded_files

    def apply_filters(self, files: list[str], verbose: bool = True) -> list[str]:
        """Load patterns and filter files in one operation.

        Convenience method that combines load_patterns() and filter_files().

        Returns List of included files after filtering.
        """
        self.load_patterns()
        included_files, excluded_files = self.filter_files(files)

        if verbose:
            print(f"\nFiles to process: {len(included_files)}")
            if excluded_files:
                print(f"Files excluded: {len(excluded_files)}")

        return included_files


# =============================================================================
# Result Tracking and Statistics
# =============================================================================


class ResultTracker:
    """Thread-safe tracker for execution results and statistics.

    This class provides centralized management of execution results with
    thread-safe operations for parallel mode. It is responsible only for
    data storage and retrieval, following the Single Responsibility Principle.
    Presentation logic is handled by ResultSummaryRenderer.
    """

    def __init__(self) -> None:
        """Initialize the result tracker."""
        self.results: dict[str, ExecutionResult] = {}
        self.results_lock = threading.Lock()
        self.overall_start_time: float | None = None
        self.overall_end_time: float | None = None

    def initialize_results(self, files: list[str]) -> None:
        """Initialize each file's entry to PENDING, preserving any prior SUCCESS.

        Re-init is idempotent for files already marked SUCCESS (e.g. by
        the warmup phase). Other existing statuses are reset to PENDING
        because this is called before each execution batch and stale
        FAILED/IN_PROGRESS entries should not leak forward.
        """
        with self.results_lock:
            for file_path in files:
                existing = self.results.get(file_path)
                if existing is not None and existing["status"] == ExecutionStatus.SUCCESS:
                    continue
                self.results[file_path] = create_execution_result(
                    status=ExecutionStatus.PENDING
                )

    def update_result(self, file_path: str, result: ExecutionResult) -> None:
        """Update result for a file (thread-safe)."""
        with self.results_lock:
            self.results[file_path] = result

    def get_result(self, file_path: str) -> ExecutionResult | None:
        """Get result for a file (thread-safe).

        Returns ExecutionResult if exists, None otherwise.
        """
        with self.results_lock:
            return self.results.get(file_path)

    def get_stats(self) -> ExecutionStats:
        """Get current execution statistics (thread-safe).

        Returns ExecutionStats with current counts.
        """
        with self.results_lock:
            return ExecutionStats.from_results(self.results)

    def get_all_results(self) -> dict[str, ExecutionResult]:
        """Get deep copy of all results (thread-safe).

        Returns Deep copy of results dictionary. Modifying the returned dictionary will not affect the internal state.
        """
        import copy

        with self.results_lock:
            return copy.deepcopy(self.results)


@dataclass
class WarmupOutcome:
    """High-level outcome of the whole warmup phase."""
    disabled: bool = False          # --disable-warmup was set
    cache_hit: bool = False         # stamp existed, warmup skipped
    reps_compiled: list[str] = field(default_factory=list)
    buckets: list[str] = field(default_factory=list)
    success: bool = True            # False if any rep failed; stamp not written


class WarmupPhase:
    """Serial pre-compile of one representative per toolchain bucket.

    Groups the filtered file list by BucketKey, picks one file per
    bucket, and compiles the representatives sequentially via
    `esphome compile` so that the toolchain caches (PlatformIO packages /
    the native ESP-IDF install) are populated before any parallel worker
    starts.
    """

    def __init__(self, config: "RunnerConfig"):
        self.config = config
        # Streaming state — populated by begin(), consumed by workers via
        # wait_for_file() and by the background compile thread. Always
        # re-initialized at the top of begin() so the same WarmupPhase
        # instance can be reused safely across invocations.
        self._gate_enabled: bool = False
        self._bucket_ready: dict[BucketKey, threading.Event] = {}
        self._file_to_bucket: dict[str, BucketKey] = {}
        self._compile_thread: threading.Thread | None = None
        self._outcome: WarmupOutcome = WarmupOutcome()
        self._result_tracker_ref: "ResultTracker | None" = None
        # Cancellation signal — set by cancel() (called from the executor's
        # interrupt handler). The compile loop checks it between reps and
        # while polling each subprocess; wait_for_file checks it before
        # blocking and after waking so workers don't hang on Ctrl-C.
        self._cancelled: threading.Event = threading.Event()
    # Maximum wall time we'll wait for one warmup `esphome compile`. A cold
    # native ESP-IDF warmup (ESPHome >= 2026.7) downloads the framework,
    # per-target compilers and a python env before the first full build, so
    # this needs generous headroom. If a subprocess hangs (e.g. a flaky
    # mirror), we'd rather fail it and let the parallel phase surface the
    # real error than block forever.
    WARMUP_REP_TIMEOUT_SECONDS: float = 1800.0
    # Poll interval for cancellation while a rep subprocess is running.
    WARMUP_POLL_INTERVAL_SECONDS: float = 0.5

    def probe_buckets(self, files: list[str]) -> dict["BucketKey", list[str]]:
        """Run `esphome config` on each file (parallel), group by BucketKey.

        Streams live `probed N/M` progress on a single overwritten terminal
        line so the user sees the phase advancing instead of a stuck prompt.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        workers = max(1, self.config.parallel_workers or 1)
        total = len(files)
        buckets: dict[BucketKey, list[str]] = {}
        # Preserve input order in bucket values; we re-append in original order after.
        results: dict[str, BucketKey] = {}

        is_tty = sys.stdout.isatty()
        prefix = f"  Probing {total} yaml(s)..."
        if is_tty:
            sys.stdout.write(prefix)
            sys.stdout.flush()

        def _report(done: int, final: bool = False) -> None:
            if is_tty:
                sys.stdout.write(f"\r{prefix} {done}/{total}\x1b[K")
                if final:
                    sys.stdout.write("\n")
                sys.stdout.flush()
            elif final:
                sys.stdout.write(f"{prefix} {done}/{total}\n")
                sys.stdout.flush()

        with ThreadPoolExecutor(max_workers=workers) as ex:
            future_to_file = {ex.submit(probe_bucket, f): f for f in files}
            done = 0
            _report(done)
            for future in as_completed(future_to_file):
                done += 1
                file_path = future_to_file[future]
                results[file_path] = future.result()
                _report(done)
            _report(done, final=True)

        for file_path in files:  # preserve input order within each bucket
            bk = results[file_path]
            buckets.setdefault(bk, []).append(file_path)
        return buckets

    def select_representatives(
        self, buckets: dict["BucketKey", list[str]]
    ) -> list[str]:
        """Pick one file per bucket.

        Strategy: first file in each bucket after sorting alphabetically.
        Deterministic and cheap; any file from a bucket warms the same
        toolchain.
        """
        reps = []
        for bk in sorted(buckets.keys()):
            files_in_bucket = buckets[bk]
            if files_in_bucket:
                reps.append(sorted(files_in_bucket)[0])
        return reps

    def begin(
        self,
        files: list[str],
        result_tracker: "ResultTracker | None" = None,
    ) -> "WarmupOutcome | None":
        """Set up warmup state and kick off background compilation.

        Returns None if streaming compilation was started in a background
        thread — the caller MUST then gate workers via wait_for_file() and
        eventually call finish() to retrieve the outcome.

        Returns a terminal WarmupOutcome when warmup is not applicable
        (disabled, cache hit, or single-worker mode) — the caller can use
        it directly and skip gating.

        Passing a result_tracker enables marking each rep as SUCCESS as
        soon as its compile finishes, so a worker assigned the rep file
        will skip it instead of recompiling.
        """
        # Re-initialize state so calling begin() twice on the same instance
        # is safe (tests, future retry loops). Anything left over from a
        # prior invocation would otherwise leak: stale outcome flags, fired
        # bucket Events that workers wait_for_file would no-op past, etc.
        self._outcome = WarmupOutcome()
        self._gate_enabled = False
        self._bucket_ready = {}
        self._file_to_bucket = {}
        self._compile_thread = None
        self._result_tracker_ref = None
        self._cancelled = threading.Event()

        if not self.config.warmup_enabled:
            self._outcome.disabled = True
            return self._outcome
        # Check the stamp BEFORE the single-worker shortcut so a prior
        # successful warmup is honored regardless of the current execution
        # mode (otherwise `-j 1` would always claim disabled even when a
        # valid cache stamp from `-j 4` exists).
        if read_warmup_stamp(
            self.config.warmup_cache_path, self.config.toolchain_fingerprint
        ):
            self._outcome.cache_hit = True
            return self._outcome
        if self.config.parallel_workers <= 1:
            self._outcome.disabled = True
            return self._outcome

        print_color(Color.BLUE, "Warmup phase:")
        buckets = self.probe_buckets(files)
        probe_failed_files = buckets.pop(PROBE_FAILED_BUCKET, [])
        if probe_failed_files:
            print_color(
                Color.YELLOW,
                f"  {len(probe_failed_files)} yaml(s) failed probe "
                "(parallel phase will surface the error):"
            )
            for f in probe_failed_files:
                print_color(Color.YELLOW, f"    - {f}")
        reps = self.select_representatives(buckets)
        self._outcome.buckets = [
            format_bucket_label(bk) for bk in sorted(buckets.keys())
        ]
        if not reps:
            self._outcome.success = True
            return self._outcome

        rep_by_bucket = {sorted(bf)[0]: bk for bk, bf in buckets.items() if bf}
        sorted_keys = sorted(buckets.keys())
        width = max(
            (len(format_bucket_label(bk)) for bk in sorted_keys), default=0
        )
        print_color(Color.BLUE, f"  {len(reps)} toolchain bucket(s) detected:")
        for bk in sorted_keys:
            label = format_bucket_label(bk).ljust(width)
            print_color(Color.BLUE, f"    {label}  →  {sorted(buckets[bk])[0]}")
        print_color(
            Color.BLUE,
            "  Compiling representatives in background; workers join as each "
            "bucket warms up (PIO output redirected to logs/*-warmup.log)."
        )

        # Set up gating: per-bucket Event + file→bucket lookup.
        self._gate_enabled = True
        for bk, bf in buckets.items():
            self._bucket_ready[bk] = threading.Event()
            for f in bf:
                self._file_to_bucket[f] = bk

        self._result_tracker_ref = result_tracker
        # Set ESPHOME_SKIP_CLEAN_BUILD BEFORE starting workers (which may
        # spawn esphome subprocesses as soon as their bucket Event fires).
        # If we waited until _compile_loop's tail to flip this, early-bucket
        # workers would inherit an unset env and do a redundant clean.
        os.environ["ESPHOME_SKIP_CLEAN_BUILD"] = "1"
        self._compile_thread = threading.Thread(
            target=self._compile_loop,
            args=(reps, rep_by_bucket),
            daemon=True,
            name="WarmupCompile",
        )
        self._compile_thread.start()
        return None  # Streaming in progress; caller must call finish()

    def _compile_loop(
        self,
        reps: list[str],
        rep_to_bucket: dict[str, "BucketKey"],
    ) -> None:
        """Sequential rep compilation, firing bucket events as each finishes.

        Invariant: by the time this function returns (for ANY reason — clean
        completion, cancellation, unexpected exception), every Event in
        self._bucket_ready is .set(). Workers blocked in wait_for_file()
        MUST observe a set Event to make progress; if even one Event is
        left unset the workers waiting on that bucket hang forever.
        """
        log_dir = self.config.log_dir
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

        failures: list[str] = []
        total = len(reps)
        cancelled_midway = False
        try:
            for idx, rep in enumerate(reps, 1):
                if self._cancelled.is_set():
                    cancelled_midway = True
                    break
                success = self._compile_one_rep(rep, idx, total, log_dir)
                if success is None:
                    # Cancellation surfaced from inside the subprocess poll —
                    # treat the remaining reps as un-attempted.
                    cancelled_midway = True
                    break
                if success:
                    self._outcome.reps_compiled.append(rep)
                    if self._result_tracker_ref is not None:
                        self._result_tracker_ref.update_result(
                            rep,
                            create_execution_result(
                                status=ExecutionStatus.SUCCESS,
                                retry_count=0,
                            ),
                        )
                else:
                    failures.append(rep)
                bucket = rep_to_bucket.get(rep)
                if bucket is not None:
                    self._bucket_ready[bucket].set()
        finally:
            # Always release every remaining Event, even on uncaught
            # exception or cancellation. Workers blocked on wait_for_file
            # depend on this to make progress.
            for ev in self._bucket_ready.values():
                ev.set()
            # Compute the real success state. Default-True from the
            # dataclass would lie about a partial / cancelled run.
            attempted = len(self._outcome.reps_compiled) + len(failures)
            self._outcome.success = (
                not failures
                and not cancelled_midway
                and attempted == total
            )

        # Post-completion bookkeeping. Only persist the stamp / nudge the
        # env on a clean full success — partial warmup must NOT be cached.
        if self._outcome.success:
            written = write_warmup_stamp(
                self.config.warmup_cache_path,
                version=self.config.esphome_version or "unknown",
                buckets=self._outcome.buckets,
            )
            if not written:
                print_color(
                    Color.YELLOW,
                    f"Warning: could not write warmup stamp at "
                    f"{self.config.warmup_cache_path}"
                )
        elif failures:
            print_color(
                Color.YELLOW,
                f"Warmup: {len(failures)} rep(s) failed; "
                "parallel phase will surface their errors"
            )
        elif cancelled_midway:
            print_color(Color.YELLOW, "Warmup: cancelled before completion")

    def _compile_one_rep(
        self,
        rep: str,
        idx: int,
        total: int,
        log_dir: Path,
    ) -> bool | None:
        """Compile a single warmup rep with cancellation polling.

        Returns True on success, False on failure, None if cancelled mid-flight.
        """
        cmd = ["esphome", "compile", rep]
        log_path = log_dir / f"{Path(rep).stem}-warmup.log"
        start = time.monotonic()
        proc: subprocess.Popen | None = None
        try:
            log_file = open(log_path, "w", encoding="utf-8")
        except OSError:
            print_color(
                Color.YELLOW,
                f"  [warmup {idx}/{total}] {rep}  ✗ could not open log"
            )
            return False
        try:
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
            except (FileNotFoundError, OSError) as e:
                print_color(
                    Color.YELLOW,
                    f"  [warmup {idx}/{total}] {rep}  ✗ spawn failed: {e}"
                )
                return False
            # Poll loop so cancellation actually preempts a long compile.
            deadline = start + self.WARMUP_REP_TIMEOUT_SECONDS
            while True:
                if self._cancelled.is_set():
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    return None
                try:
                    rc = proc.wait(timeout=self.WARMUP_POLL_INTERVAL_SECONDS)
                    break
                except subprocess.TimeoutExpired:
                    if time.monotonic() > deadline:
                        proc.terminate()
                        try:
                            proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait()
                        elapsed = time.monotonic() - start
                        print_color(
                            Color.YELLOW,
                            f"  [warmup {idx}/{total}] {rep}  "
                            f"✗ {elapsed:.1f}s  (timed out — see {log_path})"
                        )
                        return False
            success = rc == 0
        finally:
            try:
                log_file.close()
            except OSError:
                pass
        elapsed = time.monotonic() - start
        if success:
            status_str = f"✓ {elapsed:.1f}s"
        else:
            status_str = f"✗ {elapsed:.1f}s  (see {log_path})"
        # Single completion line — the parallel progress display is
        # concurrently writing to the same terminal during streaming warmup,
        # so no in-place cursor updates here.
        print_color(
            Color.BLUE,
            f"  [warmup {idx}/{total}] {rep}  {status_str}"
        )
        return success

    def cancel(self) -> None:
        """Signal the background compile loop to stop and unblock all workers.

        Called by the executor's interrupt handler. Safe to call multiple
        times and safe to call when no streaming is in progress.
        """
        self._cancelled.set()
        # Release every bucket Event so wait_for_file() returns immediately.
        for ev in self._bucket_ready.values():
            ev.set()

    def wait_for_file(self, file_path: str) -> None:
        """Worker hook: block until this file's toolchain bucket is warmed.

        No-op when streaming gating is disabled (warmup disabled, serial
        mode, cache hit, or the file isn't in any tracked bucket). Also
        returns immediately once cancel() has been called, so an interrupted
        run never leaves workers blocked indefinitely.
        """
        if not self._gate_enabled or self._cancelled.is_set():
            return
        bucket = self._file_to_bucket.get(file_path)
        if bucket is None:
            return
        event = self._bucket_ready.get(bucket)
        if event is not None:
            event.wait()

    def finish(self) -> WarmupOutcome:
        """Block until background warmup completes; return the final outcome."""
        if self._compile_thread is not None:
            self._compile_thread.join()
        return self._outcome

    def run(self, files: list[str]) -> WarmupOutcome:
        """Synchronous shim: kick off streaming and wait for completion.

        Kept for callers that don't want to gate workers on warmup progress
        (the gating is harmless either way; without it, this just runs
        warmup to completion before returning).
        """
        early = self.begin(files, result_tracker=None)
        if early is not None:
            return early
        return self.finish()


class ResultSummaryRenderer:
    """Renders execution result summaries in table format.

    This class is responsible for presenting execution results to the user.
    It follows the Single Responsibility Principle by separating presentation
    logic from data management (which is handled by ResultTracker).

    The renderer handles:
    - Table rendering with aligned columns
    - Color-coded status display
    - Time formatting and highlighting
    - Summary statistics calculation
    """

    def __init__(self, result_tracker: ResultTracker):
        """Initialize the summary renderer."""
        self.result_tracker = result_tracker

    def print_summary(
        self,
        files_to_run: list[str],
        parallel_workers: int,
        interrupted: bool = False,
    ) -> None:
        """Print final execution summary in table format."""
        summary_title = "EXECUTION SUMMARY"
        if interrupted:
            summary_title += " (INTERRUPTED)"

        print_color(Color.BLUE, summary_title)

        # Data preparation
        table_data = []
        total_time = 0.0
        max_compile_time = 0.0
        max_upload_time = 0.0
        max_total_duration = 0.0

        # Get results from tracker
        results = self.result_tracker.get_all_results()

        for file_path in files_to_run:
            result = results.get(file_path)
            if not result:
                table_data.append([file_path, "skipped", "-", "-", "-", (0, FailureType.UNKNOWN)])
                continue

            status = result["status"]
            compile_time = result.get("compile_time", 0.0)
            upload_time = result.get("upload_time", 0.0)
            retry_count = result.get("retry_count", 0)
            failure_type = result.get("failure_type", FailureType.UNKNOWN)

            end_time = result.get("end_time")
            start_time = result.get("start_time")
            if end_time is not None and start_time is not None:
                total_duration = end_time - start_time
                total_time += total_duration
            else:
                total_duration = 0.0

            # Store retry info as tuple: (retry_count, failure_type)
            table_data.append(
                [
                    file_path,
                    status,
                    compile_time,
                    upload_time,
                    total_duration,
                    (retry_count, failure_type),
                ]
            )

            max_compile_time = max(max_compile_time, compile_time)
            max_upload_time = max(max_upload_time, upload_time)
            max_total_duration = max(max_total_duration, total_duration)

        # Table rendering
        self._render_table(table_data, max_compile_time, max_upload_time, max_total_duration)

        # Print total execution time
        if parallel_workers > 0 and self.result_tracker.overall_start_time and self.result_tracker.overall_end_time:
            actual_time = self.result_tracker.overall_end_time - self.result_tracker.overall_start_time
            minutes, seconds = divmod(actual_time, 60)
            print(
                f"Total execution time: {int(minutes)}m {seconds:.1f}s (parallel mode)"
            )
        else:
            minutes, seconds = divmod(total_time, 60)
            print(f"Total execution time: {int(minutes)}m {seconds:.1f}s")

        # Print separator
        print("=" * 80)

    def _render_table(
        self,
        table_data: list[list[Any]],
        max_compile_time: float,
        max_upload_time: float,
        max_total_duration: float,
    ) -> None:
        """Render summary table with aligned columns and colors."""
        headers = ["File", "Status", "Compile (s)", "Upload (s)", "Total (s)", "Retries"]

        # Calculate column widths
        col_widths = [len(h) for h in headers]
        for row in table_data:
            for i, cell in enumerate(row):
                cell_str = self._format_cell(cell, i)
                col_widths[i] = max(col_widths[i], len(cell_str))

        col_widths[0] = max(col_widths[0], 30)

        # Print header
        header_line = " | ".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(headers))
        separator = "=" * len(header_line)

        print(separator)
        print(header_line)
        print("-" * len(header_line))

        # Print rows
        for row in table_data:
            self._print_row(
                row,
                col_widths,
                max_compile_time,
                max_upload_time,
                max_total_duration,
            )

        print(separator)

    def _format_cell(self, cell: object, column_idx: int) -> str:
        """Format a cell for display without colors.

        Returns Formatted string.
        """
        if column_idx == 1:  # Status column
            if cell == "success":
                return "✓ success"
            elif cell == "failed":
                return "✗ failed"
            elif cell == "interrupted":
                return "⚠ interrupted"
            else:
                return "⚠ skipped"
        elif isinstance(cell, float):
            return f"{cell:.1f}"
        elif column_idx == 5:  # Retry count (now a tuple: (retry_count, failure_type))
            if isinstance(cell, tuple):
                retry_count, failure_type = cell
                if failure_type == FailureType.PERMANENT:
                    return "skip"
                elif retry_count > 0:
                    return str(retry_count)
                else:
                    return "-"
            # Fallback for old format (shouldn't happen)
            return str(cell) if isinstance(cell, int) and cell > 0 else "-"
        return str(cell)

    def _print_row(
        self,
        row: list[Any],
        col_widths: list[int],
        max_compile_time: float,
        max_upload_time: float,
        max_total_duration: float,
    ) -> None:
        """Print a single table row with colors and alignment."""
        file, status, compile_time, upload_time, total_duration, retry_info = row

        # Status with color
        status_str = self._format_status(status)

        # Times with highlighting for max values
        compile_str = self._format_time(compile_time, max_compile_time)
        upload_str = self._format_time(upload_time, max_upload_time)
        total_str = self._format_time(total_duration, max_total_duration)

        # Retry info with color based on failure type
        if isinstance(retry_info, tuple):
            retry_count, failure_type = retry_info
            if failure_type == FailureType.PERMANENT:
                # Config error - use yellow "skip"
                retry_str = f"{Color.YELLOW.value}skip{Color.RESET.value}"
            elif retry_count > 0:
                # Normal retry - use yellow with count
                retry_str = f"{Color.YELLOW.value}{retry_count}{Color.RESET.value}"
            else:
                # Success or first-time failure - no color
                retry_str = "-"
        else:
            # Fallback for old format (shouldn't happen)
            retry_str = (
                f"{Color.YELLOW.value}{retry_info}{Color.RESET.value}"
                if retry_info > 0
                else "-"
            )

        # Cache stripped lengths to avoid repeated strip_ansi calls
        stripped_lengths = {
            'status': len(strip_ansi(status_str)),
            'compile': len(strip_ansi(compile_str)),
            'upload': len(strip_ansi(upload_str)),
            'total': len(strip_ansi(total_str)),
            'retry': len(strip_ansi(retry_str)),
        }

        # Construct line with proper padding for ANSI codes
        line = f"{file:<{col_widths[0]}} | "
        line += f"{status_str:<{col_widths[1] + len(status_str) - stripped_lengths['status']}} | "
        line += f"{compile_str:<{col_widths[2] + len(compile_str) - stripped_lengths['compile']}} | "
        line += f"{upload_str:<{col_widths[3] + len(upload_str) - stripped_lengths['upload']}} | "
        line += f"{total_str:<{col_widths[4] + len(total_str) - stripped_lengths['total']}} | "
        line += f"{retry_str:<{col_widths[5] + len(retry_str) - stripped_lengths['retry']}}"

        print(line)

    def _format_status(self, status: str) -> str:
        """Format status with color.

        Returns Colored status string.
        """
        if status == "success":
            return f"{Color.GREEN.value}✓ success{Color.RESET.value}"
        elif status == "failed":
            return f"{Color.RED.value}✗ failed{Color.RESET.value}"
        elif status == "interrupted":
            return f"{Color.YELLOW.value}⚠ interrupted{Color.RESET.value}"
        else:
            return f"{Color.YELLOW.value}⚠ skipped{Color.RESET.value}"

    def _format_time(self, time_val: object, max_time: float) -> str:
        """Format time value with highlighting for max.

        Returns Formatted time string with optional highlighting.
        """
        if isinstance(time_val, float):
            if time_val == max_time and time_val > 0:
                return f"{Color.YELLOW.value}{time_val:.1f}{Color.RESET.value}"
            return f"{time_val:.1f}"
        return "-"


# =============================================================================
# Progress Display Strategies
# =============================================================================


class SerialProgressDisplay:
    """Progress display for serial execution mode.

    Displays a todo list showing status of all files, with the current
    file highlighted. Uses color-coded status indicators.
    """

    def __init__(self, result_tracker: ResultTracker):
        """Initialize serial progress display."""
        self.result_tracker = result_tracker

    def show_progress(
        self,
        files_to_run: list[str],
        current_file: str | None = None,
    ) -> None:
        """Display todo list with current status."""
        print("\n" + "=" * 50)
        print_color(Color.BLUE, "EXECUTION TODO LIST")
        print("=" * 50)

        for i, file_path in enumerate(files_to_run):
            number = i + 1
            result = self.result_tracker.get_result(file_path)
            status = result["status"] if result else ExecutionStatus.PENDING

            time_str = ""
            if result and result.get("end_time") and result.get("start_time"):
                end_time = result["end_time"]
                start_time = result["start_time"]
                if end_time is not None and start_time is not None:
                    total_duration = end_time - start_time
                    time_str = f" [{total_duration:.1f}s]"

            if status == ExecutionStatus.IN_PROGRESS or (
                status == ExecutionStatus.PENDING and file_path == current_file
            ):
                print_color(Color.YELLOW, f"→ [{number}] {file_path} (IN PROGRESS...)")
            elif status == ExecutionStatus.PENDING:
                print(f"  [{number}] {file_path} (pending)")
            elif status == ExecutionStatus.SUCCESS:
                print_color(Color.GREEN, f"✓ [{number}] {file_path} (success){time_str}")
            elif status == ExecutionStatus.FAILED:
                print_color(Color.RED, f"✗ [{number}] {file_path} (failed){time_str}")
            elif status == ExecutionStatus.INTERRUPTED:
                print_color(
                    Color.YELLOW, f"⚠ [{number}] {file_path} (interrupted){time_str}"
                )

        print("=" * 50 + "\n")


class ParallelProgressDisplay:
    """Progress display for parallel execution mode.

    Displays a progress bar and list of currently running files.
    Updates continuously in a separate thread. Thread-safe output
    to prevent interleaved writes.
    """

    def __init__(self, config: RunnerConfig, result_tracker: ResultTracker, files_to_run: list[str]):
        """Initialize parallel progress display."""
        self.config = config
        self.result_tracker = result_tracker
        self.stdout_lock = threading.Lock()
        self.interrupted = False
        self.display_thread: threading.Thread | None = None

        # Calculate common prefix for relative path display
        self.common_prefix, self.common_prefix_depth = calculate_common_prefix(files_to_run)

    def start(self) -> None:
        """Start background progress display thread."""
        self.interrupted = False
        self.display_thread = threading.Thread(target=self._display_loop)
        self.display_thread.daemon = True
        self.display_thread.start()

    def stop(self) -> None:
        """Stop background progress display thread."""
        self.interrupted = True
        if self.display_thread:
            self.display_thread.join(timeout=self.config.DISPLAY_THREAD_TIMEOUT)

    def _display_loop(self) -> None:
        """Main loop for progress display (runs in background thread)."""
        last_display_lines = 0

        # Initial delay to let execution start
        time.sleep(self.config.DISPLAY_INITIAL_DELAY)

        while not self.interrupted:
            stats = self.result_tracker.get_stats()

            # Check if all completed
            if stats.completed >= stats.total and stats.total > 0:
                break

            # Skip if no data yet
            if stats.total == 0:
                time.sleep(self.config.PROGRESS_UPDATE_INTERVAL)
                continue

            # Clear previous display
            if last_display_lines > 0:
                with self.stdout_lock:
                    sys.stdout.write(f"\033[{last_display_lines}A")  # Move cursor up
                    sys.stdout.write("\033[J")  # Clear from cursor to end

            lines = self._build_progress_lines(stats)

            # Print all lines with thread-safe stdout access
            output = "\n".join(lines)
            with self.stdout_lock:
                sys.stdout.write(output + "\n")  # Add newline at end
                sys.stdout.flush()
            last_display_lines = len(lines) + 1  # +1 for the newline

            time.sleep(self.config.PROGRESS_UPDATE_INTERVAL)

        # Final clear
        if last_display_lines > 0:
            with self.stdout_lock:
                sys.stdout.write(f"\033[{last_display_lines}A")
                sys.stdout.write("\033[J")
                sys.stdout.flush()

    def _build_progress_lines(self, stats: ExecutionStats) -> list[str]:
        """Build lines for progress display.

        Returns List of lines to display.
        """
        lines = []

        # Progress bar
        bar = self._build_progress_bar(stats)
        lines.append(
            f"\nProgress: [{bar}] {stats.completed}/{stats.total} ({stats.progress_pct:.0f}%)"
        )
        lines.append("")

        # Currently running files
        if stats.in_progress > 0:
            lines.append("📋 Currently Running:")
            lines.extend(self._build_running_files_list())

        lines.append("")
        # Statistics with retrying count
        lines.append(
            f"✅ Success: {stats.success} | "
            f"❌ Failed: {stats.failed} | "
            f"🔄 Retrying: {stats.retrying} | "
            f"⏳ Remaining: {stats.pending}"
        )

        return lines

    def _build_progress_bar(self, stats: ExecutionStats) -> str:
        """Build progress bar string.

        Returns Progress bar string.
        """
        bar_length = self.config.PROGRESS_BAR_LENGTH
        filled = (
            int(bar_length * stats.completed / stats.total) if stats.total > 0 else 0
        )
        return "█" * filled + "░" * (bar_length - filled)

    def _format_path_for_display(self, file_path: str, max_length: int) -> str:
        """Format file path for display using relative paths or truncation.

        Strategy:
        1. If file fits, show full path
        2. If common prefix exists, use ../ notation to shrink display
        3. If no common prefix, use middle truncation with ...

        Returns Formatted path string, potentially with relative path or truncation.
        """
        # If full path fits, return it
        if len(file_path) <= max_length:
            return file_path

        parts = Path(file_path).parts
        filename = parts[-1]

        # Strategy 1: Use relative path notation if we have a common prefix
        if self.common_prefix and self.common_prefix_depth > 0:
            # Remove common prefix from path
            if file_path.startswith(self.common_prefix + "/"):
                relative_parts = parts[self.common_prefix_depth:]

                # Try with exact depth replacement (../../...)
                dots = "../" * self.common_prefix_depth
                candidate = dots + "/".join(relative_parts)
                if len(candidate) <= max_length:
                    return candidate

                # If still too long, add one more level (../../../...)
                # This effectively removes one more directory level
                if len(relative_parts) > 1:
                    dots = "../" * (self.common_prefix_depth + 1)
                    candidate = dots + "/".join(relative_parts[1:])
                    if len(candidate) <= max_length:
                        return candidate

                # If still too long, keep adding ../ and removing directories
                for extra_levels in range(2, len(relative_parts)):
                    dots = "../" * (self.common_prefix_depth + extra_levels)
                    candidate = dots + "/".join(relative_parts[extra_levels:])
                    if len(candidate) <= max_length:
                        return candidate

        # Strategy 2: No common prefix or relative path didn't work - use truncation
        # Try to keep first two directories + ... + filename
        if len(parts) > 3:
            first_two = '/'.join(parts[:2])
            candidate = f"{first_two}/.../{filename}"
            if len(candidate) <= max_length:
                return candidate

        # Try to keep first directory + ... + filename
        if len(parts) > 2:
            first_dir = parts[0]
            candidate = f"{first_dir}/.../{filename}"
            if len(candidate) <= max_length:
                return candidate

        # Last resort: just .../ and filename
        candidate = f".../{filename}"
        if len(candidate) <= max_length:
            return candidate

        # Even filename is too long, truncate it
        return f"...{filename[-(max_length-3):]}"

    def _build_running_files_list(self) -> list[str]:
        """Build list of currently running files.

        Returns List of formatted strings for running files.
        """
        lines = []
        worker_num = 1
        results = self.result_tracker.get_all_results()

        for file_path, result in results.items():
            if result["status"] == ExecutionStatus.IN_PROGRESS:
                elapsed = (
                    time.time() - result["start_time"] if result["start_time"] else 0
                )
                # Use full relative path with smart truncation
                display_path = self._format_path_for_display(
                    file_path, self.config.MAX_FILENAME_DISPLAY
                )
                retry_count = result.get("retry_count", 0)

                # Format retry status with emoji
                if retry_count > 0:
                    retry_str = f" 🔄 [Retry {retry_count}/{self.config.max_retries}]"
                else:
                    retry_str = ""

                # Format time with appropriate units
                if elapsed < 60:
                    time_str = f"{elapsed:>5.0f}s"
                else:
                    minutes = int(elapsed // 60)
                    seconds = int(elapsed % 60)
                    time_str = f"{minutes:>2}m{seconds:02}s"

                lines.append(
                    f"  🔧 Worker {worker_num}: {display_path:<{self.config.MAX_FILENAME_DISPLAY}} "
                    f"[⏱️  {time_str}]{retry_str}"
                )
                worker_num += 1
                if worker_num > self.config.parallel_workers:
                    break

        return lines

    def write_to_console(self, message: str) -> None:
        """Thread-safe console write.

        All console output in parallel mode should use this method
        to avoid deadlock with the progress display thread.
        """
        with self.stdout_lock:
            sys.stdout.write(message)
            sys.stdout.flush()


# =============================================================================
# Process Lifecycle Management
# =============================================================================


class ProcessManager:
    """Manages process lifecycle for ESPHome executions.

    This class handles process creation, monitoring, and termination,
    abstracting the differences between pty-based (serial) and
    subprocess-based (parallel) execution.
    """

    def __init__(self, config: RunnerConfig):
        """Initialize the process manager.

        Creates failure analyzer if enabled in configuration (Dependency Injection).
        """
        self.config = config
        self.current_pid: int | None = None
        self.running_processes: dict[str, subprocess.Popen[str]] = {}
        self.processes_lock = threading.Lock()
        self._interrupted = threading.Event()

        # Initialize failure analyzer based on configuration (DIP)
        if config.enable_failure_analysis:
            self.failure_analyzer: ESPHomeFailureAnalyzer | None = ESPHomeFailureAnalyzer()
        else:
            self.failure_analyzer = None

    @property
    def interrupted(self) -> bool:
        """Check if execution has been interrupted (thread-safe).

        Returns True if interrupted, False otherwise.
        """
        return self._interrupted.is_set()

    @interrupted.setter
    def interrupted(self, value: bool) -> None:
        """Set interrupted status (thread-safe)."""
        if value:
            self._interrupted.set()
        else:
            self._interrupted.clear()

    def build_command(self, file_path: str) -> list[str]:
        """Build ESPHome command for execution.

        Returns Command as list of strings.
        """
        if self.config.compile_only:
            command = ["esphome", "compile", file_path]
        else:
            command = ["esphome", "run", file_path]
            if self.config.no_logs_arg:
                command.append(self.config.no_logs_arg)
        return command

    def format_log_header(
        self,
        file_path: str,
        retry_count: int,
        execution_mode: str,
    ) -> str:
        """Format log header with timestamp and execution metadata.

        Returns Formatted log header string.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        command = " ".join(self.build_command(file_path))
        max_retries = self.config.max_retries
        attempt_number = retry_count + 1
        total_attempts = max_retries + 1

        if retry_count == 0:
            header_title = "EXECUTION START"
        else:
            header_title = f"RETRY ATTEMPT {retry_count}"

        header = "=" * 80 + "\n"
        header += f"=== {header_title} ===\n"
        header += f"Time: {timestamp}\n"
        header += f"Attempt: {attempt_number}/{total_attempts}\n"
        header += f"Mode: {execution_mode}\n"
        header += f"Command: {command}\n"
        header += "=" * 80 + "\n"
        header += "\n"  # Add blank line after header for better readability

        return header

    def run_with_pty(
        self,
        file_path: str,
        log_path: Path,
        retry_count: int = 0,
        interrupted: bool = False,
    ) -> ExecutionResult:
        """Execute file using pty (preserves colors).

        This method uses pty.fork() to create a pseudo-terminal, which
        preserves ANSI color codes in the output. Used for serial mode.

        Returns ExecutionResult with execution details.
        """
        command = self.build_command(file_path)
        result = create_execution_result(
            status=ExecutionStatus.IN_PROGRESS,
            start_time=time.time(),
            retry_count=retry_count,
        )

        fd = None  # Initialize fd to ensure cleanup in finally block
        try:
            pid, fd = pty.fork()

            if pid == 0:  # Child process
                try:
                    os.execvp(command[0], command)
                except FileNotFoundError:
                    sys.stderr.write(f"Error: command not found: {command[0]}\n")
                    os._exit(127)
                except Exception as e:
                    sys.stderr.write(f"Error executing command: {e}\n")
                    os._exit(126)
            else:  # Parent process
                self.current_pid = pid
                exit_code = self._monitor_pty_process(
                    fd, file_path, log_path, retry_count, result
                )

                if interrupted and result["status"] == ExecutionStatus.IN_PROGRESS:
                    result["status"] = ExecutionStatus.INTERRUPTED
                elif exit_code == 0:
                    result["status"] = ExecutionStatus.SUCCESS
                else:
                    result["status"] = ExecutionStatus.FAILED
                    # Analyze failure type for smart retry decision
                    if self.failure_analyzer:
                        result["failure_type"] = self.failure_analyzer.analyze(log_path)

        except Exception as e:
            logger.error(f"Failed to execute {file_path} with pty: {e}", exc_info=True)
            result["status"] = ExecutionStatus.FAILED
        finally:
            # Ensure file descriptor is always closed
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
            if result.get("end_time") is None:
                result["end_time"] = time.time()
            self.current_pid = None

        return result

    def _monitor_pty_process(
        self,
        fd: int,
        file_path: str,
        log_path: Path,
        retry_count: int,
        result: ExecutionResult,
    ) -> int:
        """Monitor pty process and extract timing information.

        Returns Process exit code.
        """
        exit_code = 1
        compile_time = 0.0
        upload_time = 0.0
        line_buffer = b""

        try:
            log_mode = "ab" if retry_count > 0 else "wb"
            with open(log_path, log_mode) as log_file:
                # Write formatted header with timestamp
                header = self.format_log_header(file_path, retry_count, "Serial")
                if retry_count > 0:
                    log_file.write(b"\n\n")
                log_file.write(header.encode())
                log_file.flush()  # Ensure header is written before subprocess output

                while True:
                    try:
                        data = os.read(fd, 1024)
                    except OSError:
                        break
                    if not data:
                        break

                    log_file.write(data)
                    sys.stdout.write(data.decode(sys.stdout.encoding, errors="replace"))
                    sys.stdout.flush()

                    # Process lines for timing info
                    line_buffer += data
                    while b"\n" in line_buffer:
                        line_bytes, line_buffer = line_buffer.split(b"\n", 1)
                        line_str = line_bytes.decode("utf-8", errors="replace").strip()

                        if compile_match := RegexPatterns.COMPILE_TIME.search(line_str):
                            compile_time = float(compile_match.group(1))

                        if upload_match := RegexPatterns.UPLOAD_TIME.search(line_str):
                            upload_time = float(upload_match.group(1))

            # Wait for process to finish
            if self.current_pid:
                _, exit_status = os.waitpid(self.current_pid, 0)
                if os.WIFEXITED(exit_status):
                    exit_code = os.WEXITSTATUS(exit_status)

        finally:
            # fd is closed by caller in run_with_pty
            result["compile_time"] = compile_time
            result["upload_time"] = upload_time

        return exit_code

    def run_with_subprocess(
        self,
        file_path: str,
        log_path: Path,
        retry_count: int = 0,
        start_time: float | None = None,
    ) -> ExecutionResult:
        """Execute file using subprocess (for parallel mode).

        This method uses subprocess.Popen for better parallel compatibility.
        Output is written to log files instead of console.

        Returns ExecutionResult with execution details.
        """
        command = self.build_command(file_path)
        result = create_execution_result(
            status=ExecutionStatus.IN_PROGRESS,
            start_time=start_time if start_time is not None else time.time(),
            retry_count=retry_count,
        )

        try:
            log_mode = "a" if retry_count > 0 else "w"
            with open(log_path, log_mode) as log_file:
                # Write formatted header with timestamp
                header = self.format_log_header(file_path, retry_count, "Parallel")
                if retry_count > 0:
                    log_file.write("\n\n")
                log_file.write(header)
                log_file.flush()  # Ensure header is written before subprocess output

                proc = subprocess.Popen(
                    command,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )

                # Track running process (thread-safe)
                with self.processes_lock:
                    self.running_processes[file_path] = proc

                try:
                    # Wait for completion with polling for interrupt responsiveness
                    exit_code = self._wait_for_process(proc)
                    if exit_code is None:
                        # Process was interrupted or timed out
                        result["status"] = ExecutionStatus.INTERRUPTED if self.interrupted else ExecutionStatus.TIMEOUT
                        result["end_time"] = time.time()
                        return result
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                    result["status"] = ExecutionStatus.TIMEOUT
                    result["end_time"] = time.time()
                    return result
                finally:
                    with self.processes_lock:
                        self.running_processes.pop(file_path, None)

            # Parse log for timing info
            compile_time, upload_time = self._parse_timing_from_log(log_path)
            result["compile_time"] = compile_time
            result["upload_time"] = upload_time

            # Set status based on exit code
            if exit_code == 0:
                result["status"] = ExecutionStatus.SUCCESS
            else:
                result["status"] = ExecutionStatus.FAILED
                # Analyze failure type for smart retry decision
                if self.failure_analyzer:
                    result["failure_type"] = self.failure_analyzer.analyze(log_path)

        except Exception as e:
            logger.error(f"Failed to execute {file_path} with subprocess: {e}", exc_info=True)
            result["status"] = ExecutionStatus.FAILED
            with self.processes_lock:
                self.running_processes.pop(file_path, None)
        finally:
            result["end_time"] = time.time()

        return result

    def _parse_timing_from_log(self, log_path: Path) -> tuple[float, float]:
        """Parse compile and upload times from log file.

        Returns Tuple of (compile_time, upload_time).
        """
        compile_time = 0.0
        upload_time = 0.0

        try:
            with open(log_path, "r", encoding="utf-8") as log_file:
                for line in log_file:
                    if compile_match := RegexPatterns.COMPILE_TIME.search(line):
                        compile_time = float(compile_match.group(1))
                    if upload_match := RegexPatterns.UPLOAD_TIME.search(line):
                        upload_time = float(upload_match.group(1))
        except (OSError, ValueError):
            pass

        return compile_time, upload_time

    def _wait_for_process(self, proc: subprocess.Popen[str]) -> int | None:
        """Wait for process completion, polling for interrupts and timeout.

        Returns the exit code, or None if interrupted/timeout.
        """
        poll_interval = self.config.PROCESS_POLL_INTERVAL
        elapsed = 0.0
        max_wait = self.config.PROCESS_WAIT_TIMEOUT

        while elapsed < max_wait:
            # Check if interrupted
            if self.interrupted:
                proc.terminate()
                try:
                    proc.wait(timeout=self.config.PROCESS_TERM_TIMEOUT)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=1.0)
                return None

            # Check if process finished
            exit_code = proc.poll()
            if exit_code is not None:
                return exit_code

            # Wait before next poll
            time.sleep(poll_interval)
            elapsed += poll_interval

        # Timeout - kill process
        proc.kill()
        proc.wait()
        return None

    def terminate_current_process(self) -> None:
        """Terminate the current process (serial mode).

        Sends SIGTERM for graceful shutdown. If process doesn't exist,
        fails silently. Handles TOCTOU race condition where PID might
        be reused by the system.
        """
        pid = self.current_pid
        if pid is None:
            return

        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass  # Process already finished
        except PermissionError:
            pass  # PID has been reused by another process

    def terminate_process(self, proc: subprocess.Popen[str]) -> None:
        """Terminate a single subprocess.

        Attempts graceful termination with SIGTERM, falling back to
        SIGKILL if necessary.
        """
        try:
            proc.terminate()
            proc.wait(timeout=self.config.PROCESS_TERM_TIMEOUT)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
                proc.wait(timeout=2.0)
            except (ProcessLookupError, PermissionError, OSError):
                pass
        except (ProcessLookupError, PermissionError, OSError):
            pass

    def terminate_all_processes(self) -> None:
        """Terminate all running processes (parallel mode).

        Iterates through all tracked processes and terminates them.
        Failures are handled gracefully.
        """
        # Create snapshot of processes to terminate (thread-safe)
        with self.processes_lock:
            processes_snapshot = list(self.running_processes.items())

        # Terminate processes outside the lock
        for _, proc in processes_snapshot:
            try:
                self.terminate_process(proc)
            except Exception:
                # Silently ignore termination errors during cleanup
                pass

    def cleanup_processes(self) -> None:
        """Clean up any remaining processes.

        Final cleanup pass to ensure no zombie processes remain.
        """
        # Create snapshot of processes (thread-safe)
        with self.processes_lock:
            processes_snapshot = list(self.running_processes.items())

        # Clean up processes outside the lock
        for _, proc in processes_snapshot:
            try:
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=self.config.PROCESS_CLEANUP_TIMEOUT)
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                except (ProcessLookupError, PermissionError, OSError):
                    pass
            except (ProcessLookupError, PermissionError, OSError):
                pass


# =============================================================================
# Execution Strategies
# =============================================================================


class _ExecutorBase:
    """Shared helpers for the serial/parallel executors.

    The retry loops themselves stay separate -- serial narrates to the console
    and blocks between attempts, parallel keeps IN_PROGRESS bookkeeping and
    interruptible sleeps -- but the pieces that must agree (log placement,
    warmup-skip, permanent-failure handling) live here so they cannot drift.
    """

    config: RunnerConfig
    result_tracker: ResultTracker

    def _log_path_for(self, file_path: str) -> Path:
        """Log path mirroring the yaml's directory structure, parents ensured."""
        log_path = self.config.log_dir / Path(file_path).with_suffix(".log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        return log_path

    def _already_succeeded(self, file_path: str) -> bool:
        """True for files already marked SUCCESS (e.g. by the warmup phase)."""
        existing = self.result_tracker.get_result(file_path)
        return existing is not None and existing["status"] == ExecutionStatus.SUCCESS

    def _note_permanent_failure(self, file_path: str) -> None:
        """Record the permanent-failure analysis note in the file's log."""
        append_failure_analysis_note(
            self._log_path_for(file_path), FailureType.PERMANENT
        )


class SerialExecutor(_ExecutorBase):
    """Executes files sequentially, one at a time.

    This executor runs files in serial mode, displaying output to console
    in real-time. It uses pty for process execution to preserve ANSI colors.
    """

    def __init__(
        self,
        config: RunnerConfig,
        process_manager: ProcessManager,
        result_tracker: ResultTracker,
        progress_display: SerialProgressDisplay,
    ):
        """Initialize serial executor."""
        self.config = config
        self.process_manager = process_manager
        self.result_tracker = result_tracker
        self.progress_display = progress_display
        self.interrupted = False

    def set_warmup_gate(self, warmup: "WarmupPhase") -> None:
        """No-op: warmup is skipped in serial mode (no race to mitigate)."""
        del warmup

    def execute(self, files: list[str]) -> None:
        """Execute files sequentially."""
        try:
            for file_path in files:
                if self.interrupted:
                    print_color(
                        Color.YELLOW, "Halting further execution due to interrupt."
                    )
                    break

                self.progress_display.show_progress(files, current_file=file_path)
                self._execute_file_with_retry(file_path)

        except KeyboardInterrupt:
            self._handle_interrupt()

    def _execute_file_with_retry(self, file_path: str) -> None:
        """Execute a single file with retry logic."""
        # Skip files already marked SUCCESS by the warmup phase
        if self._already_succeeded(file_path):
            return

        retry_count = 0
        success = False

        while retry_count <= self.config.max_retries and not success:
            if retry_count > 0:
                # Calculate exponential backoff delay
                delay = self.config.calculate_retry_delay(retry_count)
                print_color(
                    Color.YELLOW,
                    f"\n=== RETRY {retry_count}/{self.config.max_retries} for {file_path} "
                    f"(waiting {delay:.1f}s) ===\n"
                )
                time.sleep(delay)

            result = self._execute_single_file(file_path, retry_count)
            self.result_tracker.update_result(file_path, result)

            if result["status"] == ExecutionStatus.SUCCESS:
                success = True
                print_color(Color.GREEN, f"\n✓ Success: {file_path}")
            elif result.get("failure_type") == FailureType.PERMANENT:
                self._note_permanent_failure(file_path)
                print_color(
                    Color.YELLOW,
                    f"\n⚠ Configuration error detected in {file_path}, skipping retry"
                )
                print_color(Color.RED, f"✗ Failed: {file_path} (config error)")
                break
            elif retry_count < self.config.max_retries and not self.interrupted:
                retry_count += 1
            else:
                if result["status"] == ExecutionStatus.FAILED:
                    print_color(Color.RED, f"\n✗ Failed: {file_path}")
                break

    def _execute_single_file(self, file_path: str, retry_count: int) -> ExecutionResult:
        """Execute a single file.

        Returns ExecutionResult with execution details.
        """
        log_path = self._log_path_for(file_path)

        print("=" * 50)
        print_color(Color.BLUE, f"Running: {file_path}")
        if retry_count > 0:
            print_color(Color.YELLOW, f"(Retry attempt {retry_count})")
        print("=" * 50)

        result = self.process_manager.run_with_pty(
            file_path, log_path, retry_count, self.interrupted
        )

        return result

    def _handle_interrupt(self) -> None:
        """Handle keyboard interrupt."""
        self.interrupted = True
        print_color(Color.YELLOW, "\nInterrupt signal received! Stopping...")
        print_color(
            Color.YELLOW,
            f"Terminating current task (PID: {self.process_manager.current_pid})...",
        )
        self.process_manager.terminate_current_process()


class ParallelExecutor(_ExecutorBase):
    """Executes files in parallel using multiple workers.

    This executor runs multiple files simultaneously using ThreadPoolExecutor.
    Output is written to log files instead of console. Progress is displayed
    via a background thread.
    """

    def __init__(
        self,
        config: RunnerConfig,
        process_manager: ProcessManager,
        result_tracker: ResultTracker,
        progress_display: ParallelProgressDisplay,
    ):
        """Initialize parallel executor."""
        self.config = config
        self.process_manager = process_manager
        self.result_tracker = result_tracker
        self.progress_display = progress_display
        self.interrupted = False
        # Slow-start state: enforce minimum gap between task starts to mitigate
        # cold-cache toolchain install races. Mutable (unlike the frozen
        # config) so the runner can zero it when the warmup stamp shows the
        # toolchain caches are already warm.
        self.slow_start_interval = config.slow_start_interval
        self.last_task_start_time: float | None = None
        self.start_time_lock = threading.Lock()
        # Streaming warmup gate: workers block here until their toolchain
        # bucket has been warmed up by WarmupPhase.
        self._warmup_gate: "WarmupPhase | None" = None

    def set_warmup_gate(self, warmup: "WarmupPhase") -> None:
        """Register a WarmupPhase so workers can block on per-bucket readiness.

        Called by ESPHomeRunner before execute() when streaming warmup is
        active. Workers call warmup.wait_for_file() at the start of each
        file's execution; it's a no-op when streaming is disabled.
        """
        self._warmup_gate = warmup

    def _wait_for_slow_start(self) -> None:
        """Block until the slow-start interval has elapsed since the last task start.

        Called at the top of every `_execute_file_with_retry` invocation. Thread-safe
        via `start_time_lock` — holding the lock while sleeping is intentional, it
        forces other workers to queue up and stagger their starts.
        """
        if self.slow_start_interval <= 0:
            return
        with self.start_time_lock:
            if self.last_task_start_time is not None:
                elapsed = time.time() - self.last_task_start_time
                if elapsed < self.slow_start_interval:
                    self._interruptible_sleep(
                        self.slow_start_interval - elapsed
                    )
            self.last_task_start_time = time.time()

    def execute(self, files: list[str]) -> None:
        """Execute files in parallel."""
        # Import ThreadPoolExecutor here to avoid issues
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Initialize all files as pending
        self.result_tracker.initialize_results(files)

        executor = None
        try:
            # Create executor manually for better shutdown control
            executor = ThreadPoolExecutor(max_workers=self.config.parallel_workers)

            # Start progress display
            if hasattr(self.progress_display, "start"):
                self.progress_display.start()
                # Give the display thread time to start
                time.sleep(self.config.DISPLAY_INITIAL_DELAY)

            # Submit all tasks immediately
            futures = {
                executor.submit(self._execute_file_with_retry, file_path): file_path
                for file_path in files
            }

            # Wait for completion
            for future in as_completed(futures):
                if self.interrupted:
                    break
                try:
                    future.result()
                except Exception as e:
                    if not self.interrupted:
                        file_path = futures[future]
                        error_msg = f"{Color.RED.value}\nError processing {file_path}: {e}{Color.RESET.value}\n"
                        # Use thread-safe console write if available
                        if hasattr(self.progress_display, 'write_to_console'):
                            self.progress_display.write_to_console(error_msg)
                        else:
                            print_color(Color.RED, f"\nError processing {file_path}: {e}")

        except KeyboardInterrupt:
            self._handle_interrupt()
        finally:
            # Stop progress display first
            if hasattr(self.progress_display, "stop"):
                self.progress_display.stop()

            # Terminate all running processes immediately
            try:
                self.process_manager.terminate_all_processes()
            except Exception:
                pass

            # Shutdown executor gracefully with brief timeout
            if executor is not None:
                try:
                    # Give threads a brief moment to notice the interrupt flag
                    time.sleep(self.config.EXECUTOR_SHUTDOWN_DELAY)
                    # Try graceful shutdown first
                    executor.shutdown(wait=True, cancel_futures=True)
                except Exception:
                    # If graceful shutdown fails, force it
                    try:
                        executor.shutdown(wait=False, cancel_futures=True)
                    except Exception:
                        pass

            # Final cleanup with short timeout
            try:
                self.process_manager.cleanup_processes()
            except Exception:
                pass

    def _execute_file_with_retry(self, file_path: str) -> None:
        """Execute a single file with retry logic."""
        # Skip files already marked SUCCESS by the warmup phase
        if self._already_succeeded(file_path):
            return

        # Streaming warmup gate: block until this file's toolchain bucket has
        # been pre-compiled. No-op when warmup is disabled or already done.
        if self._warmup_gate is not None:
            self._warmup_gate.wait_for_file(file_path)
            # Warmup may have marked the rep SUCCESS while we were blocked.
            if self._already_succeeded(file_path):
                return
            # Bail out early if we were interrupted during the wait.
            if self.interrupted:
                return

        # Stagger starts to avoid concurrent pioarduino install_esptool races
        self._wait_for_slow_start()

        retry_count = 0
        last_result = None

        while retry_count <= self.config.max_retries:
            # Check for interrupt before starting execution
            if self.interrupted:
                result = create_execution_result(status=ExecutionStatus.INTERRUPTED)
                self.result_tracker.update_result(file_path, result)
                return

            # Determine start_time: preserve from first attempt across retries
            if retry_count == 0:
                start_time: float = time.time()
            else:
                # Safely get start_time from last_result, fallback to current time if missing
                last_start_time = last_result.get("start_time") if last_result else None
                start_time = last_start_time if last_start_time is not None else time.time()

            # Update status to IN_PROGRESS immediately before execution
            in_progress_result = create_execution_result(
                status=ExecutionStatus.IN_PROGRESS,
                start_time=start_time,
                retry_count=retry_count
            )
            self.result_tracker.update_result(file_path, in_progress_result)

            result = self._execute_single_file(file_path, retry_count, start_time)
            last_result = result

            # Check for interrupt
            if self.interrupted or result["status"] == ExecutionStatus.INTERRUPTED:
                result["status"] = ExecutionStatus.INTERRUPTED
                self.result_tracker.update_result(file_path, result)
                return

            # Check for success
            if result["status"] == ExecutionStatus.SUCCESS:
                self.result_tracker.update_result(file_path, result)
                return

            # Check for permanent failure (skip retry for config errors)
            if result.get("failure_type") == FailureType.PERMANENT:
                self._note_permanent_failure(file_path)
                self.result_tracker.update_result(file_path, result)
                return

            # Failed - check if should retry
            if retry_count < self.config.max_retries:
                # Still have retries left - keep IN_PROGRESS status but update retry_count
                retry_count += 1
                in_progress_result["retry_count"] = retry_count
                self.result_tracker.update_result(file_path, in_progress_result)
                # Calculate exponential backoff delay and sleep in short intervals
                delay = self.config.calculate_retry_delay(retry_count)
                self._interruptible_sleep(delay)
            else:
                # No more retries - update to final FAILED status
                self.result_tracker.update_result(file_path, result)
                return

    def _execute_single_file(
        self, file_path: str, retry_count: int, start_time: float
    ) -> ExecutionResult:
        """Execute a single file.

        Returns ExecutionResult with execution details.
        """
        result = self.process_manager.run_with_subprocess(
            file_path, self._log_path_for(file_path), retry_count, start_time
        )

        return result

    def _interruptible_sleep(self, duration: float) -> None:
        """Sleep in short intervals to allow interrupt checking."""
        interval = self.config.INTERRUPT_POLL_INTERVAL
        elapsed = 0.0
        while elapsed < duration and not self.interrupted:
            sleep_time = min(interval, duration - elapsed)
            time.sleep(sleep_time)
            elapsed += sleep_time

    def _handle_interrupt(self) -> None:
        """Handle keyboard interrupt."""
        if self.interrupted:
            # Second interrupt - force exit
            print_color(Color.RED, "\n\nForce exit requested!")
            sys.exit(130)

        self.interrupted = True
        self.process_manager.interrupted = True  # Signal process manager to stop
        print("\n")  # Move to new line after progress display
        print_color(Color.YELLOW, "Interrupt signal received! Stopping all workers...")
        print_color(Color.YELLOW, "(Press Ctrl+C again to force exit)")

        # Cancel any in-flight streaming warmup so workers blocked in
        # wait_for_file() unblock immediately and the warmup subprocess
        # gets terminated instead of running to natural completion.
        if self._warmup_gate is not None:
            try:
                self._warmup_gate.cancel()
            except Exception:
                pass

        # Terminate all running processes
        try:
            self.process_manager.terminate_all_processes()
        except Exception:
            pass  # Ignore errors during interrupt


def create_executor(
    config: RunnerConfig,
    process_manager: ProcessManager,
    result_tracker: ResultTracker,
) -> "SerialExecutor | ParallelExecutor":
    """Create the executor (and its progress display) for the configured mode."""
    if config.parallel_workers > 0:
        return ParallelExecutor(
            config=config,
            process_manager=process_manager,
            result_tracker=result_tracker,
            progress_display=ParallelProgressDisplay(
                config=config,
                result_tracker=result_tracker,
                files_to_run=config.files_to_run,
            ),
        )
    return SerialExecutor(
        config=config,
        process_manager=process_manager,
        result_tracker=result_tracker,
        progress_display=SerialProgressDisplay(result_tracker=result_tracker),
    )


# =============================================================================
# Main Runner Coordinator
# =============================================================================


class ESPHomeRunner:
    """Main coordinator for ESPHome multi-run execution.

    This class follows the Single Responsibility Principle by delegating
    specific responsibilities to specialized components. It coordinates the
    overall execution flow by composing these components together.
    """

    def __init__(self, config: RunnerConfig):
        """Initialize the runner with dependency injection.

        Creates all necessary components and wires them together using
        composition. This follows the Dependency Inversion Principle by
        depending on abstractions (protocols) rather than concrete classes.

        Raises ConfigurationError if log directory cannot be created.
        """
        self.config = config

        # Create log directory
        try:
            self.config.log_dir.mkdir(exist_ok=True)
        except OSError as e:
            print_color(
                Color.RED,
                f"Error: Cannot create log directory '{self.config.log_dir}': {e}",
            )
            sys.exit(1)

        # Initialize components (Dependency Injection)
        self.file_filter = FileFilter(config.exclude_file)
        self.process_manager = ProcessManager(config)
        self.result_tracker = ResultTracker()

        # Create executor using factory (Dependency Inversion Principle)
        self.executor = create_executor(
            config=config,
            process_manager=self.process_manager,
            result_tracker=self.result_tracker,
        )

        # Warmup phase (runs between file-filter and executor dispatch)
        self.warmup = WarmupPhase(config)

        # Files to execute (will be populated after filtering)
        self.files_to_run: list[str] = []

    def run(self) -> None:
        """Execute the main runner workflow.

        This is the main entry point that orchestrates the entire execution:
        1. Filter files based on exclusion patterns
        2. Display execution mode and configuration
        3. Execute files using selected strategy
        4. Display final summary

        The method delegates specific tasks to specialized components,
        maintaining a high level of abstraction.
        """
        # Step 1: Filter files
        self.files_to_run = self.file_filter.apply_filters(
            self.config.files_to_run, verbose=True
        )

        if not self.files_to_run:
            print_color(
                Color.RED, "Error: All files were excluded. No files to process."
            )
            return

        # Step 2: Display execution information
        self._print_header()

        # Start the wall-clock timer here so the final "Total execution time"
        # covers the whole run including warmup, not just parallel dispatch.
        self.result_tracker.overall_start_time = time.time()

        # Step 2.5: Warmup phase — populate PIO toolchain cache.
        # begin() runs the probe synchronously and either:
        #   - returns a terminal outcome (warmup disabled / serial / cache hit),
        #     in which case we apply post-warmup actions immediately, or
        #   - starts background compilation and returns None; workers will
        #     gate on per-bucket events as each rep finishes, so dispatch
        #     can overlap with warmup instead of waiting for it to finish.
        self.executor.set_warmup_gate(self.warmup)
        warmup_outcome = self.warmup.begin(
            self.files_to_run, self.result_tracker
        )
        streaming = warmup_outcome is None
        if not streaming and warmup_outcome is not None:
            # Terminal outcome — replicate the legacy post-warmup actions.
            for rep in warmup_outcome.reps_compiled:
                self.result_tracker.update_result(
                    rep,
                    create_execution_result(
                        status=ExecutionStatus.SUCCESS,
                        retry_count=0,
                    ),
                )
            if warmup_outcome.success and not warmup_outcome.disabled:
                os.environ["ESPHOME_SKIP_CLEAN_BUILD"] = "1"
            if warmup_outcome.cache_hit and isinstance(
                self.executor, ParallelExecutor
            ):
                # Toolchain caches proven warm -- staggered starts would only
                # add idle time; the races they mitigate need a cold cache.
                self.executor.slow_start_interval = 0.0

        # Step 3: Execute files
        try:
            self.executor.execute(self.files_to_run)
        finally:
            # Streaming warmup may still be running if the executor exited
            # early (e.g. Ctrl-C). Always join so we don't leave a daemon
            # subprocess running, and so reps_compiled / success reflect the
            # final state for the summary.
            if streaming:
                self.warmup.finish()
            # Always record end time and display summary, even on error
            self.result_tracker.overall_end_time = time.time()

            # Step 4: Display summary
            print_color(Color.BLUE, "\nExecution finished. Generating summary...")

            # Create renderer and display summary (Separation of Concerns)
            renderer = ResultSummaryRenderer(self.result_tracker)
            renderer.print_summary(
                self.files_to_run,
                self.config.parallel_workers,
                interrupted=getattr(self.executor, "interrupted", False),
            )

    def _print_header(self) -> None:
        """Print execution header with configuration details."""
        print_color(Color.BLUE, "ESPHome Multi-Run Script")

        if self.config.parallel_workers > 0:
            mode_msg = f"[Parallel Mode - {self.config.parallel_workers} workers]"
            print_color(Color.BLUE, mode_msg)
            if self.config.slow_start_interval > 0:
                print_color(
                    Color.BLUE,
                    f"[Slow start: {self.config.slow_start_interval:.1f}s between task starts]"
                )

        if self.config.compile_only:
            print_color(Color.YELLOW, "[Compile-only mode - no uploads]")

        # Display failure analysis status
        if self.config.enable_failure_analysis:
            print_color(
                Color.GREEN,
                "[Smart failure analysis: ENABLED - config errors skip retry]"
            )
        else:
            print_color(
                Color.YELLOW,
                "[Smart failure analysis: DISABLED - all errors will retry]"
            )

        # Display warmup status — order matches WarmupPhase.begin() so the
        # header never claims a state begin() won't honor.
        version = self.config.esphome_version or "unknown"
        if not self.config.warmup_enabled:
            print_color(Color.YELLOW, "[Warmup: disabled]")
        elif read_warmup_stamp(
            self.config.warmup_cache_path, self.config.toolchain_fingerprint
        ):
            print_color(Color.BLUE, f"[Warmup: cache hit for ESPHome v{version}]")
        elif self.config.warmup_cache_path.is_file():
            print_color(
                Color.BLUE,
                f"[Warmup: toolchain changed — re-warming for ESPHome v{version}]",
            )
        elif self.config.parallel_workers <= 1:
            print_color(
                Color.YELLOW,
                "[Warmup: skipped (single worker — no race to mitigate)]"
            )
        else:
            print_color(Color.BLUE, f"[Warmup: enabled (ESPHome v{version})]")

        print(f"Starting at: {datetime.now()}")


# =============================================================================
# CLI Argument Parsing and Entry Point
# =============================================================================


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns Parsed arguments namespace.
    """
    description = """ESPHome Multi-Run Tool - Batch compile and upload multiple ESPHome configurations

BASIC USAGE:
  %(prog)s file1.yaml file2.yaml          Run specific files
  %(prog)s *.yaml                         Run files matching pattern
  %(prog)s -j 4 -p "*.yaml"               Run with 4 parallel workers

EXECUTION MODES:
  Serial (default):  Files run one by one with live output to console
  Parallel (-j N):   ⚠️  EXPERIMENTAL: N files run simultaneously, output saved to logs/ directory

PARALLEL MODE OPTIMIZATIONS:
  ✓ Exponential Backoff: Retry delays increase exponentially (3s → 6s → 12s → ...)
    - Gives system more time to recover from transient failures
    - Reduces retry-induced load
  ⚠ Resource contention may still occur with high worker counts
  ⚠ Consider reducing worker count (-j) if experiencing frequent failures

COMMON EXAMPLES:
  # Run all YAML files serially with live output
  %(prog)s *.yaml

  # Parallel compile and upload 4 files
  %(prog)s -j 4 *.yaml

  # Parallel compile only (no upload)
  %(prog)s -j 4 -c *.yaml

  # Multi-level directory examples
  %(prog)s examples/*/*.yaml                      # All YAML files in examples subdirectories
  %(prog)s examples/Brand/*/*.yaml                # All Brand configurations
  %(prog)s examples/*/Category/*.yaml             # All category configs across brands
  %(prog)s -j 4 -c examples/Brand/*/*.yaml        # Parallel compile all Brand configs

  # Multiple patterns (use -p flag for each)
  %(prog)s -j 4 -p "examples/BrandA/*/*.yaml" -p "examples/BrandB/*/*.yaml"

  # Specific directories
  %(prog)s -d examples/BrandA/CategoryA -d examples/BrandB/CategoryA

  # Disable smart failure analysis (retry all errors)
  %(prog)s -j 4 -c -F examples/*/*.yaml

EXCLUSION FILE FORMAT:
  Use glob patterns in the exclusion file (default: .esphome-run-exclude):
    # Comment lines start with #
    test-*.yaml          # Exclude all test files
    obsolete-device.yaml # Exclude specific file
    *-backup.yaml        # Exclude all backup files

  Default exclusion patterns (when no .esphome-run-exclude file exists):
    secrets.yaml         # ESPHome secrets file
    secrets.yml
    .*.yaml              # Hidden YAML files
    .*.yml

  Note: If .esphome-run-exclude exists, ONLY patterns in the file are used.
        To keep default behavior, add the patterns above to your exclude file.

FEATURES:
  ✓ Smart failure analysis (skips retry on config errors, use -F to disable)
  ✓ Exponential backoff retry (configurable with -r/--max-retries, default: 3)
  ✓ Default exclusion patterns (auto-excludes secrets.yaml when no exclude file)
  ✓ Preserves directory structure in logs/ (mirrors your source structure)
  ✓ Color-coded output and progress tracking
  ✓ Detailed execution summary with timing statistics
  ✓ Graceful interrupt handling (Ctrl+C)
  ✓ Real-time progress display in parallel mode

DIRECTORY STRUCTURE:
  Source files:  examples/Brand/Category/climate.yaml
  Log output:    logs/examples/Brand/Category/climate.log
"""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="For more information: https://esphome.io/",
    )

    parser.add_argument(
        "files",
        nargs="*",
        help="One or more YAML files to run",
    )

    parser.add_argument(
        "-p",
        "--pattern",
        action="append",
        help="Run all YAML files matching glob pattern (e.g., 'sensor-*.yaml')\n"
        "Can be specified multiple times",
    )

    parser.add_argument(
        "-d",
        "--dir",
        action="append",
        help="Run all YAML files in specified directory\n"
        "Can be specified multiple times",
    )

    parser.add_argument(
        "--logs",
        action="store_true",
        help="Enable log monitoring after upload\n"
        "(default runs with --no-logs for faster execution)",
    )

    parser.add_argument(
        "--exclude-file",
        default=".esphome-run-exclude",
        help="Path to exclusion file with glob patterns\n"
        "(default: .esphome-run-exclude)",
    )

    parser.add_argument(
        "-j",
        "--parallel",
        type=int,
        default=0,
        metavar="N",
        help="⚠️  EXPERIMENTAL: Run N builds in parallel (default: 0 = serial mode)\n"
        "WARNING: Parallel uploads may conflict if using same USB port.\n"
        "Recommended: use with -c/--compile-only flag",
    )

    parser.add_argument(
        "-c",
        "--compile-only",
        action="store_true",
        help="Only compile configurations, skip upload step\n"
        "Recommended for parallel mode to avoid USB port conflicts",
    )

    parser.add_argument(
        "-r",
        "--max-retries",
        type=int,
        default=3,
        metavar="N",
        help="Maximum number of retry attempts for failed builds\n"
        "(default: 3, minimum: 0)",
    )

    parser.add_argument(
        "-F",
        "--disable-failure-analysis",
        action="store_true",
        help="Disable smart failure analysis (retry all failures)\n"
        "By default, configuration errors skip retry to save time.\n"
        "Use this flag to retry all failures regardless of error type.",
    )

    parser.add_argument(
        "--disable-warmup",
        action="store_true",
        help="Skip the toolchain warmup phase.\n"
        "By default, before parallel compilation a small number of\n"
        "representative configs are compiled serially to populate the\n"
        "PlatformIO toolchain cache, avoiding concurrent-extraction\n"
        "races. Use this to turn it off.",
    )

    parser.add_argument(
        "--warmup-cache-dir",
        default=None,
        metavar="PATH",
        help="Override the warmup cache stamp directory\n"
        "(default: OS-native user cache directory)",
    )

    parser.add_argument(
        "--slow-start-interval",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Minimum gap (seconds) between parallel task starts.\n"
        "Default: 10.0. Mitigates cold-cache toolchain install races\n"
        "(pioarduino install_esptool, native ESP-IDF extraction).\n"
        "Automatically 0 when the warmup stamp shows warm caches.\n"
        "Set to 0 to disable.",
    )

    return parser.parse_args()


def collect_files(args: argparse.Namespace) -> list[str]:
    """Collect all files to run based on arguments.

    Returns Sorted list of unique file paths.
    """
    files_to_run = set(args.files)

    # Add files from patterns
    if args.pattern:
        for pattern in args.pattern:
            files_to_run.update(glob.glob(pattern))

    # Add files from directories
    if args.dir:
        for directory in args.dir:
            dir_path = Path(directory)
            files_to_run.update(str(p) for p in dir_path.glob("*.yaml"))
            files_to_run.update(str(p) for p in dir_path.glob("*.yml"))

    return sorted(list(files_to_run))


def validate_arguments(args: argparse.Namespace, files: list[str]) -> None:
    """Validate command-line arguments and collected files.

    Raises SystemExit if validation fails.
    """
    if args.parallel < 0:
        print_color(Color.RED, "Error: Parallel workers must be 0 or positive.")
        sys.exit(1)

    if not files:
        print_color(Color.RED, "Error: No YAML files specified.")
        sys.exit(1)


def create_runner_config(args: argparse.Namespace, files: list[str]) -> RunnerConfig:
    """Create RunnerConfig from command-line arguments.

    Returns RunnerConfig instance.

    Raises ConfigurationError if parameters are invalid.
    """
    # Validate max_retries
    if args.max_retries < 0:
        raise ConfigurationError(f"max_retries must be non-negative, got: {args.max_retries}")

    # Invert the logic: --no-logs is the default, --logs disables it
    use_no_logs = not args.logs

    # Invert the logic: failure analysis enabled by default, --disable-failure-analysis disables it
    enable_failure_analysis = not args.disable_failure_analysis

    # Warmup wiring: enabled by default, --disable-warmup turns it off
    warmup_enabled = not args.disable_warmup
    if args.warmup_cache_dir:
        warmup_cache_dir = Path(args.warmup_cache_dir)
    else:
        warmup_cache_dir = _default_cache_dir()
    esphome_version = get_esphome_version()
    toolchain_fingerprint = get_toolchain_fingerprint()

    slow_start_interval = (
        args.slow_start_interval if args.slow_start_interval is not None else 10.0
    )
    if slow_start_interval < 0:
        raise ConfigurationError(
            f"slow_start_interval must be non-negative, got: {slow_start_interval}"
        )

    return RunnerConfig(
        files_to_run=files,
        exclude_file=Path(args.exclude_file),
        no_logs=use_no_logs,
        parallel_workers=args.parallel,
        compile_only=args.compile_only,
        max_retries=args.max_retries,
        enable_failure_analysis=enable_failure_analysis,
        warmup_enabled=warmup_enabled,
        warmup_cache_dir=warmup_cache_dir,
        esphome_version=esphome_version,
        toolchain_fingerprint=toolchain_fingerprint,
        slow_start_interval=slow_start_interval,
    )


def main() -> None:
    """Main entry point for the CLI application.

    This function orchestrates the entire CLI workflow:
    1. Parse command-line arguments
    2. Collect files to run
    3. Validate inputs
    4. Create configuration
    5. Create and run ESPHomeRunner

    The function maintains separation of concerns by delegating
    all business logic to the ESPHomeRunner class.
    """
    args = parse_arguments()
    files = collect_files(args)
    validate_arguments(args, files)

    config = create_runner_config(args, files)
    runner = ESPHomeRunner(config)
    runner.run()


if __name__ == "__main__":
    # Ensure that KeyboardInterrupt is raised on SIGINT
    signal.signal(signal.SIGINT, signal.default_int_handler)
    main()
