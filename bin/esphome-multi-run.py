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
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, TypedDict

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# Core Data Structures, Enums, and Exceptions
# =============================================================================


class ExecutionStatus(str, Enum):
    """Status enumeration for execution results.

    Attributes:
        PENDING: Task has not started yet
        IN_PROGRESS: Task is currently running
        SUCCESS: Task completed successfully
        FAILED: Task completed with errors
        INTERRUPTED: Task was interrupted by user
        TIMEOUT: Task exceeded timeout limit
    """

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    TIMEOUT = "timeout"
    OFFLINE = "offline"


class FailureType(str, Enum):
    """Failure type enumeration for retry decision.

    Used to distinguish between permanent errors (configuration issues)
    and transient errors (network/resource issues) to optimize retry strategy.

    Attributes:
        PERMANENT: Permanent error that won't be fixed by retry
                  (e.g., YAML syntax error, file not found)
        TRANSIENT: Transient error that might be fixed by retry
                  (e.g., network timeout, resource contention)
        UNKNOWN: Unknown failure type, retry conservatively
    """

    PERMANENT = "permanent"
    TRANSIENT = "transient"
    UNKNOWN = "unknown"


class Color(str, Enum):
    """ANSI color codes for terminal output.

    Attributes:
        RED: Error messages and failures
        GREEN: Success messages
        YELLOW: Warnings and in-progress items
        BLUE: Information and headers
        RESET: Reset to default terminal color
    """

    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    PURPLE = "\033[0;35m"
    BLUE = "\033[0;36m"
    RESET = "\033[0m"


class ExecutionResult(TypedDict):
    """Type-safe structure for execution results.

    Attributes:
        status: Current execution status
        start_time: Unix timestamp when execution started (None if not started)
        end_time: Unix timestamp when execution ended (None if not ended)
        compile_time: Time taken for compilation in seconds
        upload_time: Time taken for upload in seconds
        retry_count: Number of retry attempts made
        failure_type: Type of failure for retry decision (permanent/transient/unknown)
    """

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


class ProcessTerminationError(ESPHomeRunnerError):
    """Raised when a process fails to terminate properly.

    Attributes:
        pid: Process ID that failed to terminate
        file_path: File being processed when termination failed
    """

    def __init__(self, pid: int, file_path: str):
        self.pid = pid
        self.file_path = file_path
        super().__init__(f"Failed to terminate process {pid} for {file_path}")


class FileExecutionError(ESPHomeRunnerError):
    """Raised when file execution fails.

    Attributes:
        file_path: Path to the file that failed
        exit_code: Process exit code
        original_error: Original exception if any
    """

    def __init__(
        self,
        file_path: str,
        exit_code: int,
        original_error: Exception | None = None,
    ):
        self.file_path = file_path
        self.exit_code = exit_code
        self.original_error = original_error
        super().__init__(
            f"Failed to execute {file_path} (exit code: {exit_code})"
        )


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

    Attributes:
        files_to_run: List of YAML file paths to process
        exclude_file: Path to file containing exclusion patterns
        no_logs: Whether to disable log monitoring after upload
        parallel_workers: Number of parallel workers (0 = serial mode)
        compile_only: Whether to skip upload step
        log_dir: Directory for log files
        max_retries: Maximum number of retry attempts (configurable)
        slow_start_interval: Seconds between task starts in parallel mode
        enable_failure_analysis: Enable smart failure analysis to skip retry on config errors

    Constants:
        RETRY_DELAY_SECONDS: Delay between retries
        PROCESS_TERM_TIMEOUT: Timeout for graceful process termination
        PROCESS_CLEANUP_TIMEOUT: Timeout for process cleanup
        PROCESS_WAIT_TIMEOUT: Maximum time to wait for process completion
        PROGRESS_UPDATE_INTERVAL: Update interval for progress display
        PROGRESS_BAR_LENGTH: Character length of progress bar
        MAX_FILENAME_DISPLAY: Maximum filename length in display
        MAX_PARALLEL_WORKERS_WARNING: Threshold for worker count warning
    """

    files_to_run: list[str]
    exclude_file: Path = Path(".esphome-run-exclude")
    no_logs: bool = True
    build_path: Path = Path("/data/build")
    parallel_workers: int = 0
    compile_only: bool = False
    log_dir: Path = Path("logs")
    max_retries: int = 3  # Configurable via CLI
    slow_start_interval: float = 5.0  # Default when .esphome is empty (0 = disabled)
    enable_failure_analysis: bool = True  # Enable smart failure detection (skip retry on config errors)
    force: bool = False

    # Constants
    RETRY_DELAY_SECONDS: float = 3.0  # Deprecated: use RETRY_BASE_DELAY
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

    # Slow start configuration for parallel mode (batch sizes are constant)
    SLOW_START_INITIAL_WORKERS: int = 1  # Start with 1 workers
    SLOW_START_INCREMENT: int = 1  # Add 1 workers at a time

    # Display and timing constants
    DISPLAY_THREAD_TIMEOUT: float = 2.0
    DISPLAY_INITIAL_DELAY: float = 0.2
    INTERRUPT_POLL_INTERVAL: float = 0.1
    PROCESS_POLL_INTERVAL: float = 0.1
    EXECUTOR_SHUTDOWN_DELAY: float = 0.1

    def __post_init__(self) -> None:
        """Validate configuration after initialization.

        Raises:
            ConfigurationError: If configuration is invalid
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

        Returns:
            "--no-logs" if no_logs is True, empty string otherwise
        """
        return "--no-logs" if self.no_logs else ""

    def calculate_retry_delay(self, retry_count: int) -> float:
        """Calculate retry delay using exponential backoff.

        Formula: delay = min(RETRY_BASE_DELAY * (RETRY_EXPONENTIAL_BASE ^ retry_count), RETRY_MAX_DELAY)

        Args:
            retry_count: Current retry attempt number (0-indexed)

        Returns:
            Delay in seconds, capped at RETRY_MAX_DELAY

        Examples:
            >>> config = RunnerConfig(files_to_run=["test.yaml"])
            >>> config.calculate_retry_delay(0)  # First retry
            3.0
            >>> config.calculate_retry_delay(1)  # Second retry
            6.0
            >>> config.calculate_retry_delay(2)  # Third retry
            12.0
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

    Attributes:
        completed: Number of completed executions (success + failed + interrupted)
        in_progress: Number of currently running executions
        pending: Number of pending executions
        failed: Number of failed executions
        success: Number of successful executions
        retrying: Number of executions currently retrying (in_progress with retry_count > 0)
        total: Total number of executions
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

        Returns:
            Percentage of completed executions (0-100)
        """
        return (self.completed / self.total * 100) if self.total > 0 else 0.0

    @classmethod
    def from_results(cls, results: dict[str, ExecutionResult]) -> "ExecutionStats":
        """Create stats from execution results.

        Uses Counter for efficient aggregation of status counts.

        Args:
            results: Dictionary mapping file paths to execution results

        Returns:
            ExecutionStats instance with aggregated statistics
        """
        status_counts = Counter(r["status"] for r in results.values())

        completed_statuses = {
            ExecutionStatus.SUCCESS,
            ExecutionStatus.FAILED,
            ExecutionStatus.INTERRUPTED,
            ExecutionStatus.TIMEOUT,
            ExecutionStatus.OFFLINE,
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

    Args:
        status: Execution status
        start_time: Start timestamp (None if not started)
        end_time: End timestamp (None if not ended)
        compile_time: Compilation time in seconds
        upload_time: Upload time in seconds
        retry_count: Number of retry attempts
        failure_type: Type of failure (permanent/transient/unknown)

    Returns:
        Properly typed ExecutionResult instance
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
    """Print a message in a given color.

    Args:
        color: ANSI color to use
        message: Message to print
    """
    print(f"{color.value}{message}{Color.RESET.value}")


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from a string.

    Args:
        text: String potentially containing ANSI codes

    Returns:
        String with ANSI codes removed
    """
    return RegexPatterns.ANSI_ESCAPE.sub("", text)


def append_failure_analysis_note(log_path: Path, failure_type: FailureType) -> None:
    """Append failure analysis note to log file.

    Writes a detailed explanation to the log file when a permanent failure
    is detected, helping users understand why retry was skipped.

    Args:
        log_path: Path to log file to append note
        failure_type: Type of failure detected
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

    Args:
        file_paths: List of file paths to analyze

    Returns:
        Tuple of (common_prefix_path, depth) where:
        - common_prefix_path: The common directory prefix (empty if none)
        - depth: Number of directory levels in the common prefix

    Examples:
        >>> calculate_common_prefix([
        ...     "examples/Brand/CategoryA/file1.yaml",
        ...     "examples/Brand/CategoryB/file2.yaml"
        ... ])
        ("examples/Brand", 2)

        >>> calculate_common_prefix(["file1.yaml", "file2.yaml"])
        ("", 0)

        >>> calculate_common_prefix(["examples/Brand/file.yaml"])
        ("examples/Brand", 2)
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


class FailureAnalyzer(Protocol):
    """Protocol for failure analysis strategies.

    This protocol defines the interface for analyzing execution failures
    to determine if they are permanent (configuration errors) or transient
    (network/resource issues). Follows Interface Segregation Principle.
    """

    def analyze(self, log_path: Path) -> FailureType:
        """Analyze log file to determine failure type.

        Args:
            log_path: Path to execution log file

        Returns:
            FailureType indicating if error is permanent, transient, or unknown
        """
        ...


class ESPHomeFailureAnalyzer:
    """Analyzes ESPHome execution logs to identify permanent failures.

    This implementation detects common configuration errors that won't be
    fixed by retry, such as YAML syntax errors or missing files. Follows
    Single Responsibility and Open/Closed principles.

    Attributes:
        PERMANENT_ERROR_PATTERNS: List of regex patterns for permanent errors
    """

    # Patterns for permanent errors (Open for extension)
    PERMANENT_ERROR_PATTERNS: list[str] = [
        r"Invalid YAML syntax",
        r"Failed config",
    ]

    def analyze(self, log_path: Path) -> FailureType:
        """Analyze ESPHome log file for permanent errors.

        Reads the first 100 lines of the log file (configuration errors
        typically appear early) and checks for known permanent error patterns.

        Args:
            log_path: Path to log file to analyze

        Returns:
            FailureType.PERMANENT if configuration error detected,
            FailureType.UNKNOWN otherwise (conservative retry)
        """
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as log_file:
                # Read first 100 lines (config errors appear early)
                lines = []
                for _ in range(100):
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

    Attributes:
        exclude_file: Path to the exclusion file
        patterns: List of active exclusion patterns

    Constants:
        DEFAULT_PATTERNS: Default exclusion patterns used when no exclude file exists
    """

    # Default exclusion patterns (used when no exclude file)
    DEFAULT_PATTERNS = [
        "secrets.yaml",
        "secrets.yml",
        ".*.yaml",  # Hidden YAML files
        ".*.yml",   # Hidden YML files
    ]

    def __init__(self, exclude_file: Path):
        """Initialize the file filter.

        Args:
            exclude_file: Path to file containing exclusion patterns
        """
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

        Args:
            files: List of file paths to filter

        Returns:
            Tuple of (included_files, excluded_files)
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

        Args:
            files: List of file paths to filter
            verbose: Whether to print summary information

        Returns:
            List of included files after filtering
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

    Attributes:
        results: Dictionary mapping file paths to execution results
        results_lock: Thread lock for safe concurrent access
        overall_start_time: Wall clock start time for parallel mode
        overall_end_time: Wall clock end time for parallel mode
    """

    def __init__(self) -> None:
        """Initialize the result tracker."""
        self.results: dict[str, ExecutionResult] = {}
        self.results_lock = threading.Lock()
        self.overall_start_time: float | None = None
        self.overall_end_time: float | None = None

    def initialize_results(self, files: list[str]) -> None:
        """Initialize all files with PENDING status.

        Args:
            files: List of file paths to initialize
        """
        with self.results_lock:
            for file_path in files:
                self.results[file_path] = create_execution_result(
                    status=ExecutionStatus.PENDING
                )

    def update_result(self, file_path: str, result: ExecutionResult) -> None:
        """Update result for a file (thread-safe).

        Args:
            file_path: Path to file
            result: New execution result
        """
        with self.results_lock:
            self.results[file_path] = result

    def get_result(self, file_path: str) -> ExecutionResult | None:
        """Get result for a file (thread-safe).

        Args:
            file_path: Path to file

        Returns:
            ExecutionResult if exists, None otherwise
        """
        with self.results_lock:
            return self.results.get(file_path)

    def get_stats(self) -> ExecutionStats:
        """Get current execution statistics (thread-safe).

        Returns:
            ExecutionStats with current counts
        """
        with self.results_lock:
            return ExecutionStats.from_results(self.results)

    def get_all_results(self) -> dict[str, ExecutionResult]:
        """Get deep copy of all results (thread-safe).

        Returns:
            Deep copy of results dictionary. Modifying the returned
            dictionary will not affect the internal state.
        """
        import copy

        with self.results_lock:
            return copy.deepcopy(self.results)


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

    Attributes:
        result_tracker: ResultTracker instance providing data
    """

    def __init__(self, result_tracker: ResultTracker):
        """Initialize the summary renderer.

        Args:
            result_tracker: ResultTracker instance to get data from
        """
        self.result_tracker = result_tracker

    def print_summary(
        self,
        files_to_run: list[str],
        parallel_workers: int,
        interrupted: bool = False,
    ) -> None:
        """Print final execution summary in table format.

        Args:
            files_to_run: Ordered list of files that were run
            parallel_workers: Number of parallel workers (0 = serial)
            interrupted: Whether execution was interrupted
        """
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
        """Render summary table with aligned columns and colors.

        Args:
            table_data: Table data rows
            max_compile_time: Maximum compile time (for highlighting)
            max_upload_time: Maximum upload time (for highlighting)
            max_total_duration: Maximum total duration (for highlighting)
        """
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

        Args:
            cell: Cell value
            column_idx: Column index

        Returns:
            Formatted string
        """
        if column_idx == 1:  # Status column
            if cell == "success":
                return "✓ success"
            elif cell == "failed":
                return "✗ failed"
            elif cell == "offline":
                return "⚠ offline"
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
        """Print a single table row with colors and alignment.

        Args:
            row: Row data [file, status, compile, upload, total, retries]
            col_widths: Column widths for alignment
            max_compile_time: Max compile time for highlighting
            max_upload_time: Max upload time for highlighting
            max_total_duration: Max total duration for highlighting
        """
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

        Args:
            status: Status string

        Returns:
            Colored status string
        """
        if status == "success":
            return f"{Color.GREEN.value}✓ success{Color.RESET.value}"
        elif status == "failed":
            return f"{Color.RED.value}✗ failed{Color.RESET.value}"
        elif status == "interrupted":
            return f"{Color.YELLOW.value}⚠ interrupted{Color.RESET.value}"
        elif status == "offline":
            return f"{Color.PURPLE.value}⚠ offline{Color.RESET.value}"
        else:
            return f"{Color.YELLOW.value}⚠ skipped{Color.RESET.value}"

    def _format_time(self, time_val: object, max_time: float) -> str:
        """Format time value with highlighting for max.

        Args:
            time_val: Time value (float, string, or other)
            max_time: Maximum time value to compare against

        Returns:
            Formatted time string with optional highlighting
        """
        if isinstance(time_val, float):
            if time_val == max_time and time_val > 0:
                return f"{Color.YELLOW.value}{time_val:.1f}{Color.RESET.value}"
            return f"{time_val:.1f}"
        return "-"


# =============================================================================
# Progress Display Strategies
# =============================================================================


class SerialProgressProtocol(Protocol):
    """Protocol for serial execution progress display.

    This protocol defines the interface for progress display in serial mode,
    where files are processed one at a time with live console output.
    Follows the Interface Segregation Principle by providing only the
    methods needed for serial execution.
    """

    def show_progress(
        self,
        files_to_run: list[str],
        current_file: str | None = None,
    ) -> None:
        """Display current progress.

        Args:
            files_to_run: List of all files to process
            current_file: Currently processing file (if any)
        """
        ...


class ParallelProgressProtocol(Protocol):
    """Protocol for parallel execution progress display.

    This protocol defines the interface for progress display in parallel mode,
    where multiple files are processed simultaneously with output to log files.
    Follows the Interface Segregation Principle by providing only the
    methods needed for parallel execution.
    """

    def start(self) -> None:
        """Start the progress display (e.g., background thread)."""
        ...

    def stop(self) -> None:
        """Stop the progress display and clean up resources."""
        ...

    def write_to_console(self, message: str) -> None:
        """Write a message to console in a thread-safe manner.

        Args:
            message: Message to write to console
        """
        ...


class SerialProgressDisplay:
    """Progress display for serial execution mode.

    Displays a todo list showing status of all files, with the current
    file highlighted. Uses color-coded status indicators.

    Attributes:
        result_tracker: Tracker for execution results
    """

    def __init__(self, result_tracker: ResultTracker):
        """Initialize serial progress display.

        Args:
            result_tracker: Result tracker instance
        """
        self.result_tracker = result_tracker

    def show_progress(
        self,
        files_to_run: list[str],
        current_file: str | None = None,
    ) -> None:
        """Display todo list with current status.

        Args:
            files_to_run: List of all files to process
            current_file: Currently processing file
        """
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
            elif status == ExecutionStatus.OFFLINE:
                print_color(
                    Color.PURPLE, f"⚠ [{number}] {file_path} (offline){time_str}"
                )

        print("=" * 50 + "\n")


class ParallelProgressDisplay:
    """Progress display for parallel execution mode.

    Displays a progress bar and list of currently running files.
    Updates continuously in a separate thread. Thread-safe output
    to prevent interleaved writes.

    Attributes:
        config: Runner configuration
        result_tracker: Tracker for execution results
        stdout_lock: Lock for thread-safe console output
        interrupted: Flag indicating execution was interrupted
        display_thread: Background thread for progress updates
        common_prefix: Common directory prefix for all files
        common_prefix_depth: Number of directory levels in common prefix
    """

    def __init__(self, config: RunnerConfig, result_tracker: ResultTracker, files_to_run: list[str]):
        """Initialize parallel progress display.

        Args:
            config: Runner configuration
            result_tracker: Result tracker instance
            files_to_run: List of all files to be processed
        """
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

        Args:
            stats: Current execution statistics

        Returns:
            List of lines to display
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

        Args:
            stats: Current execution statistics

        Returns:
            Progress bar string
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

        Args:
            file_path: Full file path to format
            max_length: Maximum display length

        Returns:
            Formatted path string, potentially with relative path or truncation

        Examples:
            >>> # With common prefix "examples/Brand" (depth=2)
            >>> display._format_path_for_display("examples/Brand/CategoryA/file.yaml", 60)
            "../../CategoryA/file.yaml"
            >>> display._format_path_for_display("examples/Brand/CategoryA/long-name.yaml", 30)
            "../../../long-name.yaml"
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

        Returns:
            List of formatted strings for running files
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

        Args:
            message: Message to write to stdout
        """
        with self.stdout_lock:
            sys.stdout.write(message)
            sys.stdout.flush()

    def show_progress(
        self,
        files_to_run: list[str],
        current_file: str | None = None,
    ) -> None:
        """Display current progress (not used in parallel mode).

        This method is required by the ProgressDisplay protocol but is
        not used in parallel mode, where progress is displayed via the
        background thread.

        Args:
            files_to_run: List of all files to process
            current_file: Currently processing file (if any)
        """
        pass  # Progress displayed via background thread


# =============================================================================
# Process Lifecycle Management
# =============================================================================


class ProcessOutput(Protocol):
    """Protocol for handling process output.

    This protocol allows different output handling strategies to be
    injected into the process manager.
    """

    def handle_output(self, data: bytes) -> None:
        """Handle output data from process.

        Args:
            data: Raw bytes from process stdout/stderr
        """
        ...


class ConsoleOutput:
    """Writes process output to console.

    This implementation writes output to stdout in real-time,
    suitable for serial execution mode.
    """

    def handle_output(self, data: bytes) -> None:
        """Write data to console.

        Args:
            data: Raw bytes to write to stdout
        """
        sys.stdout.write(data.decode(sys.stdout.encoding, errors="replace"))
        sys.stdout.flush()


class NullOutput:
    """Discards process output.

    This implementation is used in parallel mode where output
    is written to log files instead of console.
    """

    def handle_output(self, data: bytes) -> None:
        """Discard output data.

        Args:
            data: Raw bytes (ignored)
        """
        pass


class ProcessManager:
    """Manages process lifecycle for ESPHome executions.

    This class handles process creation, monitoring, and termination,
    abstracting the differences between pty-based (serial) and
    subprocess-based (parallel) execution.

    Attributes:
        config: Runner configuration
        current_pid: PID of current process (serial mode)
        running_processes: Map of file paths to running processes (parallel mode)
        failure_analyzer: Optional failure analyzer for smart retry decisions
    """

    def __init__(self, config: RunnerConfig):
        """Initialize the process manager.

        Creates failure analyzer if enabled in configuration (Dependency Injection).

        Args:
            config: Runner configuration
        """
        self.config = config
        self.current_pid: int | None = None
        self.running_processes: dict[str, subprocess.Popen[str]] = {}
        self.processes_lock = threading.Lock()
        self._interrupted = threading.Event()

        # Initialize failure analyzer based on configuration (DIP)
        if config.enable_failure_analysis:
            self.failure_analyzer: FailureAnalyzer | None = ESPHomeFailureAnalyzer()
        else:
            self.failure_analyzer = None

    @property
    def interrupted(self) -> bool:
        """Check if execution has been interrupted (thread-safe).

        Returns:
            True if interrupted, False otherwise
        """
        return self._interrupted.is_set()

    @interrupted.setter
    def interrupted(self, value: bool) -> None:
        """Set interrupted status (thread-safe).

        Args:
            value: True to set interrupted, False to clear
        """
        if value:
            self._interrupted.set()
        else:
            self._interrupted.clear()

    def build_command(self, file_path: str) -> list[str]:
        """Build ESPHome command for execution.

        Args:
            file_path: Path to YAML file to process

        Returns:
            Command as list of strings
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

        Args:
            file_path: Path to YAML file being processed
            retry_count: Current retry attempt number (0 for first attempt)
            execution_mode: Execution mode (e.g., "Serial", "Parallel")

        Returns:
            Formatted log header string
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
        build_path: Path,
        log_path: Path,
        retry_count: int = 0,
        interrupted: bool = False,
    ) -> ExecutionResult:
        """Execute file using pty (preserves colors).

        This method uses pty.fork() to create a pseudo-terminal, which
        preserves ANSI color codes in the output. Used for serial mode.

        Args:
            file_path: Path to YAML file to process
            log_path: Path to log file
            retry_count: Current retry attempt number
            interrupted: Whether execution was interrupted

        Returns:
            ExecutionResult with execution details
        """
        command = self.build_command(file_path)
        result = create_execution_result(
            status=ExecutionStatus.IN_PROGRESS,
            start_time=time.time(),
            retry_count=retry_count,
        )

        if build_path:
            os.environ['ESPHOME_BUILD_PATH'] = f"{build_path}"

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

        Args:
            fd: File descriptor for pty
            file_path: Path to YAML file being processed
            log_path: Path to log file
            retry_count: Current retry attempt
            result: ExecutionResult to update with timing info

        Returns:
            Process exit code
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
        build_path: Path,
        log_path: Path,
        retry_count: int = 0,
        start_time: float | None = None,
    ) -> ExecutionResult:
        """Execute file using subprocess (for parallel mode).

        This method uses subprocess.Popen for better parallel compatibility.
        Output is written to log files instead of console.

        Args:
            file_path: Path to YAML file to process
            log_path: Path to log file
            retry_count: Current retry attempt number
            start_time: Optional pre-set start time (if None, uses current time)

        Returns:
            ExecutionResult with execution details
        """
        command = self.build_command(file_path)
        result = create_execution_result(
            status=ExecutionStatus.IN_PROGRESS,
            start_time=start_time if start_time is not None else time.time(),
            retry_count=retry_count,
        )

        esphome_env = os.environ.copy()
        if build_path:
            esphome_env["ESPHOME_BUILD_PATH"] = f"{build_path}"

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
                    env=esphome_env
                )

                # Track running process (thread-safe)
                with self.processes_lock:
                    self.running_processes[file_path] = proc

                try:
                    # Wait for completion with polling for interrupt responsiveness
                    exit_code = self._wait_for_process(proc, file_path)
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

        Args:
            log_path: Path to log file

        Returns:
            Tuple of (compile_time, upload_time)
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

    def _wait_for_process(self, proc: subprocess.Popen[str], file_path: str) -> int | None:
        """Wait for process with interrupt checking.

        Uses polling to check both process completion and interrupt status.

        Args:
            proc: Process to wait for
            file_path: Path of file being processed (for tracking)

        Returns:
            Process exit code, or None if interrupted/timeout
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

        Args:
            proc: Process to terminate
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
        for file_path, proc in processes_snapshot:
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


class ExecutorProtocol(Protocol):
    """Protocol for execution strategies.

    This protocol defines the interface that all executor implementations
    must follow. Using Protocol instead of ABC provides structural
    subtyping and better type checking with mypy.
    """

    def execute(self, files: list[str]) -> None:
        """Execute all files according to strategy.

        Args:
            files: List of file paths to execute
        """
        ...


class SerialExecutor:
    """Executes files sequentially, one at a time.

    This executor runs files in serial mode, displaying output to console
    in real-time. It uses pty for process execution to preserve ANSI colors.

    Attributes:
        config: Runner configuration
        process_manager: Process lifecycle manager
        result_tracker: Result tracker
        progress_display: Progress display strategy
        interrupted: Flag indicating execution was interrupted
    """

    def __init__(
        self,
        config: RunnerConfig,
        process_manager: ProcessManager,
        result_tracker: ResultTracker,
        progress_display: SerialProgressProtocol,
    ):
        """Initialize serial executor.

        Args:
            config: Runner configuration
            process_manager: Process manager instance
            result_tracker: Result tracker instance
            progress_display: Progress display instance
        """
        self.config = config
        self.process_manager = process_manager
        self.result_tracker = result_tracker
        self.progress_display = progress_display
        self.interrupted = False

    def execute(self, files: list[str]) -> None:
        """Execute files sequentially.

        Args:
            files: List of file paths to execute
        """
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
        """Execute a single file with retry logic.

        Args:
            file_path: Path to file to execute
        """
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
                # Permanent error detected - skip retry
                # Append analysis note to log file
                log_path = self.config.log_dir / Path(file_path).with_suffix('.log')
                append_failure_analysis_note(log_path, FailureType.PERMANENT)

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

        Args:
            file_path: Path to file to execute
            retry_count: Current retry attempt

        Returns:
            ExecutionResult with execution details
        """
        file = Path(file_path)
        basename = Path(file_path).stem
        with open(file, "r") as f_in:
            for line in f_in.readline():
                if "devicename:" in line:
                    basename = line.split(":")[1].rstrip().strip()
                    break

        build_path = self.config.build_path
        # Preserve directory structure in logs
        log_path = self.config.log_dir / file.with_suffix('.log')
        # Ensure parent directory exists
        log_path.parent.mkdir(parents=True, exist_ok=True)

        param = '-n' if sys.platform.lower() == 'win32' else '-c'

        if not self.config.force:
            command = ['ping', param, '1', f"{basename}.local"]
            response = subprocess.call(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT)
            if response != 0:
                return create_execution_result(
                    status=ExecutionStatus.OFFLINE
                )

        print("=" * 50)
        print_color(Color.BLUE, f"Running: {file_path}")
        if retry_count > 0:
            print_color(Color.YELLOW, f"(Retry attempt {retry_count})")
        print("=" * 50)

        result = self.process_manager.run_with_pty(
            file_path, build_path, log_path, retry_count, self.interrupted
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


class ParallelExecutor:
    """Executes files in parallel using multiple workers.

    This executor runs multiple files simultaneously using ThreadPoolExecutor.
    Output is written to log files instead of console. Progress is displayed
    via a background thread.

    Attributes:
        config: Runner configuration
        process_manager: Process lifecycle manager
        result_tracker: Result tracker
        progress_display: Progress display strategy (ParallelProgressDisplay)
        interrupted: Flag indicating execution was interrupted
    """

    def __init__(
        self,
        config: RunnerConfig,
        process_manager: ProcessManager,
        result_tracker: ResultTracker,
        progress_display: ParallelProgressProtocol,
    ):
        """Initialize parallel executor.

        Args:
            config: Runner configuration
            process_manager: Process manager instance
            result_tracker: Result tracker instance
            progress_display: Progress display instance (ParallelProgressProtocol)
        """
        self.config = config
        self.process_manager = process_manager
        self.result_tracker = result_tracker
        self.progress_display = progress_display
        self.interrupted = False

        # Slow start tracking for execution (not just submission)
        self.last_task_start_time: float | None = None
        self.start_time_lock = threading.Lock()

    def execute(self, files: list[str]) -> None:
        """Execute files in parallel with slow start mechanism.

        Implements gradual worker ramp-up to avoid initial resource contention.
        This helps prevent failures caused by multiple workers competing for
        shared resources like PlatformIO package downloads or build cache.

        Args:
            files: List of file paths to execute
        """
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

            # Submit tasks with slow start if enabled
            if self.config.slow_start_interval > 0 and len(files) > self.config.SLOW_START_INITIAL_WORKERS:
                futures = self._execute_with_slow_start(executor, files)
            else:
                # Submit all tasks immediately (original behavior)
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

    def _execute_with_slow_start(
        self,
        executor: Any,  # ThreadPoolExecutor
        files: list[str]
    ) -> dict[Any, str]:  # dict[Future, str]
        """Submit tasks with gradual ramp-up (slow start).

        Instead of submitting all tasks at once, this method gradually submits
        tasks in batches. The ThreadPoolExecutor will automatically manage the
        actual concurrency based on max_workers. This approach helps avoid
        resource contention during initial compilation (e.g., PlatformIO package
        downloads, compiler initialization).

        Strategy:
        1. Submit initial batch of tasks
        2. Wait SLOW_START_INTERVAL seconds
        3. Submit next batch of tasks
        4. Repeat until all tasks are submitted

        Args:
            executor: ThreadPoolExecutor instance
            files: List of file paths to execute

        Returns:
            Dictionary mapping Future objects to file paths
        """
        from concurrent.futures import Future

        futures: dict[Future[None], str] = {}
        files_to_submit = list(files)  # Make a copy to work with

        # Calculate initial batch size
        initial_batch_size = min(
            self.config.SLOW_START_INITIAL_WORKERS,
            len(files)
        )

        # Submit initial batch
        for _ in range(initial_batch_size):
            if files_to_submit and not self.interrupted:
                file_path = files_to_submit.pop(0)
                future = executor.submit(self._execute_file_with_retry, file_path)
                futures[future] = file_path

        # Gradually submit remaining tasks in batches
        while files_to_submit and not self.interrupted:
            # Wait before submitting next batch
            time.sleep(self.config.slow_start_interval)

            # Calculate next batch size
            next_batch_size = min(
                self.config.SLOW_START_INCREMENT,
                len(files_to_submit)
            )

            # Submit next batch
            for _ in range(next_batch_size):
                if files_to_submit and not self.interrupted:
                    file_path = files_to_submit.pop(0)
                    future = executor.submit(self._execute_file_with_retry, file_path)
                    futures[future] = file_path

        return futures

    def _wait_for_slow_start(self) -> None:
        """Wait to enforce slow start interval between task executions.

        This method ensures that tasks don't start simultaneously, even when
        workers become available. It enforces a minimum interval between any
        two task starts, preventing resource contention throughout the entire
        execution (not just at initial submission).

        The wait is interruptible to maintain responsiveness to Ctrl+C.

        Thread-safe: Uses lock to coordinate across all worker threads.
        Other threads will wait at the lock until the current thread finishes
        waiting and updates the last_task_start_time.
        """
        if self.config.slow_start_interval <= 0:
            return

        with self.start_time_lock:
            # Check if we need to wait
            if self.last_task_start_time is not None:
                elapsed = time.time() - self.last_task_start_time
                if elapsed < self.config.slow_start_interval:
                    wait_time = self.config.slow_start_interval - elapsed
                    # Sleep while holding the lock - this blocks other threads
                    # from starting tasks, which is exactly what we want
                    self._interruptible_sleep(wait_time)

            # Update last start time
            self.last_task_start_time = time.time()

    def _execute_file_with_retry(self, file_path: str) -> None:
        """Execute a single file with retry logic.

        Args:
            file_path: Path to file to execute
        """
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
                # Enforce slow start interval to prevent simultaneous task starts
                self._wait_for_slow_start()
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
                # Permanent error detected - don't retry
                # Append analysis note to log file
                log_path = self.config.log_dir / Path(file_path).with_suffix('.log')
                append_failure_analysis_note(log_path, FailureType.PERMANENT)

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

        Args:
            file_path: Path to file to execute
            retry_count: Current retry attempt
            start_time: Pre-set start time (from after slow start wait)

        Returns:
            ExecutionResult with execution details
        """
        file = Path(file_path)
        basename = Path(file_path).stem
        with open(file, "") as f_in:
            for line in f_in.readline():
                if "devicename:" in line:
                    basename = line.split(":")[1].rstrip().strip()
                    break

        # Preserve directory structure in logs
        log_path = self.config.log_dir / file.with_suffix('.log')
        build_path = self.config.build_path
        # Ensure parent directory exists
        log_path.parent.mkdir(parents=True, exist_ok=True)

        param = '-n' if sys.platform.lower() == 'win32' else '-c'

        if not self.config.force:
            command = ['ping', param, '1', f"{basename}.local"]
            response = subprocess.call(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT)
            if response != 0:
                return create_execution_result(
                    status=ExecutionStatus.OFFLINE
                )

        result = self.process_manager.run_with_subprocess(
            file_path, build_path, log_path, retry_count, start_time
        )

        return result

    def _interruptible_sleep(self, duration: float) -> None:
        """Sleep in short intervals to allow interrupt checking.

        Args:
            duration: Total sleep duration in seconds
        """
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

        # Terminate all running processes
        try:
            self.process_manager.terminate_all_processes()
        except Exception:
            pass  # Ignore errors during interrupt


# =============================================================================
# Executor Factory
# =============================================================================


class ExecutorFactory:
    """Factory for creating execution strategies.

    This class implements the Factory Pattern to create appropriate executor
    instances based on configuration. It encapsulates the creation logic and
    dependencies, improving testability and adhering to the Dependency
    Inversion Principle.

    The factory handles:
    - Strategy selection (serial vs parallel)
    - Progress display creation
    - Dependency injection for executors
    """

    @staticmethod
    def create(
        config: RunnerConfig,
        process_manager: ProcessManager,
        result_tracker: ResultTracker,
    ) -> ExecutorProtocol:
        """Create appropriate executor based on configuration.

        Args:
            config: Runner configuration
            process_manager: Process lifecycle manager
            result_tracker: Result tracker

        Returns:
            ExecutorProtocol: Serial or parallel executor instance

        Examples:
            >>> factory = ExecutorFactory()
            >>> executor = factory.create(config, process_mgr, tracker)
        """
        if config.parallel_workers > 0:
            # Create parallel executor with parallel progress display
            progress_display = ParallelProgressDisplay(
                config=config,
                result_tracker=result_tracker,
                files_to_run=config.files_to_run,
            )
            return ParallelExecutor(
                config=config,
                process_manager=process_manager,
                result_tracker=result_tracker,
                progress_display=progress_display,
            )
        else:
            # Create serial executor with serial progress display
            progress_display = SerialProgressDisplay(result_tracker=result_tracker)
            return SerialExecutor(
                config=config,
                process_manager=process_manager,
                result_tracker=result_tracker,
                progress_display=progress_display,
            )


# =============================================================================
# Main Runner Coordinator
# =============================================================================


class ESPHomeRunner:
    """Main coordinator for ESPHome multi-run execution.

    This class follows the Single Responsibility Principle by delegating
    specific responsibilities to specialized components. It coordinates the
    overall execution flow by composing these components together.

    Attributes:
        config: Runner configuration
        file_filter: File filtering component
        process_manager: Process lifecycle manager
        result_tracker: Result tracking component
        executor: Execution strategy (serial or parallel)
        files_to_run: Filtered list of files to execute
    """

    def __init__(self, config: RunnerConfig):
        """Initialize the runner with dependency injection.

        Creates all necessary components and wires them together using
        composition. This follows the Dependency Inversion Principle by
        depending on abstractions (protocols) rather than concrete classes.

        Args:
            config: Runner configuration

        Raises:
            ConfigurationError: If log directory cannot be created
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
        self.executor = ExecutorFactory.create(
            config=config,
            process_manager=self.process_manager,
            result_tracker=self.result_tracker,
        )

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

        # Step 3: Execute files with timing
        self.result_tracker.overall_start_time = time.time()
        try:
            self.executor.execute(self.files_to_run)
        finally:
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
            # Build parallel mode message with slow start info
            mode_msg = f"[Parallel Mode - {self.config.parallel_workers} workers"

            if self.config.slow_start_interval > 0:
                mode_msg += f", Slow Start: {self.config.slow_start_interval:.1f}s interval"
            else:
                mode_msg += ", No Slow Start"

            mode_msg += "]"
            print_color(Color.BLUE, mode_msg)

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

        print(f"Starting at: {datetime.now()}")


# =============================================================================
# CLI Argument Parsing and Entry Point
# =============================================================================


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace
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
  ✓ Slow Start: Enforces minimum interval between task starts
    - Dynamic default: 0s if any .esphome has cache, 5s if all .esphome are empty
    - Prevents multiple tasks from starting simultaneously on first run
    - Reduces resource contention during package downloads
    - Set --slow-start-interval 0 to force disable
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
  ✓ Slow start mechanism for parallel mode (configurable with --slow-start-interval)
  ✓ Automatic .esphome cache detection (checks each YAML file's directory)
  ✓ Default exclusion patterns (auto-excludes secrets.yaml when no exclude file)
  ✓ Preserves directory structure in logs/ (mirrors your source structure)
  ✓ Color-coded output and progress tracking
  ✓ Detailed execution summary with timing statistics
  ✓ Graceful interrupt handling (Ctrl+C)
  ✓ Real-time progress display in parallel mode

DIRECTORY STRUCTURE:
  Source files:  examples/Brand/Category/climate.yaml
  Cache check:   examples/Brand/Category/.esphome/
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
        "-b",
        "--build_path",
        help="Set the specified directory to ESPHOME_BUILD_PATH\n"
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
        "--slow-start-interval",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Seconds between task starts in parallel mode\n"
        "(default: 0 if any .esphome has cache, 5.0 if all .esphome are empty)",
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
        "-f",
        "--force",
        action="store_true",
        help="Skip check the device Online or Not\n"
        "If the device is not online, still compile or run\n",
    )

    return parser.parse_args()


def collect_files(args: argparse.Namespace) -> list[str]:
    """Collect all files to run based on arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        Sorted list of unique file paths
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

    Args:
        args: Parsed command-line arguments
        files: Collected files to run

    Raises:
        SystemExit: If validation fails
    """
    if args.parallel < 0:
        print_color(Color.RED, "Error: Parallel workers must be 0 or positive.")
        sys.exit(1)

    if not files:
        print_color(Color.RED, "Error: No YAML files specified.")
        sys.exit(1)


def is_esphome_cache_empty(build_path, files_to_run: list[str]) -> bool:
    """Check if .esphome directories are empty for the given files.

    Uses aggressive strategy: returns False if ANY .esphome directory
    has cache content, disabling slow start for faster execution.

    This function checks each YAML file's directory for a .esphome cache
    directory. If any file's directory has cache, slow start is disabled
    to allow immediate parallel execution.

    Args:
        files_to_run: List of YAML files to check

    Returns:
        True if all .esphome directories are empty/missing (use slow start)
        False if any .esphome directory has content (skip slow start)

    Examples:
        >>> is_esphome_cache_empty(examples/Brand/Category/.esphome, ["examples/Brand/Category/climate.yaml"])
        # Checks examples/Brand/Category/.esphome/
    """
    if not files_to_run:
        return True  # No files, use slow start (safe default)

    if build_path:
        esphome_dir = Path(build_path).parent
        # Has .esphome directory?
        if esphome_dir.exists():
            try:
                # Has any cache content -> skip slow start
                if any(esphome_dir.iterdir()):
                    return False
            except (OSError, PermissionError):
                # Can't read -> skip this directory, continue checking others
                pass
    else:
        # Get unique directories containing the YAML files
        yaml_dirs = set()
        for file_path in files_to_run:
            yaml_dir = Path(file_path).parent
            yaml_dirs.add(yaml_dir)

        # Check each directory's .esphome (aggressive strategy)
        for yaml_dir in yaml_dirs:
            esphome_dir = yaml_dir / ".esphome"

            # Has .esphome directory?
            if esphome_dir.exists():
                try:
                    # Has any cache content -> skip slow start
                    if any(esphome_dir.iterdir()):
                        return False
                except (OSError, PermissionError):
                    # Can't read -> skip this directory, continue checking others
                    continue

    # All .esphome directories are empty/missing -> use slow start
    return True


def create_runner_config(args: argparse.Namespace, files: list[str]) -> RunnerConfig:
    """Create RunnerConfig from command-line arguments.

    Args:
        args: Parsed command-line arguments
        files: Collected files to run

    Returns:
        RunnerConfig instance

    Raises:
        ConfigurationError: If parameters are invalid
    """
    # Validate max_retries
    if args.max_retries < 0:
        raise ConfigurationError(f"max_retries must be non-negative, got: {args.max_retries}")

    # Determine slow_start_interval with dynamic default
    if args.slow_start_interval is None:
        # Dynamic default: 0 if any .esphome has cache, 5.0 if all are empty
        slow_start_interval = 5.0 if is_esphome_cache_empty(args.build_path, files) else 0.0
    else:
        slow_start_interval = args.slow_start_interval
        # Validate slow_start_interval
        if slow_start_interval < 0:
            raise ConfigurationError(f"slow_start_interval must be non-negative, got: {slow_start_interval}")

    # Invert the logic: --no-logs is the default, --logs disables it
    use_no_logs = not args.logs

    # Invert the logic: failure analysis enabled by default, --disable-failure-analysis disables it
    enable_failure_analysis = not args.disable_failure_analysis

    return RunnerConfig(
        files_to_run=files,
        exclude_file=Path(args.exclude_file),
        no_logs=use_no_logs,
        build_path=args.build_path,
        parallel_workers=args.parallel,
        compile_only=args.compile_only,
        max_retries=args.max_retries,
        slow_start_interval=slow_start_interval,
        enable_failure_analysis=enable_failure_analysis,
        force=args.force
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
