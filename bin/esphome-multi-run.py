#!/usr/bin/env python3
import argparse
import glob
import os
import pty
import re
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Dict, Optional, Tuple

# --- ANSI Color Codes ---
RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
BLUE = "\033[0;36m"
NC = "\033[0m"


def print_color(color: str, message: str):
    """Prints a message in a given color."""
    print(f"{color}{message}{NC}")


class ESPHomeRunner:
    """
    Manages the execution and tracking of multiple ESPHome YAML files.
    """

    def __init__(self, files_to_run: List[str], exclude_file: str, no_logs: bool, parallel_workers: int = 0, compile_only: bool = False):
        self.all_files = files_to_run
        self.exclude_file = exclude_file
        self.no_logs_arg = "--no-logs" if no_logs else ""
        self.parallel_workers = parallel_workers
        self.compile_only = compile_only
        self.files_to_run: List[str] = []
        self.current_process_pid: Optional[int] = None
        self.interrupted = False
        self.running_processes: Dict[str, subprocess.Popen] = {}  # For parallel mode
        self.results_lock = threading.Lock()  # Thread safety for results
        self.progress_lock = threading.Lock()  # Thread safety for progress display
        self.overall_start_time = None  # Track actual wall clock time
        self.overall_end_time = None

        # --- Result Tracking ---
        self.results: Dict[str, Dict] = {}  # file -> {status, start_time, end_time}
        self.log_dir = "logs"
        os.makedirs(self.log_dir, exist_ok=True)

    def _filter_excluded_files(self):
        """Filters files based on the exclusion file."""
        patterns = []
        if os.path.exists(self.exclude_file):
            print(f"Active exclusion patterns from {self.exclude_file}:")
            with open(self.exclude_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        patterns.append(line)
                        print(f"  - {line}")

        excluded_count = 0
        for file in self.all_files:
            basename = os.path.basename(file)
            is_excluded = False
            for pattern in patterns:
                if glob.fnmatch.fnmatch(basename, pattern):
                    is_excluded = True
                    break
            if is_excluded:
                excluded_count += 1
                print_color(YELLOW, f"Excluding: {file} (matched exclusion pattern)")
            else:
                self.files_to_run.append(file)

        print(f"\nFiles to process: {len(self.files_to_run)}")
        if excluded_count > 0:
            print(f"Files excluded: {excluded_count}")

    def run(self):
        """The main execution loop."""
        self._filter_excluded_files()
        if not self.files_to_run:
            print_color(RED, "Error: All files were excluded. No files to process.")
            return

        print_color(BLUE, "ESPHome Multi-Run Script")
        if self.parallel_workers > 0:
            print_color(BLUE, f"[Parallel Mode - {self.parallel_workers} workers]")
        if self.compile_only:
            print_color(YELLOW, "[Compile-only mode - no uploads]")
        print(f"Starting at: {datetime.now()}")

        # Track actual start/end time for parallel execution
        self.overall_start_time = time.time()

        if self.parallel_workers > 0:
            self._run_parallel()
        else:
            self._run_serial()

        self.overall_end_time = time.time()

        print_color(BLUE, "\nExecution finished. Generating summary...")
        self._print_summary()

    def _run_serial(self):
        """Serial execution mode (original behavior)."""
        try:
            for file_path in self.files_to_run:
                if self.interrupted:
                    print_color(YELLOW, "Halting further execution due to interrupt.")
                    break
                self._print_todo_list(current_file=file_path)

                # Retry logic for serial mode
                max_retries = 2
                retry_count = 0
                success = False

                while retry_count <= max_retries and not success:
                    if retry_count > 0:
                        print_color(YELLOW, f"\n=== RETRY {retry_count} for {file_path} ===\n")
                        time.sleep(3)  # 3 second delay before retry

                    self._run_single_file(file_path, retry_count)

                    # Check if successful
                    if self.results[file_path]["status"] == "success":
                        success = True
                    elif retry_count < max_retries and not self.interrupted:
                        retry_count += 1
                    else:
                        break

        except KeyboardInterrupt:
            self.interrupted = True
            print_color(YELLOW, "\nInterrupt signal received! Stopping...")
            if self.current_process_pid:
                print_color(
                    YELLOW,
                    f"Terminating current task (PID: {self.current_process_pid})...",
                )
                try:
                    # Send SIGTERM for graceful shutdown
                    os.kill(self.current_process_pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass  # Process already finished

    def _run_parallel(self):
        """Parallel execution mode with progress display."""
        # Initialize all files as pending
        for file_path in self.files_to_run:
            self.results[file_path] = {
                "status": "pending",
                "start_time": None,
                "end_time": None,
                "compile_time": 0.0,
                "upload_time": 0.0,
                "retry_count": 0,
            }

        try:
            with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
                # Submit all tasks
                futures = {
                    executor.submit(self._run_single_file_parallel, file_path): file_path
                    for file_path in self.files_to_run
                }

                # Start progress display thread
                progress_thread = threading.Thread(target=self._display_parallel_progress)
                progress_thread.daemon = True
                progress_thread.start()

                # Wait for completion
                for future in as_completed(futures):
                    if self.interrupted:
                        executor.shutdown(wait=False)
                        break
                    try:
                        future.result()
                    except Exception as e:
                        file_path = futures[future]
                        print_color(RED, f"\nError processing {file_path}: {e}")

        except KeyboardInterrupt:
            self.interrupted = True
            print("\n")  # Move to new line after progress display
            print_color(YELLOW, "Interrupt signal received! Stopping all workers...")
            # Terminate all running processes
            for file_path, proc in list(self.running_processes.items()):
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except:
                    try:
                        proc.kill()
                    except:
                        pass
        finally:
            # Clean up any remaining processes
            for file_path, proc in list(self.running_processes.items()):
                try:
                    if proc.poll() is None:
                        proc.terminate()
                        proc.wait(timeout=2)
                except:
                    pass

    def _display_parallel_progress(self):
        """Display progress for parallel mode."""
        last_display_lines = 0

        while not self.interrupted:
            with self.results_lock:
                completed = sum(1 for r in self.results.values() if r["status"] in ["success", "failed"])
                in_progress = sum(1 for r in self.results.values() if r["status"] == "in_progress")
                pending = sum(1 for r in self.results.values() if r["status"] == "pending")
                failed = sum(1 for r in self.results.values() if r["status"] == "failed")
                success = sum(1 for r in self.results.values() if r["status"] == "success")
                total = len(self.files_to_run)

            if completed == total:
                break

            # Clear previous display
            if last_display_lines > 0:
                sys.stdout.write(f"\033[{last_display_lines}A")  # Move cursor up
                sys.stdout.write("\033[J")  # Clear from cursor to end

            lines = []

            # Progress bar
            progress_pct = (completed / total) * 100 if total > 0 else 0
            bar_length = 40
            filled = int(bar_length * completed / total) if total > 0 else 0
            bar = "█" * filled + "░" * (bar_length - filled)

            lines.append(f"\nProgress: [{bar}] {completed}/{total} ({progress_pct:.0f}%)")
            lines.append("")

            # Currently running
            if in_progress > 0:
                lines.append("Currently Running:")
                worker_num = 1
                with self.results_lock:
                    for file_path, result in self.results.items():
                        if result["status"] == "in_progress":
                            elapsed = time.time() - result["start_time"] if result["start_time"] else 0
                            basename = os.path.basename(file_path)
                            retry_str = f" (retry {result.get('retry_count', 0)})" if result.get('retry_count', 0) > 0 else ""
                            lines.append(f"  Worker {worker_num}: {basename:<30} [{elapsed:>6.0f}s]{retry_str}")
                            worker_num += 1
                            if worker_num > self.parallel_workers:
                                break

            lines.append("")
            lines.append(f"Completed: {success} ✓ | Failed: {failed} ✗ | Remaining: {pending}")

            # Print all lines
            output = "\n".join(lines)
            sys.stdout.write(output)
            sys.stdout.flush()
            last_display_lines = len(lines)

            time.sleep(0.5)

        # Final clear
        if last_display_lines > 0:
            sys.stdout.write(f"\033[{last_display_lines}A")
            sys.stdout.write("\033[J")

    def _run_single_file_parallel(self, file_path: str) -> Tuple[str, bool]:
        """Execute a single file in parallel mode (no console output)."""
        basename = os.path.splitext(os.path.basename(file_path))[0]
        log_path = os.path.join(self.log_dir, f"{basename}.log")

        with self.results_lock:
            self.results[file_path] = {
                "status": "in_progress",
                "start_time": time.time(),
                "end_time": None,
                "compile_time": 0.0,
                "upload_time": 0.0,
                "retry_count": 0,
            }

        # Use compile command for parallel mode to avoid upload conflicts
        if self.compile_only:
            command = ["esphome", "compile", file_path]
        else:
            command = ["esphome", "run", file_path]
            if self.no_logs_arg:
                command.append(self.no_logs_arg)

        max_retries = 2  # Retry failed builds once
        retry_count = 0

        while retry_count <= max_retries:
            try:
                # Use subprocess for better parallel compatibility
                with open(log_path, "a" if retry_count > 0 else "w") as log_file:
                    if retry_count > 0:
                        log_file.write(f"\n\n=== RETRY {retry_count} ===\n\n")

                    proc = subprocess.Popen(
                        command,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1
                    )

                    # Track running process
                    with self.results_lock:
                        self.running_processes[file_path] = proc

                    # Wait for completion
                    exit_code = proc.wait()

                    # Remove from running processes
                    with self.results_lock:
                        if file_path in self.running_processes:
                            del self.running_processes[file_path]

                # Parse log for timing info
                compile_time = 0.0
                upload_time = 0.0
                with open(log_path, "r") as log_file:
                    for line in log_file:
                        compile_match = re.search(r"Took (\d+\.\d+) seconds", line)
                        if compile_match:
                            compile_time = float(compile_match.group(1))
                        upload_match = re.search(r"Upload took (\d+\.\d+) seconds", line)
                        if upload_match:
                            upload_time = float(upload_match.group(1))

                with self.results_lock:
                    self.results[file_path]["retry_count"] = retry_count

                    if self.interrupted:
                        self.results[file_path]["status"] = "interrupted"
                        self.results[file_path]["end_time"] = time.time()
                        self.results[file_path]["compile_time"] = compile_time
                        self.results[file_path]["upload_time"] = upload_time
                        return file_path, False
                    elif exit_code == 0:
                        self.results[file_path]["status"] = "success"
                        self.results[file_path]["end_time"] = time.time()
                        self.results[file_path]["compile_time"] = compile_time
                        self.results[file_path]["upload_time"] = upload_time
                        return file_path, True
                    else:
                        # If failed and we haven't exhausted retries, try again
                        if retry_count < max_retries:
                            retry_count += 1
                            time.sleep(3)  # 3 second delay before retry
                            continue
                        else:
                            self.results[file_path]["status"] = "failed"
                            self.results[file_path]["end_time"] = time.time()
                            self.results[file_path]["compile_time"] = compile_time
                            self.results[file_path]["upload_time"] = upload_time
                            return file_path, False

            except Exception:
                with self.results_lock:
                    if retry_count < max_retries:
                        retry_count += 1
                        time.sleep(3)  # 3 second delay before retry
                        continue
                    else:
                        self.results[file_path]["status"] = "failed"
                        self.results[file_path]["end_time"] = time.time()
                        if file_path in self.running_processes:
                            del self.running_processes[file_path]
                        return file_path, False

        # This should not be reached, but just in case
        with self.results_lock:
            if file_path not in self.results or self.results[file_path].get("end_time") is None:
                self.results[file_path]["status"] = "failed"
                self.results[file_path]["end_time"] = time.time()
        return file_path, False

    def _run_single_file(self, file_path: str, retry_count: int = 0):
        """Executes a single ESPHome YAML file within a pseudo-terminal to preserve color."""
        basename = os.path.splitext(os.path.basename(file_path))[0]
        log_path = os.path.join(self.log_dir, f"{basename}.log")

        print("=" * 50)
        print_color(BLUE, f"Running: {file_path}")
        if retry_count > 0:
            print_color(YELLOW, f"(Retry attempt {retry_count})")
        print("=" * 50)

        # Initialize or update results
        if file_path not in self.results:
            self.results[file_path] = {
                "status": "in_progress",
                "start_time": time.time(),
                "end_time": None,
                "compile_time": 0.0,
                "upload_time": 0.0,
                "retry_count": retry_count,
            }
        else:
            self.results[file_path]["status"] = "in_progress"
            self.results[file_path]["retry_count"] = retry_count
            if retry_count == 0:
                self.results[file_path]["start_time"] = time.time()

        # Use compile command for parallel mode to avoid upload conflicts
        if self.compile_only:
            command = ["esphome", "compile", file_path]
        else:
            command = ["esphome", "run", file_path]
            if self.no_logs_arg:
                command.append(self.no_logs_arg)

        try:
            # Use pty.fork() to create a pseudo-terminal
            pid, fd = pty.fork()

            if pid == 0:  # Child process
                try:
                    os.execvp(command[0], command)
                except FileNotFoundError:
                    print(f"Error: command not found: {command[0]}")
                    os._exit(127)

            else:  # Parent process
                self.current_process_pid = pid
                exit_code = 1  # Default to failure

                # Timing and parsing variables
                compile_time = 0.0
                upload_time = 0.0
                line_buffer = b""

                try:
                    # Append to log if retry, otherwise overwrite
                    log_mode = "ab" if retry_count > 0 else "wb"
                    with open(log_path, log_mode) as log_file:
                        if retry_count > 0:
                            log_file.write(f"\n\n=== RETRY {retry_count} ===\n\n".encode())

                        while True:
                            try:
                                data = os.read(fd, 1024)
                            except OSError:
                                break
                            if not data:
                                break

                            log_file.write(data)
                            sys.stdout.write(
                                data.decode(sys.stdout.encoding, errors="replace")
                            )
                            sys.stdout.flush()

                            # Process lines for timing info
                            line_buffer += data
                            while b"\n" in line_buffer:
                                line_bytes, line_buffer = line_buffer.split(b"\n", 1)
                                line_str = line_bytes.decode(
                                    "utf-8", errors="replace"
                                ).strip()

                                compile_match = re.search(
                                    r"Took (\d+\.\d+) seconds", line_str
                                )
                                if compile_match:
                                    compile_time = float(compile_match.group(1))

                                upload_match = re.search(
                                    r"Upload took (\d+\.\d+) seconds", line_str
                                )
                                if upload_match:
                                    upload_time = float(upload_match.group(1))

                    _, exit_status = os.waitpid(pid, 0)

                    if os.WIFEXITED(exit_status):
                        exit_code = os.WEXITSTATUS(exit_status)

                finally:
                    self.current_process_pid = None
                    self.results[file_path]["compile_time"] = compile_time
                    self.results[file_path]["upload_time"] = upload_time

            if self.interrupted and self.results[file_path]["status"] == "in_progress":
                self.results[file_path]["status"] = "interrupted"
            elif exit_code == 0:
                self.results[file_path]["status"] = "success"
                print_color(GREEN, f"\n✓ Success: {file_path}")
            else:
                self.results[file_path]["status"] = "failed"
                print_color(RED, f"\n✗ Failed: {file_path} (Exit Code: {exit_code})")

        except Exception as e:
            self.results[file_path]["status"] = "failed"
            print_color(
                RED, f"\n✗ An unexpected error occurred while running {file_path}: {e}"
            )
        finally:
            if self.results[file_path].get("end_time") is None:
                self.results[file_path]["end_time"] = time.time()
            self.current_process_pid = None

    def _print_todo_list(self, current_file: Optional[str] = None):
        """Displays the list of files and their current status."""
        print("\n" + "=" * 50)
        print_color(BLUE, "EXECUTION TODO LIST")
        print("=" * 50)
        for i, file_path in enumerate(self.files_to_run):
            number = i + 1
            result = self.results.get(file_path)
            status = result["status"] if result else "pending"

            time_str = ""
            if result and result.get("end_time"):
                total_d = result["end_time"] - result["start_time"]
                time_str = f" [{total_d:.1f}s]"

            if status == "in_progress" or (
                status == "pending" and file_path == current_file
            ):
                print_color(YELLOW, f"→ [{number}] {file_path} (IN PROGRESS...)")
            elif status == "pending":
                print(f"  [{number}] {file_path} (pending)")
            elif status == "success":
                print_color(GREEN, f"✓ [{number}] {file_path} (success){time_str}")
            elif status == "failed":
                print_color(RED, f"✗ [{number}] {file_path} (failed){time_str}")
            elif status == "interrupted":
                print_color(YELLOW, f"⚠ [{number}] {file_path} (interrupted){time_str}")
        print("=" * 50 + "\n")

    def _print_summary(self):
        """Prints the final execution summary in a table format."""
        summary_title = "EXECUTION SUMMARY"
        if self.interrupted:
            summary_title += " (INTERRUPTED)"

        print_color(BLUE, summary_title)

        # --- Data Preparation ---
        table_data = []
        total_time = 0
        max_compile_time = 0.0
        max_upload_time = 0.0
        max_total_duration = 0.0

        for file_path in self.files_to_run:
            result = self.results.get(file_path)
            if not result:
                table_data.append([file_path, "skipped", "-", "-", "-", 0])
                continue

            status = result["status"]
            compile_t = result.get("compile_time", 0.0)
            upload_t = result.get("upload_time", 0.0)
            retry_cnt = result.get("retry_count", 0)

            if result.get("end_time"):
                total_d = result["end_time"] - result["start_time"]
                total_time += total_d
            else:
                total_d = 0.0

            table_data.append([file_path, status, compile_t, upload_t, total_d, retry_cnt])

            if compile_t > max_compile_time:
                max_compile_time = compile_t
            if upload_t > max_upload_time:
                max_upload_time = upload_t
            if total_d > max_total_duration:
                max_total_duration = total_d

        # --- Table Rendering ---
        headers = ["File", "Status", "Compile (s)", "Upload (s)", "Total (s)", "Retries"]

        def strip_ansi(text: str) -> str:
            """Removes ANSI escape codes from a string."""
            ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
            return ansi_escape.sub("", text)

        # Column width calculation
        col_widths = [len(h) for h in headers]
        for row in table_data:
            for i, cell in enumerate(row):
                cell_str = str(cell)
                if i == 1:  # Status column
                    if cell == "success":
                        cell_str = f"✓ success"
                    elif cell == "failed":
                        cell_str = f"✗ failed"
                    elif cell == "interrupted":
                        cell_str = f"⚠ interrupted"
                    else:
                        cell_str = f"⚠ skipped"
                elif isinstance(cell, float):
                    cell_str = f"{cell:.1f}"
                elif i == 5:  # Retry count column
                    cell_str = str(cell) if cell > 0 else "-"

                col_widths[i] = max(col_widths[i], len(cell_str))

        col_widths[0] = max(col_widths[0], 30)

        # Header
        header_line = " | ".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(headers))
        separator = "=" * len(header_line)

        print(separator)
        print(header_line)
        print("-" * len(header_line))

        # Rows
        for row in table_data:
            file, status, compile_t, upload_t, total_d, retry_cnt = row

            # Status coloring
            if status == "success":
                status_str = f"{GREEN}✓ success{NC}"
            elif status == "failed":
                status_str = f"{RED}✗ failed{NC}"
            elif status == "interrupted":
                status_str = f"{YELLOW}⚠ interrupted{NC}"
            else:
                status_str = f"{YELLOW}⚠ skipped{NC}"

            # Time coloring
            compile_str = (
                f"{YELLOW}{compile_t:.1f}{NC}"
                if compile_t == max_compile_time and compile_t > 0
                else (f"{compile_t:.1f}" if isinstance(compile_t, float) else "-")
            )
            upload_str = (
                f"{YELLOW}{upload_t:.1f}{NC}"
                if upload_t == max_upload_time and upload_t > 0
                else (f"{upload_t:.1f}" if isinstance(upload_t, float) else "-")
            )
            total_str = (
                f"{YELLOW}{total_d:.1f}{NC}"
                if total_d == max_total_duration and total_d > 0
                else (f"{total_d:.1f}" if isinstance(total_d, float) else "-")
            )

            # Retry count coloring (highlight if > 0)
            retry_str = (
                f"{YELLOW}{retry_cnt}{NC}" if retry_cnt > 0 else "-"
            )

            # Construct the line with padding, considering the invisible ANSI characters
            line = f"{file:<{col_widths[0]}} | "
            line += f"{status_str:<{col_widths[1] + len(status_str) - len(strip_ansi(status_str))}} | "
            line += f"{compile_str:<{col_widths[2] + len(compile_str) - len(strip_ansi(compile_str))}} | "
            line += f"{upload_str:<{col_widths[3] + len(upload_str) - len(strip_ansi(upload_str))}} | "
            line += f"{total_str:<{col_widths[4] + len(total_str) - len(strip_ansi(total_str))}} | "
            line += f"{retry_str:<{col_widths[5] + len(retry_str) - len(strip_ansi(retry_str))}}"
            print(line)

        print(separator)

        # Use actual wall clock time for parallel mode, sum of individual times for serial
        if self.parallel_workers > 0 and hasattr(self, 'overall_start_time') and hasattr(self, 'overall_end_time'):
            actual_time = self.overall_end_time - self.overall_start_time
            minutes, seconds = divmod(actual_time, 60)
            print(f"Total execution time: {int(minutes)}m {seconds:.1f}s (parallel mode)")
        else:
            minutes, seconds = divmod(total_time, 60)
            print(f"Total execution time: {int(minutes)}m {seconds:.1f}s")
        print(separator)


def main():
    """Main function to parse arguments and start the runner."""
    description = """ESPHome Multi-Run Tool - Batch compile and upload multiple ESPHome configurations

BASIC USAGE:
  %(prog)s file1.yaml file2.yaml          Run specific files
  %(prog)s *.yaml                         Run files matching pattern
  %(prog)s -j 4 -p "*.yaml"               Run with 4 parallel workers

EXECUTION MODES:
  Serial (default):  Files run one by one with live output to console
  Parallel (-j N):   N files run simultaneously, output saved to logs/ directory

PARALLEL MODE KNOWN ISSUES:
  ⚠ Resource contention: Multiple workers may compete for shared resources
    (e.g., PlatformIO package downloads, build cache, compiler resources)
  ⚠ This can cause random compilation failures, especially on first run
  ⚠ The automatic retry mechanism helps mitigate these issues
  ⚠ Consider reducing worker count (-j) if experiencing frequent failures

COMMON EXAMPLES:
  # Run all YAML files serially with live output
  %(prog)s *.yaml

  # Parallel compile and upload 4 files
  %(prog)s -j 4 *.yaml

  # Parallel compile 4 files
  %(prog)s -j 4 -c *.yaml

EXCLUSION FILE FORMAT:
  Use glob patterns in the exclusion file (default: .esphome-run-exclude):
    # Comment lines start with #
    test-*.yaml          # Exclude all test files
    obsolete-device.yaml # Exclude specific file
    *-backup.yaml        # Exclude all backup files

FEATURES:
  ✓ Automatic retry for failed builds (max 2 retries)
  ✓ Color-coded output and progress tracking
  ✓ Detailed execution summary with timing statistics
  ✓ Graceful interrupt handling (Ctrl+C)
  ✓ Real-time progress display in parallel mode
  ✓ All output logged to logs/ directory
"""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="For more information: https://esphome.io/",
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="One or more YAML files to run"
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
        "-j", "--parallel",
        type=int,
        default=0,
        metavar="N",
        help="Run N builds in parallel (default: 0 = serial mode)\n"
             "WARNING: Parallel uploads may conflict if using same USB port.\n"
             "Recommended: use with -c/--compile-only flag",
    )
    parser.add_argument(
        "-c", "--compile-only",
        action="store_true",
        help="Only compile configurations, skip upload step\n"
             "Recommended for parallel mode to avoid USB port conflicts",
    )

    args = parser.parse_args()

    # Validate parallel argument
    if args.parallel < 0:
        print_color(RED, "Error: Parallel workers must be 0 or positive.")
        sys.exit(1)

    files_to_run = set(args.files)

    if args.pattern:
        for p in args.pattern:
            files_to_run.update(glob.glob(p))

    if args.dir:
        for d in args.dir:
            files_to_run.update(glob.glob(os.path.join(d, "*.yaml")))
            files_to_run.update(glob.glob(os.path.join(d, "*.yml")))

    if not files_to_run:
        print_color(RED, "Error: No YAML files specified.")
        parser.print_help()
        sys.exit(1)

    # Invert the logic: --no-logs is the default, --logs disables it.
    use_no_logs = not args.logs
    runner = ESPHomeRunner(sorted(list(files_to_run)), args.exclude_file, use_no_logs, args.parallel, args.compile_only)
    runner.run()


if __name__ == "__main__":
    # This ensures that KeyboardInterrupt is raised on SIGINT
    signal.signal(signal.SIGINT, signal.default_int_handler)
    main()
