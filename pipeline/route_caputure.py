#!/usr/bin/env python3
import argparse
import os
import subprocess
import signal
import sys
import shutil
import time
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINE_DIR.parent


def resolve_project_path(raw_path):
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "..":
        return path.resolve()
    cwd_candidate = path.resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (PROJECT_ROOT / path).resolve()


def get_terminal_cmd(cmd_list, env, block=False):
    """
    Wraps a command to run in a new terminal window.
    Returns (terminal_cmd_list, is_blocking)
    """
    # Convert command list to string for shell execution
    env_str = " ".join([f"{k}={v}" for k, v in env.items() if k.startswith("MODELD_") or k == "PYTHONPATH" or k == "PATH"])
    cmd_str = f"{env_str} {' '.join(cmd_list)}"
    
    # Common terminal emulators
    # Format: (terminal, non_blocking_args, blocking_args)
    terminals = [
        ("gnome-terminal", 
         ["--", "bash", "-c", f"{cmd_str}; echo 'Process finished. Press enter to exit.'; read"],
         ["--wait", "--", "bash", "-c", f"{cmd_str}; echo 'Process finished. Press enter to exit.'; read"]),
        ("konsole", 
         ["-e", "bash", "-c", f"{cmd_str}; echo 'Process finished. Press enter to exit.'; read"],
         ["--nofork", "-e", "bash", "-c", f"{cmd_str}; echo 'Process finished. Press enter to exit.'; read"]),
        ("xfce4-terminal", 
         ["-x", "bash", "-c", f"{cmd_str}; echo 'Process finished. Press enter to exit.'; read"],
         ["--disable-server", "-x", "bash", "-c", f"{cmd_str}; echo 'Process finished. Press enter to exit.'; read"]),
        ("xterm", 
         ["-e", "bash", "-c", f"{cmd_str}; echo 'Process finished. Press enter to exit.'; read"],
         ["-e", "bash", "-c", f"{cmd_str}; echo 'Process finished. Press enter to exit.'; read"]), # xterm -e blocks by default
    ]

    for term, args_nb, args_b in terminals:
        if shutil.which(term):
            if block:
                # gnome-terminal --wait blocks. xterm blocks by default.
                return [term] + args_b, True
            else:
                return [term] + args_nb, False # terminals usually return immediately (daemonize) or block.
            
    return None, False


def count_raw_frames(dataset_dir):
    frame_count = 0
    latest_mtime = None
    for raw_dir in dataset_dir.glob("segment_*/raw"):
        if not raw_dir.is_dir():
            continue
        for frame_path in raw_dir.glob("*.png"):
            frame_count += 1
            try:
                mtime = frame_path.stat().st_mtime
            except OSError:
                continue
            latest_mtime = mtime if latest_mtime is None else max(latest_mtime, mtime)
    return frame_count, latest_mtime


def wait_for_replay_or_dataset_idle(replay_process, dataset_dir, idle_seconds):
    initial_count, _ = count_raw_frames(dataset_dir)
    last_count = initial_count
    last_change = time.monotonic()
    saw_new_frames = False

    while True:
        returncode = replay_process.poll()
        if returncode is not None:
            return returncode, False

        frame_count, _ = count_raw_frames(dataset_dir)
        if frame_count != last_count:
            if frame_count > initial_count:
                saw_new_frames = True
            last_count = frame_count
            last_change = time.monotonic()

        idle_for = time.monotonic() - last_change
        if saw_new_frames and idle_seconds > 0 and idle_for >= idle_seconds:
            print(
                f"\nNo new raw frames for {idle_seconds:.0f}s after capture started; "
                "stopping replay and moving on."
            )
            replay_process.terminate()
            try:
                return replay_process.wait(timeout=10), True
            except subprocess.TimeoutExpired:
                replay_process.kill()
                return replay_process.wait(), True

        time.sleep(2)


def run_pipeline(args):
    """
    Runs the Openpilot replay tool and modeld detection script concurrently.
    """
    
    # Ensure .local/bin is in PATH for uv detection
    local_bin = os.path.expanduser("~/.local/bin")
    if local_bin not in os.environ["PATH"]:
        os.environ["PATH"] = f"{local_bin}:{os.environ['PATH']}"

    # Hardcoded configuration (formerly general args)
    openpilot_dir = PROJECT_ROOT.parent / "openpilot"
    python_cmd = None
    new_terminal_modeld = args.new_terminal_modeld
    new_terminal_replay = False
    dry_run = False

    # Resolve openpilot directory
    op_dir = openpilot_dir.resolve()
    if not op_dir.exists():
        print(f"ERROR: Openpilot directory not found at {op_dir}")
        print(f"Please ensure openpilot exists next to {PROJECT_ROOT}")
        return 1

    # Paths (relative to openpilot_dir unless absolute)
    replay_path = Path(args.replay_path)
    if not replay_path.is_absolute():
        replay_path = op_dir / replay_path

    modeld_path = Path(args.modeld_path)
    if not modeld_path.is_absolute():
        modeld_path = op_dir / modeld_path

    dataset_dir = resolve_project_path(args.dataset_dir).resolve()

    # Check if tools exist before starting any long-running process.
    missing_tools = False
    if not replay_path.exists():
        print(f"ERROR: Replay tool not found at {replay_path}")
        print("Run setup_openpilot.sh or build Openpilot so tools/replay/replay exists.")
        missing_tools = True
    elif not os.access(replay_path, os.X_OK):
        print(f"ERROR: Replay tool is not executable: {replay_path}")
        missing_tools = True

    if not modeld_path.exists():
        print(f"ERROR: Modeld script not found at {modeld_path}")
        print("Run setup_openpilot.sh so the custom modeld scripts are copied into Openpilot.")
        missing_tools = True

    if missing_tools:
        return 1

    # Environment variables for modeld
    modeld_env = os.environ.copy()
    modeld_env["MODELD_DATASET_DIR"] = str(dataset_dir)
    modeld_env["MODELD_MAX_SEGMENT"] = str(args.max_segment)
    modeld_env["MODELD_SEGMENT_FRAMES"] = str(args.segment_frames)
    modeld_env["MODELD_DATASET_TARGET_FPS"] = str(args.dataset_fps)
    # Ensure python path includes openpilot dir
    modeld_env["PYTHONPATH"] = f"{op_dir}:{modeld_env.get('PYTHONPATH', '')}"

    # Construct commands
    replay_cmd = [str(replay_path), args.route, "--no-loop"] + args.replay_flags.split()
    
    # Helper to construct modeld cmd manually since we removed args.python_cmd
    cmd = []
    if python_cmd:
        cmd.extend(python_cmd.split())
    else:
        # Auto-detect environment
        if (op_dir / "poetry.lock").exists():
            print("Detected poetry environment.")
            cmd.extend(["poetry", "run", "python3"])
        elif (op_dir / "Pipfile").exists():
            print("Detected pipenv environment.")
            cmd.extend(["pipenv", "run", "python3"])
        elif (op_dir / "uv.lock").exists():
            print("Detected uv environment.")
            
            # Find uv
            uv_path = shutil.which("uv")
            if not uv_path:
                print("WARNING: uv not found in PATH via shutil.which even after updating PATH.")
                print(f"PATH is: {os.environ['PATH']}")
                uv_path = "uv" # Fallback
            
            print(f"Using uv at: {uv_path}")
            cmd.extend([uv_path, "run", "python3"])
        else:
            venv_python = op_dir / "venv" / "bin" / "python3"
            if venv_python.exists():
                print(f"Detected venv at {venv_python}")
                cmd.append(str(venv_python))
            else:
                 cmd.append("python3")
    cmd.append(str(modeld_path))
    modeld_cmd = cmd

    print("="*40)
    print("Openpilot Pipeline Runner")
    print("="*40)
    print(f"Working Directory: {op_dir}")
    print(f"Replay Command: {' '.join(replay_cmd)}")
    print(f"Modeld Command: {' '.join(modeld_cmd)}")
    print(
        "Modeld Env: "
        f"MODELD_DATASET_DIR={dataset_dir}, "
        f"MODELD_MAX_SEGMENT={args.max_segment}, "
        f"MODELD_SEGMENT_FRAMES={args.segment_frames}, "
        f"MODELD_DATASET_TARGET_FPS={args.dataset_fps}"
    )
    if new_terminal_modeld:
        print("Modeld will run in a NEW TERMINAL window.")
    else:
        print("Modeld will run as a managed background process.")
    if new_terminal_replay:
        print("Replay will run in a NEW TERMINAL window.")
    print("="*40)

    if dry_run:
        print("Dry run complete. Exiting.")
        return 0

    # Start modeld
    modeld_process = None
    use_terminal_modeld = False
    
    if new_terminal_modeld:
        term_cmd, _ = get_terminal_cmd(modeld_cmd, modeld_env, block=False)
        if term_cmd:
            print(f"\nLaunching modeld in new terminal: {' '.join(term_cmd)}")
            subprocess.Popen(term_cmd, cwd=op_dir) 
            use_terminal_modeld = True
            print("Modeld launched in separate window. Check it for output/errors.")
        else:
            print("\nWARNING: No supported terminal emulator found (gnome-terminal, konsole, xfce4-terminal, xterm).")
            print("Falling back to same-terminal execution.")
    
    if not use_terminal_modeld:
        print("\nStarting modeld in background (Output Suppressed)...")
        try:
            # Run from op_dir so relative imports/paths work
            modeld_process = subprocess.Popen(
                modeld_cmd, 
                env=modeld_env, 
                cwd=op_dir, 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL
            )
        except Exception as e:
            print(f"Error starting modeld: {e}")
            return 1

    # Give modeld a moment to initialize
    time.sleep(2)

    # Start replay
    print("\nStarting replay...")
    try:
        if new_terminal_replay:
            term_cmd, is_blocking = get_terminal_cmd(replay_cmd, os.environ.copy(), block=True)
            if term_cmd:
                print(f"Launching replay in new terminal: {' '.join(term_cmd)}")
                if is_blocking:
                    subprocess.run(term_cmd, check=True, cwd=op_dir)
                else:
                    print("WARNING: Terminal emulator does not support blocking wait. Pipeline might finish prematurely.")
                    print("Attempting to run anyway, but modeld might be killed early.")
                    subprocess.run(term_cmd, check=True, cwd=op_dir)
            else:
                 print("\nWARNING: No supported terminal emulator found for replay.")
                 print("Falling back to same-terminal execution.")
                 subprocess.run(replay_cmd, check=True, cwd=op_dir)
        else:
            replay_process = subprocess.Popen(replay_cmd, cwd=op_dir)
            returncode, stopped_for_idle = wait_for_replay_or_dataset_idle(
                replay_process=replay_process,
                dataset_dir=dataset_dir,
                idle_seconds=args.auto_stop_idle_seconds,
            )
            if returncode != 0 and not stopped_for_idle:
                raise subprocess.CalledProcessError(returncode, replay_cmd)
            
    except subprocess.CalledProcessError as e:
        print(f"Replay tool failed with exit code {e.returncode}")
        return 1
    except OSError as e:
        print(f"Error starting replay: {e}")
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 130
    finally:
        # Terminate modeld when replay finishes or is interrupted
        if modeld_process:
            print("\nStopping modeld...")
            modeld_process.terminate()
            try:
                modeld_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                modeld_process.kill()
        elif use_terminal_modeld:
            print("\nReplay finished.")
            print("NOTE: Modeld running in the separate terminal may keep running or wait for input.")
            print("Please close that window manually if it remains open.")
            
        print("Pipeline finished.")

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Openpilot Replay and Modeld Pipeline")
    
    parser.add_argument("--route", type=str, default="d34c14daa88a1e86/000000ca--7c5d326170",
                        help="Route ID (default: 'd34c14daa88a1e86/000000ca--7c5d326170')")
    parser.add_argument("--replay-flags", type=str, default="",
                        help="Flags for replay (default: '')")
    parser.add_argument("--replay-path", type=str, default="./tools/replay/replay",
                        help="Path to replay executable (default: ./tools/replay/replay)")

    # Modeld arguments
    parser.add_argument("--dataset-dir", type=str, default="./datasets/leaf_run",
                        help="Directory for modeld output (default: ./datasets/leaf_run)")
    parser.add_argument("--max-segment", type=int, default=20,
                        help="Max segment for modeld (default: 20)")
    parser.add_argument("--segment-frames", type=int, default=175,
                        help="Frames per segment (MODELD_SEGMENT_FRAMES) (default: 175)")
    parser.add_argument("--dataset-fps", type=float, default=10.0,
                        help="Target dataset capture FPS for raw/telemetry/features (default: 10.0)")
    parser.add_argument("--modeld-path", type=str, default="selfdrive/modeld/modeld_detection_second.py",
                        help="Path to modeld script (default: selfdrive/modeld/modeld_detection_second.py)")
    parser.add_argument("--auto-stop-idle-seconds", type=float, default=45.0,
                        help="Stop replay after this many seconds with no new raw frames once capture starts (default: 45). Use 0 to disable.")
    parser.add_argument("--new-terminal-modeld", action="store_true",
                        help="Run modeld in a separate terminal for debugging. This disables managed shutdown.")



    args = parser.parse_args()
    
    # Ensure dataset directory exists
    resolve_project_path(args.dataset_dir).mkdir(parents=True, exist_ok=True)

    sys.exit(run_pipeline(args))
