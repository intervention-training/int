try:
    from math_verify.metric import math_metric
    from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig
    from math_verify.errors import TimeoutException
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")

import multiprocessing
import queue
import signal
import os

def compute_score(model_output: str, ground_truth: str, timeout: float = 10.0) -> bool:

    def last_boxed_only_string(string):
        idx = string.rfind("boxed{")
        if idx < 0:
            return None

        i = idx
        right_brace_idx = None
        num_left_braces_open = 0

        while i < len(string):
            if string[i] == "{":
                num_left_braces_open += 1
            if string[i] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1

        return string[idx:right_brace_idx + 1] if right_brace_idx is not None else None

    def remove_boxed(s: str) -> str:
        left = "boxed{"
        if s[:len(left)] != left:
            return None
        if s[-1] != "}":
            return None
        return s[len(left):-1]

    model_answer = last_boxed_only_string(model_output)
    if model_answer is None:
        return 0.
    model_answer = remove_boxed(model_answer)
    if model_answer is None:
        return 0.
    
    # Wrap the ground truth in \boxed{} format for verification
    model_answer_boxed = "\\boxed{" + model_answer + "}"
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    
    def _verify_worker(result_queue, gt_boxed, pred_boxed):
        # Ignore SIGALRM to prevent math_verify's timeout signal from causing issues
        # when the process is terminated externally
        def ignore_signal(signum, frame):
            pass
        
        # Set signal handler to ignore SIGALRM before doing anything
        old_handler = signal.signal(signal.SIGALRM, ignore_signal)
        
        try:
            verify_func = math_metric(
                gold_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
                pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
            )
            
            # Re-set signal handler right before calling verify_func, in case
            # math_metric initialization changed it
            signal.signal(signal.SIGALRM, ignore_signal)
            
            ret_score, _ = verify_func([gt_boxed], [pred_boxed])
            result_queue.put(ret_score)
        except Exception as e:
            # Catch all exceptions including TimeoutException to prevent propagation
            # This prevents "Exception ignored" messages during process cleanup
            result_queue.put(0.)
        finally:
            # Always reset signal handler to ignore before process exits
            # This prevents signal handlers from being triggered during cleanup
            try:
                signal.signal(signal.SIGALRM, ignore_signal)
            except:
                pass
    
    # Run verification in a separate process with hard timeout
    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=_verify_worker, args=(result_queue, ground_truth_boxed, model_answer_boxed))
    process.daemon = True  # Daemon processes are automatically terminated when parent exits
    process.start()
    process.join(timeout=timeout)
    
    if process.is_alive():
        # Use more aggressive termination: kill immediately to avoid signal handler issues
        # SIGKILL cannot be caught, so it won't trigger math_verify's signal handlers
        try:
            os.kill(process.pid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            # Process already dead
            pass
        process.join(timeout=0.1)
        return 0.
    
    try:
        ret_score = result_queue.get(timeout=0.1)
    except queue.Empty:
        return 0.

    return ret_score