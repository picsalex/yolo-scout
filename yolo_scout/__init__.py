import os
import warnings

# Must be set before numba is imported anywhere (its thread pool can't shrink later)
os.environ.setdefault("NUMBA_NUM_THREADS", "4")

warnings.filterwarnings(
    "ignore", message=r"invalid escape sequence.*", category=SyntaxWarning, module=r"glob2\.fnmatch"
)
warnings.filterwarnings("ignore", message="QuickGELU mismatch.*", category=UserWarning, module=r"open_clip\.factory")
