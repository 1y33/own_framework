import time, random, plotext as plt
from rich.console import Console
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live

console = Console()

# buffers
log_lines   = []   # left-hand rolling log
loss_vals   = []   # for plot
steps       = []

# -------- helpers --------
def make_log_panel(max_rows: int = 30) -> Panel:
    """Fixed-height slice of latest log lines."""
    visible = "\n".join(log_lines[-max_rows:])
    return Panel(visible, title="Loss Log", border_style="cyan")

import plotext as plt
from rich.panel import Panel
from rich.console import Console

console  = Console()
LEFT_PAD = 38                     # width you fixed for the log pane

def make_plot_panel() -> Panel:
    # ---- 1) clear the ENTIRE figure, data + axes ----
    plt.clear_data()              # clears previous series
    plt.clear_figure()            # resets axes / titles

    # ---- 2) size to the REAL right-hand area ----
    term_w, term_h = console.size
    right_w  = max(20, term_w - LEFT_PAD - 3)   # -3 = borders
    right_h  = max(10, term_h - 4)              # leave room for panel title
    plt.plotsize(right_w, right_h)

    # ---- 3) re-draw ----
    if steps:                     # only plot when we have data
        plt.plot(steps, loss_vals, marker='dot')
    plt.title("Loss vs Step")
    plt.xlabel("Step")
    plt.ylabel("Loss")

    # ---- 4) build string once per refresh ----
    plot_str = plt.build()
    return Panel(plot_str, title="Live Plot", border_style="magenta")

# -------- static layout: left fixed width, right fills rest --------
layout = Layout()
layout.split_row(
    Layout(name="left",  size=38),   # <= fixed char width
    Layout(name="right")             # fills remainder
)

with Live(layout, refresh_per_second=4, screen=True):
    for step in range(1, 201):
        # --- simulate training update ---
        loss = random.uniform(3.5, 7)
        log_lines.append(f"Step {step:03d} | Loss: {loss:.4f}")
        steps.append(step)
        loss_vals.append(loss)

        # --- update panels in-place ---
        layout["left"].update(make_log_panel(max_rows=console.size.height - 4))
        layout["right"].update(make_plot_panel())

        time.sleep(0.1)
