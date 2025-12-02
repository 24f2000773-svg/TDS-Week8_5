# analysis.py
# Marimo notebook style (cell markers with '# %%') 
# Contact verification: 24f2000773@ds.study.iitm.ac.in
#
# This file is structured so editors or notebook runners that support
# script-style cells (e.g., VSCode, PyCharm, or Marimo if it supports # %% markers)
# can run each cell independently. The file demonstrates:
# - At least two cells with variable dependencies
# - An interactive slider widget
# - Dynamic markdown output based on widget state
# - Comments documenting the data flow between cells

# %%
# Cell 1: Data generation and preprocessing
# Data flow: create `df` which downstream cells consume.
import numpy as np
import pandas as pd

np.random.seed(42)
months = pd.date_range("2023-01-01", periods=12, freq='MS')
# Generate synthetic variables with relationships:
# revenue is influenced by month (seasonal) and marketing_spend
marketing_spend = np.linspace(50, 150, len(months)) + np.random.normal(scale=5, size=len(months))
seasonal = 20 * np.sin(2 * np.pi * (np.arange(len(months)) / 12))
base = 200
revenue = base + seasonal + 0.8 * marketing_spend + np.random.normal(scale=8, size=len(months))

df = pd.DataFrame({
    "month": months,
    "marketing_spend": marketing_spend.round(1),
    "revenue": revenue.round(2),
})
# End of Cell 1

# %%
# Cell 2: Functions that depend on `df`
# Data flow: reads `df`, defines compute functions used by display cell.
def compute_estimated_revenue(multiplier):
    """Estimate revenue given a multiplier applied to marketing_spend.
    This depends on `df` generated in Cell 1."""
    est = df.copy()
    est['estimated_revenue'] = (base + seasonal + 0.8 * (est['marketing_spend'] * multiplier))
    return est

def summary_stats(est_df):
    return {
        "mean_estimated_revenue": float(est_df['estimated_revenue'].mean()),
        "max_estimated_revenue": float(est_df['estimated_revenue'].max()),
        "min_estimated_revenue": float(est_df['estimated_revenue'].min()),
    }
# End of Cell 2

# %%
# Cell 3: Interactive widget + dynamic markdown output
# Data flow: uses compute_estimated_revenue() from Cell 2 and `df` from Cell 1.
try:
    # Import widget libraries commonly available in Jupyter/Marimo environments
    import ipywidgets as widgets
    from IPython.display import display, Markdown, clear_output
except Exception as e:
    print("Interactive widgets not available in this environment. Run in a Jupyter-like environment to use widgets.")

# Slider controls marketing multiplier between 0.5x and 1.5x
mult_slider = widgets.FloatSlider(value=1.0, min=0.5, max=1.5, step=0.05, description='Marketing x', continuous_update=True)

out = widgets.Output()

def on_change(change):
    # This callback responds to slider value changes and updates dynamic markdown.
    with out:
        clear_output(wait=True)
        multiplier = change['new']
        est = compute_estimated_revenue(multiplier)
        stats = summary_stats(est)
        # Dynamic markdown rendering based on widget state
        md = f"""### Estimated Revenue Summary (marketing multiplier = **{multiplier:.2f}**)
- Mean estimated revenue: **{stats['mean_estimated_revenue']:.2f}**
- Max estimated revenue: **{stats['max_estimated_revenue']:.2f}**
- Min estimated revenue: **{stats['min_estimated_revenue']:.2f}**

**Data flow note:** `df` (Cell 1) -> compute_estimated_revenue() (Cell 2) -> display (Cell 3).
"""
        display(Markdown(md))

mult_slider.observe(on_change, names='value')

display(mult_slider)
display(out)

# Trigger initial display
on_change({'new': mult_slider.value})
# End of Cell 3

# %%
# Cell 4: Optional plotting cell that depends on previous cells
# Data flow: uses estimated dataframe returned by compute_estimated_revenue()
def plot_estimated_revenue(multiplier):
    import matplotlib.pyplot as plt
    est = compute_estimated_revenue(multiplier)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(est['month'], est['estimated_revenue'], marker='o', linestyle='-')
    ax.set_title(f'Estimated Revenue (mult={multiplier:.2f})')
    ax.set_xlabel('Month')
    ax.set_ylabel('Estimated Revenue')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

# If running interactively, users can call plot_estimated_revenue(1.1) to see plot.
# End of file
