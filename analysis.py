# analysis.py
# Marimo notebook style (cell markers with '# %%') 
# Contact verification: 24f2000773@ds.study.iitm.ac.in
#
# This file demonstrates:
# - At least two cells with variable dependencies
# - An interactive slider widget (explicitly created and displayed)
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
    # Use the same base and seasonal defined in Cell 1 context
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
# Cell 3: Interactive widget + dynamic markdown output (explicit slider and interact)
# Data flow: uses compute_estimated_revenue() from Cell 2 and `df` from Cell 1.
try:
    import ipywidgets as widgets
    from IPython.display import display, Markdown, clear_output
    has_widgets = True
except Exception as e:
    has_widgets = False
    print("ipywidgets not available in this environment. Run in Jupyter to see the interactive slider.")

if has_widgets:
    # Explicit FloatSlider widget
    mult_slider = widgets.FloatSlider(value=1.0, min=0.5, max=1.5, step=0.01, description='Marketing x', continuous_update=True, readout_format='.2f')
    out = widgets.Output()

    def update_display(multiplier):
        # Shares data with other cells: compute_estimated_revenue() depends on df
        est = compute_estimated_revenue(multiplier)
        stats = summary_stats(est)
        md = f"""### Estimated Revenue Summary (marketing multiplier = **{multiplier:.2f}**)
- Mean estimated revenue: **{stats['mean_estimated_revenue']:.2f}**
- Max estimated revenue: **{stats['max_estimated_revenue']:.2f}**
- Min estimated revenue: **{stats['min_estimated_revenue']:.2f}**

**Data flow note:** `df` (Cell 1) -> compute_estimated_revenue() (Cell 2) -> display (Cell 3).
"""
        with out:
            clear_output(wait=True)
            display(Markdown(md))

    # Callback for slider change
    def on_change(change):
        if change['name'] == 'value':
            update_display(change['new'])

    mult_slider.observe(on_change, names='value')

    # Also provide an interact-based control for compatibility
    try:
        from ipywidgets import interact
        interact(update_display, multiplier=mult_slider)
    except Exception:
        # Fallback: display the slider and output explicitly
        display(mult_slider)
        display(out)
        # Trigger initial display
        update_display(mult_slider.value)
else:
    # Non-interactive fallback: print static summary for multiplier=1.0
    est = compute_estimated_revenue(1.0)
    stats = summary_stats(est)
    print("Static summary (multiplier=1.0):", stats)

# End of Cell 3

# %%
# Cell 4: Plotting cell that depends on previous cells
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
