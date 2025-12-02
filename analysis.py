import marimo

__generated_with = "0.8.0"
app = marimo.App()

@app.cell
def __():
    import marimo as mo
    # Email: 24f2000773@ds.study.iitm.ac.in
    return mo,

@app.cell
def __(mo):
    # Cell 1: Define the interactive slider widget
    # This initiates the data flow by creating a user input source
    slider = mo.ui.slider(start=1, end=50, step=1, label="Select Input Variable (x)")
    return slider,

@app.cell
def __(slider):
    # Cell 2: Variable dependency calculation
    # Data flow: Receives 'slider' object from Cell 1
    # Extracts the current value and computes a dependent variable (y = x^2)
    x = slider.value
    y = x ** 2
    return x, y

@app.cell
def __(mo, slider, x, y):
    # Cell 3: Dynamic Markdown Output
    # Data flow: Receives 'x' and 'y' from Cell 2 to update the text dynamically
    # Also embeds the 'slider' widget for interaction
    mo.md(
        f"""
        # Interactive Relationship Demo
        
        Adjust the input variable to see the quadratic relationship:
        
        {slider}
        
        ---
        
        ### Real-time Analysis
        * **Input Variable (x):** {x}
        * **Dependent Variable (x²):** {y}
        """
    )
    return
