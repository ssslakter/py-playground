from uuid import uuid4

from fasthtml.common import *
from plotly.io import to_json

plotly_headers = [
    Script(src="https://cdn.plot.ly/plotly-latest.min.js")
]

def plotly2fasthtml(chart):
    chart_json = to_json(chart)
    return Script(f"""
        var plotly_data = {chart_json};
        Plotly.react('canvas', plotly_data.data, plotly_data.layout);
        """)