from PIL import Image
from colors import compress_img, extract_colors, scatter_plotly
from fasthtml.common import *
from fh_plotly import plotly2fasthtml, plotly_headers

gridlink = Link(rel="stylesheet",
                href="https://cdnjs.cloudflare.com/ajax/libs/flexboxgrid/6.3.1/flexboxgrid.min.css", type="text/css")

headers = [picolink,
           gridlink,
           *plotly_headers,
           Script(open(Path(__file__).parent / "markdown.js").read())]

styles = [
    Style(".htmx-indicator {opacity:0;}"),
    Style(".htmx-indicator .htmx-request {opacity:1;}"),
]

app, rt = fast_app(hdrs=headers+styles)


def Loader(id="loader"):
    return Div("Loading...", id=id, cls="htmx-indicator")


@rt('/')
def get():
    return Div(P('Hello World!'),
               Div(*(image(id.strip('.png')) for id in os.listdir('./images')), cls="row"),
               Loader(),
               Div(id="main"))


def image(id):
    if os.path.exists(f"images/{id}.png"):
        return Div(Img(src=f"/images/{id}.png", width=200),
                   id=f'img-{id}',
                   hx_get=f"/colorize/{id}",
                   hx_target="#main",
                   hx_indicator="#loader")
    else: return P(f'Image {id} not found')


@rt('/colorize/{id}')
def get(id: str):
    img = Image.open(f"images/{id}.png").convert("RGB")
    img = compress_img(img)
    img = extract_colors(img)
    fig = scatter_plotly(img)
    fig.update_layout(width=800, height=800)
    return Div(plotly2fasthtml(fig))


serve()
