from io import BytesIO
from colors import compress_img, extract_colors, scatter_plotly
from fasthtml.common import *
from fh_plotly import plotly2fasthtml, plotly_headers
from PIL import Image

gridlink = Link(rel="stylesheet",
                href="https://cdnjs.cloudflare.com/ajax/libs/flexboxgrid/6.3.1/flexboxgrid.min.css", type="text/css")

headers = [picolink,
           gridlink,
           *plotly_headers,
           Link(rel="stylesheet", href="/static/styles.css")]

app, rt = fast_app(hdrs=headers)


def Loader(id="loader"):
    return H2("Loading...", id=id, cls="htmx-indicator", aria_busy="true")


@rt('/')
def get():
    return Titled('Picture colors visualizer!',
               P("This website allows you to visualize the colors of an image."),
               P("You can upload your own images by using the upload button below or use the default images."),
               Grid(*(image(id.strip('.png')) for id in os.listdir('../images'))),
               Form(Input(name="file", type="file", accept="image/*"),
                    Button("Upload"),
                    hx_post="/colorize",
                    hx_target="#script",
                    hx_indicator="#loader",
                    ),
               P("You should see a scatter plot of the colors below."),
               Loader(),
               Div(id="canvas"),
               Div(id="script"))


def image(id):
    return Div(Img(src=f"/images/{id}.png"),
                   id=f'img-{id}',
                   hx_get=f"/colorize/{id}",
                   hx_target="#script",
                   hx_indicator="#loader")


def colorize(img):
    img = compress_img(img)
    img = extract_colors(img)
    fig = scatter_plotly(img)
    fig.update_layout(width=800, height=800)
    return plotly2fasthtml(fig)

@rt('/colorize/{id}')
def get(id: str):
    img = Image.open(f"images/{id}.png").convert("RGB")
    return colorize(img)

@rt('/colorize')
async def post(file: UploadFile):
    img = await file.read()
    img = Image.open(BytesIO(img)).convert("RGB")
    return colorize(img)


serve()
