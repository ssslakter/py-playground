from io import BytesIO

import torch, numpy as np
from colors import compress_img, extract_colors, scatter_plotly, segment_k_means
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
               Div(id="script"),
               Div(
                   H2("K-means segmentation"),
                   P("You can also segment the image using k-means. Upload an image above and click the button below."),
                   Form(Input(name="file", type="file", accept="image/*"),
                    Button("Segment"),
                    hx_post="/k-means",
                    hx_target="#segmented",
                    hx_indicator="#loader2",
                    ),
                   Loader("loader2"),
                   Div(id="segmented"),
                   ))


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


@rt('/k-means')
async def post(file: UploadFile):
    img = await file.read()
    img = Image.open(BytesIO(img)).convert("RGB")
    img = torch.tensor(np.array(img)/255).to('cuda')
    img = segment_k_means(img)
    fname = f"generated/{file.filename}"
    img.save(fname)
    return Img(src=fname)

serve()
