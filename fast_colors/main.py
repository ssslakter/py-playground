from io import BytesIO

from colors import compress_img, extract_colors, scatter_plotly, segment_k_means
from fasthtml.common import *
from fh_plotly import plotly2fasthtml, plotly_headers
from PIL import Image


headers = [picolink,
           *plotly_headers,
           Link(rel="stylesheet", href="/static/styles.css"),
           Script(src="/static/scripts.js")
           ]

app, rt = fast_app(hdrs=headers)



def Loader(id="loader"):
    return H2("Loading...", id=id, cls="htmx-indicator", aria_busy="true")


def FileForm():
    return Form(
        Input(name="file", id="file-inp", type="file", accept="image/*"),
        Button("Upload"),
        hx_post="/colorize",
        hx_target="#script",
        hx_indicator="#loader"
    )


@rt('/')
def get():
    return Titled('Picture colors visualizer!',
                  P("This website allows you to visualize the colors of an image."),
                  P("You can upload your own images by using the upload button below or use the default images."),
                  Grid(*(image(id.strip('.png')) for id in os.listdir('../images'))),
                  FileForm(),
                  P("You should see a scatter plot of the colors below."),
                  Div(
                      Img(id="preview"),
                    Div(Loader(), id="canvas", cls="canvas"),
                      cls="split-container"
                  ),
                  Div(id="script"),
                  Div(
                      H2("K-means segmentation"),
                      P("You can also segment the image using k-means. Upload an image above and click the button below."),
                      Form(Input(name="file", type="file", accept="image/*"),
                           Input(name="k", type="number", value="2"),
                           Button("Segment"),
                           hx_post="/k-means",
                           hx_target="#segmented",
                           hx_indicator="#loader2"
                           ),
                      Loader("loader2"),
                      Div(id="segmented"),
                  )), Footer("Made with ❤️ by Ssslakter", cls="container")


def image(id):
    return Div(Img(src=f"/images/{id}.png"),
               cls="img-card",
               id=f'img-{id}',
               hx_post=f"/colorize/{id}",
               hx_target="#script",
               hx_indicator="#loader")


def colorize(img):
    img = compress_img(img)
    img = extract_colors(img)
    fig = scatter_plotly(img)
    fig.update_layout(autosize=True,
                      width=None,
                      height=None,
                      margin=dict(l=0, r=0, t=0, b=0))
    return plotly2fasthtml(fig)


@rt('/colorize/{id}')
def post(id: str):
    img = Image.open(f"images/{id}.png").convert("RGB")
    return colorize(img)


@rt('/colorize')
async def post(file: UploadFile):
    img = await file.read()
    img = Image.open(BytesIO(img)).convert("RGB")
    return colorize(img)


@rt('/k-means')
async def post(file: UploadFile, k: int = 2):
    if k < 1: k = 1
    Path('generated').mkdir(exist_ok=True)
    name, ext = file.filename.split('.')
    fname = f"generated/{name}+{k}.{ext}"
    if not os.path.exists(fname):
        img = await file.read()
        img = Image.open(BytesIO(img)).convert("RGB")
        img = await run_in_threadpool(segment_k_means, img, k)
        img.save(fname)
    return Img(src=fname)

serve()
