from flask import Flask, render_template, request
from PIL import Image
from mllib.mlutils import my_model


app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route("/image", methods=["POST"])
def home():
    request.files['file'].save("./temp_image.png")
    return str(my_model.predict("./temp_image.png"))

if __name__ == '__main__':
    app.run(debug = True)
