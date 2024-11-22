from flask import Flask, render_template
from photo import photo_blueprint
from video import video_blueprint
from webcam import webcam_blueprint

app = Flask(__name__)

# Đăng ký blueprint cho các routes liên quan đến xử lý ảnh
app.register_blueprint(photo_blueprint, url_prefix="/photo")

# Đăng ký blueprint cho các routes liên quan đến xử lý video
app.register_blueprint(video_blueprint, url_prefix="/video")

# Đăng ký blueprint cho các routes liên quan đến webcam
app.register_blueprint(webcam_blueprint, url_prefix="/webcam")

@app.route("/")
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)