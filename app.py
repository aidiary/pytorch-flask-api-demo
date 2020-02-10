import os
from flask import Flask, render_template, request, redirect
from inference import get_prediction, format_class_name


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        # templateから送られてきたFileStorageオブジェクト
        file = request.files.get('file')
        if not file:
            return
        # クライアントから送られた画像ファイルを読み込める
        image_bytes = file.read()
        class_id, class_name = get_prediction(image_bytes)
        class_name = format_class_name(class_name)
        return render_template('result.html', class_id=class_id, class_name=class_name)

    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True, port=5000)
