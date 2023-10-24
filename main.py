import tempfile
import os
import subprocess
from flask import Flask, request, render_template, redirect, send_file, jsonify
from skimage.transform import resize
from PIL import Image
from tensorflow.keras.models import load_model
from skimage import io
import base64
import glob
import numpy as np
import tensorflow as tf

app = Flask(__name__)
model = load_model('modelo_entrenado.h5')

main_html = """
<html>
<head>
</head>
<body>   
<script>
  var mousePressed = false;
  var lastX, lastY;
  var ctx;

   function getRndInteger(min, max) {
    return Math.floor(Math.random() * (max - min) ) + min;
   }

  function InitThis() {
      ctx = document.getElementById('myCanvas').getContext("2d");


      numero = getRndInteger(0, 10);
      letra = ["Katakana A", "Katakana E", "Katakana I","Katakana O","Katakana U"];
      random = Math.floor(Math.random() * letra.length);
      aleatorio = letra[random];

      document.getElementById('mensaje').innerHTML  = 'Dibujando un ' + aleatorio;
      document.getElementById('numero').value = aleatorio;

      $('#myCanvas').mousedown(function (e) {
          mousePressed = true;
          Draw(e.pageX - $(this).offset().left, e.pageY - $(this).offset().top, false);
      });

      $('#myCanvas').mousemove(function (e) {
          if (mousePressed) {
              Draw(e.pageX - $(this).offset().left, e.pageY - $(this).offset().top, true);
          }
      });

      $('#myCanvas').mouseup(function (e) {
          mousePressed = false;
      });
  	    $('#myCanvas').mouseleave(function (e) {
          mousePressed = false;
      });
  }

  function Draw(x, y, isDown) {
      if (isDown) {
          ctx.beginPath();
          ctx.strokeStyle = 'black';
          ctx.lineWidth = 11;
          ctx.lineJoin = "round";
          ctx.moveTo(lastX, lastY);
          ctx.lineTo(x, y);
          ctx.closePath();
          ctx.stroke();
      }
      lastX = x; lastY = y;
  }

  function clearArea() {
      // Use the identity matrix while clearing the canvas
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  }

  //https://www.askingbox.com/tutorial/send-html5-canvas-as-image-to-server
  function prepareImg() {
     var canvas = document.getElementById('myCanvas');
     document.getElementById('myImage').value = canvas.toDataURL();
  }



</script>
<body onload="InitThis();">
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/1.7.1/jquery.min.js" type="text/javascript"></script>
    <script type="text/javascript" ></script>
    <div align="center">
        <h1 id="mensaje">Dibujando...</h1>
        <canvas id="myCanvas" width="200" height="200" style="border:2px solid black"></canvas>
        <br/>
        <br/>
        <button onclick="javascript:clearArea();return false;">Borrar</button>
    </div>
    <div align="center">
      <form method="post" action="upload" onsubmit="javascript:prepareImg();"  enctype="multipart/form-data">
      <input id="numero" name="numero" type="hidden" value="">
      <input id="myImage" name="myImage" type="hidden" value="">
      <input id="bt_upload" type="submit" value="Enviar">
      </form>
    </div>
</body>
</html>

"""


@app.route("/")
def main():
    return(main_html)


@app.route('/upload', methods=['POST'])
def upload():
    try:
        # check if the post request has the file part
        img_data = request.form.get('myImage').replace("data:image/png;base64,","")
        aleatorio = request.form.get('numero')
        print(aleatorio)
        with tempfile.NamedTemporaryFile(delete = False, mode = "w+b", suffix='.png', dir=str(aleatorio)) as fh:
            fh.write(base64.b64decode(img_data))
        #file = request.files['myImage']
        print("Image uploaded")
    except Exception as err:
        print("Error occurred")
        print(err)

    return redirect("/", code=302)


@app.route('/prepare', methods=['GET'])
def prepare_dataset():
    images = []
    d = ["Katakana A", "Katakana E", "Katakana I","Katakana O","Katakana U"]
    digits = []
    for digit in d:
        filelist = glob.glob('{}/*.png'.format(digit))
        images_read = io.concatenate_images(io.imread_collection(filelist))
        images_read = images_read[:, :, :, 3]
        digits_read = np.array([digit] * images_read.shape[0])
        images.append(images_read)
        digits.append(digits_read)
    images = np.vstack(images)
    digits = np.concatenate(digits)
    np.save('X.npy', images)
    np.save('y.npy', digits)
    return "Archvos .npy generados correctamente"

@app.route('/save', methods=['GET'])
def process_and_save_images():
    # Load the X.npy file
    X_puro = np.load('X.npy')

    X = []
    size = (28, 28)
    for x in X_puro:
        X.append(resize(x, size))
    X = np.array(X)

    directory_destino = 'Prediccion'

    if not os.path.exists(directory_destino):
        os.mkdir(directory_destino)

    for i, image in enumerate(X):
        filename = f'{directory_destino}/imagen_{i}.jpg'
        cv.imwrite(filename, (image * 255).astype(np.uint8))

    return f'Imagenes .jpg guardadas en "{directory_destino}".'

@app.route('/X.npy', methods=['GET'])
def download_X():
    return send_file('X.npy')


@app.route('/y.npy', methods=['GET'])
def download_y():
    return send_file('y.npy')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({'error': 'No se ha proporcionado ninguna imagen'})
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'Nombre de archivo de imagen no válido'})
        
        image = Image.open(image_file)
        image = image.convert("L") 
        image = image.resize((28, 28), Image.ANTIALIAS)
        
        imagen_array = np.array(image)
        imagen_tensor = tf.convert_to_tensor(imagen_array, dtype=tf.float32)
        imagen_tensor = tf.reshape(imagen_tensor, (1, 28, 28, 1))
        
        prediction = model.predict(imagen_tensor)

        classes = ["Katakana A", "Katakana E", "Katakana I", "Katakana O", "Katakana U"]
 
        predicted_class = classes[np.argmax(prediction)]
        return jsonify({'prediction': predicted_class})
    elif request.method == 'GET':
        return render_template("Prediccion.html")

   
    


    


@app.route('/predicciones')
def show_predictions():
    nums = request.args.get('nums')
    img_data = request.args.get('img_data')
    componentes = nums.split(', ')
    nums = [float(componente) for componente in componentes]
    letras = ["Katakana A", "Katakana E", "Katakana I","Katakana O","Katakana U"]
    if img_data is not None:
        return render_template('Prediccion.html', nums=nums, letras=letras, img_data=img_data)
    else:
        return redirect("/", code=302)

if __name__ == "__main__":
    digits = ["Katakana A", "Katakana E", "Katakana I","Katakana O","Katakana U"]
    for d in digits:
        if not os.path.exists(str(d)):
            os.mkdir(str(d))
    app.run()
