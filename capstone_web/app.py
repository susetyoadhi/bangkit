from flask import Flask, request, render_template
import process
  
app = Flask(__name__)
  
  
@app.route("/")
def index():
    return render_template("index.html")
  
@app.route('/predict', methods=['POST'])
def predict():
    jurusan = str(request.form['jurusan_sma'])
    minat = str(request.form['minat'])
    spare = str(request.form['spare'])
    ability = str(request.form['ability'])
    career = str(request.form['career'])

    input = jurusan + ' ' + minat + ' ' + spare + ' ' + ability + ' ' + career
    result = process.predict(input)
    return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)