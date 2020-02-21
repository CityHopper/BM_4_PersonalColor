from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from keras.backend import set_session
from sklearn.externals import joblib

from predict import face_classification

app = Flask(__name__)

upload_dir = os.path.join('./static/image')


#업로드 HTML 렌더링
@app.route('/')
def render_file():
   return render_template('upload.html')

#파일 업로드 처리
@app.route('/fileUpload', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['img_file']
      #저장할 경로 + 파일명
      f.save(upload_dir +'/' + secure_filename(f.filename))
      return render_template("upload.html", upload_file_name=f.filename)


# 이미지 다운로드
@app.route("/img/<filename>/<gubun>")
def download_img(filename, gubun):
    download_dir = ""

    if gubun == "face":
        download_dir = upload_dir
    elif gubun == "pal":
        download_dir = "static/palette/"

    return send_from_directory(download_dir, filename, as_attachment=True, mimetype="image/jpeg")


# 계절별 화장품 이미지 다운로드
@app.route("/cosmetic_img/<gubun_season>/<gubun_cosmetic>")
def download_cosmetic_img(gubun_season, gubun_cosmetic):
    download_dir = "static/cosmetic/" + gubun_season + "/" + gubun_cosmetic + "/"
    filename = gubun_cosmetic + ".jpg"

    return send_from_directory(download_dir, filename, as_attachment=True, mimetype="image/jpeg")

@app.route("/predict", methods = ['GET', 'POST'])
def predict_face():
   file_name = request.form["i_filename"]

   print("예측() 파일명 : " + file_name)

   face_tone, face_pccs, face_season = face_classification(model, sess, graph, file_name).get_predicted()
   #face_tone, face_pccs, face_season = face_classification("model", "sess", "graph", file_name).get_predicted()

   print("예측 결과 : " + face_tone, face_pccs, face_season)
   
   face_tone_kor = ""
   if face_tone == "warm": face_tone_kor = "웜"
   elif face_tone == "cool": face_tone_kor = "쿨"

   face_season_kor = ""
   if face_season == "spring": face_season_kor = "봄"
   elif face_season == "summer": face_season_kor = "여름"
   elif face_season == "autumn": face_season_kor = "가을"
   elif face_season == "winter": face_season_kor = "겨울"
    
   return render_template("upload.html", face_tone=face_tone, face_pccs=face_pccs, face_season=face_season, face_tone_kor=face_tone_kor, face_season_kor=face_season_kor, upload_file_name=file_name)


if __name__ == '__main__':
   print("******* 모델이 로딩된 후에 서비스를 시작합니다. *******")
   global model
   global graph
   global sess
   sess = tf.Session()
   set_session(sess)
   graph = tf.get_default_graph()

   model = joblib.load('./model/classifier.sav')

   #서버 실행
   app.run(host="0.0.0.0", port="5555")
