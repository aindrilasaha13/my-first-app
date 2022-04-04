from flask import Flask,render_template,request,send_file
import requests
import matplotlib.pyplot as plt
import io,base64

app = Flask(__name__)

@app.get("/")
def add():
    return render_template("home.html")

# @app.post("/name")
# def name():
#     text_response = requests.get("https://testing-first-flask-app.herokuapp.com//hello")
#     name = request.form.get("name")
#     display_string = text_response.text + " "+name
    
#     img_response = requests.get("https://testing-first-flask-app.herokuapp.com//image")
#     # file = open("static\\downloaded.jpg", "wb")
#     # file.write(img_response.content)
#     # file.close()
#     # img = "/static/downloaded.jpg"

#     plot_url = img_response.content.decode('utf-8')
#     return render_template('new_page.html',display_string=display_string, images={ 'image': plot_url })

@app.get("/hello")
def hello():
    return "Hello" 

@app.post("/name")
def image():
 
    x = [1,2,3,4,5]
    y = [1,4,9,16,25]
    
    plt.title("squares")
    plt.plot(x, y, color='red')

    plt.xlabel('numbers',fontsize=10)
    plt.ylabel("squares",fontsize=10)
    plt.xticks(fontsize = 10)
    plt.yticks(fontsize = 10)

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    plot_url = base64.b64encode(img.getvalue()).decode()
    return render_template('new_page.html',images={ 'image': plot_url })
    
    return plot_url
    
    # img = "static\\plotted_squares.jpg"
    # plt.savefig(img)
    # return send_file(img,as_attachment=True,mimetype='image/jpg') 
  
if __name__ == "__main__":
    app.run(debug=True)