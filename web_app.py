
import pickle
from flask import *
import Emojifier

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        mem=request.form['mem']
        t=Emojifier.emo(mem)
        return render_template("emo.html",t=t)
    return render_template("emo.html")    

if __name__ == '__main__':
    app.run(debug=True)
