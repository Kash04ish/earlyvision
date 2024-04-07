from urllib.parse import quote_plus
from flask import Flask, render_template, url_for, request, session, redirect
from flask_pymongo import PyMongo
import bcrypt

app = Flask(__name__)

username = "username"
password = "your_password"

encoded_username = quote_plus(username)
encoded_password = quote_plus(password)

app.config['MONGO_DBNAME'] = 'username'
app.config['MONGO_URI'] = f"mongodb+srv://{encoded_username}:{encoded_password}@cluster0.1pnybnk.mongodb.net/"
mongo = PyMongo(app)

@app.route('/')
def user():
    if 'username' in session:
        return 'You are logged in as the following user:' + session['username']
    return render_template('user.html')

@app.route('/login', methods=['POST'])
def login():
    users = mongo.db.users
    login_user = users.find_one({'name': request.form['username']})
    if login_user:
        if bcrypt.checkpw(request.form['pass'].encode('utf-8'), login_user['password']):
            session['username'] = request.form['username']
            return redirect(url_for('user'))
    return 'Invalid username or password combination'

@app.route('/register', methods=['POST', 'GET'])
def register():
    if request.method == 'POST':
        users = mongo.db.users
        existing_user = users.find_one({'name': request.form['username']})
        if existing_user is None:
            hashpass = bcrypt.hashpw(request.form['pass'].encode('utf-8'), bcrypt.gensalt())
            users.insert({'name': request.form['username'], 'password': hashpass})
            session['username'] = request.form['username']
            return redirect(url_for('user'))
        return "Username already in database"

    return render_template('register.html')

if __name__ == '_main_':
    app.secret_key = 'secretivekeyagain'
    app.run(debug=True)