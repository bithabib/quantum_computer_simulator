# This is login python file
from app import app
from flask import render_template
@app.route('/login')
def login():
    return render_template('admin/login.html')