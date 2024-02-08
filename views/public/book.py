from app import app
from flask import render_template
@app.route('/book')
def book():
    return render_template('public/book.html')