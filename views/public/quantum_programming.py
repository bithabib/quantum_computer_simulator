from app import app
from flask import render_template
@app.route('/programming')
def programming():
    return render_template('public/quantum_programming.html')