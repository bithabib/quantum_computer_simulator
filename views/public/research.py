from app import app
from flask import render_template


@app.route('/research')
@app.route('/reproducibility')
def research():
    return render_template('public/research.html')
