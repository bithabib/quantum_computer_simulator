from app import app
from flask import Flask, request, render_template_string
import sys
import io
@app.route('/run_python_code', methods=['POST'])
def run_python_code():
    print(request.json['code'])
    code = request.json['code']
    sys.stdout = io.StringIO()  # Redirect stdout to capture output
    try:
        exec(code)
        output = sys.stdout.getvalue()
    except Exception as e:
        output = str(e)
    sys.stdout = sys.__stdout__  # Reset stdout
    return output

