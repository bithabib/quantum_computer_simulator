from flask import Flask, request, redirect
app = Flask(__name__)
app.secret_key = "super secret key"

# Import the views public and private
# import public views
from views.public import home
from views.public import book
from views.public import quantum_programming
from views.public import quantum_gate_simulator
from views.public import quantum_historical_experiment_simulator
# import api 
from views.public.api import interpreter

# import private views
from views import login_required_function
from views import token
from views.admin import login
# app.run()
@app.before_request
def redirect_to_domain():
    if request.host.startswith('3.89.197.36') and request.host.endswith(':5000'):
        return redirect('https://quantum.qbithabib.com', code=301)
