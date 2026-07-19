import os
import secrets

from flask import Flask, request, redirect

app = Flask(__name__)

# Session signing key. Set QUANTUM_SECRET_KEY in the environment in production;
# otherwise a random key is generated at start-up (sessions then reset on every
# restart, which is safe but not persistent). Never hard-code a key here - a
# committed key lets anyone forge session cookies.
app.secret_key = os.environ.get("QUANTUM_SECRET_KEY") or secrets.token_hex(32)

# Import the views public and private
# import public views
from views.public import home
from views.public import book
from views.public import quantum_programming
from views.public import quantum_gate_simulator
from views.public import quantum_historical_experiment_simulator
from views.public import research
# import api
from views.public.api import interpreter  # legacy route, now disabled
from views.public.api import quantum_lab

# import private views
from views import login_required_function
from views import token
from views.admin import login
# app.run()
@app.before_request
def redirect_to_domain():
    if request.host.startswith('3.89.197.36') and request.host.endswith(':5000'):
        return redirect('https://quantum.qbithabib.com', code=301)
