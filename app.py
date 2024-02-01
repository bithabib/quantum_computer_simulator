from flask import Flask
app = Flask(__name__)
app.secret_key = "super secret key"

# Import the views public and private
# import public views
from views.public import quantum_gate_simulator 
app.run()