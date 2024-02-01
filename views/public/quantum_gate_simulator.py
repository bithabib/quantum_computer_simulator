from app import app
from flask import render_template
@app.route('/gate')
def gateSimulator():
    return render_template('public/quantum_gate_simulator.html')