from app import app
from flask import render_template
@app.route('/experiment')
def historicalExperiment():
    return render_template('public/quantum_historical_experiment_simulator.html')