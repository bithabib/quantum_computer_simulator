# This is login python file
from traceback import print_tb
from app import app
from flask import request, session, redirect, url_for, render_template
import csv


@app.route('/api/token', methods=['POST'])
def token():
    # Read CSV file function
    def read_csv_file(file_path):
        with open(file_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            data = list(reader)
        return data

    email = request.form.get('email')
    password = request.form.get('password')

    try:
        # Read CSV file
        data = read_csv_file('data_storage/admin_user.csv')
        for row in data:
            if email == row['email'] and str(password) == str(row['password']):
                session['localId'] = email
                session['user_id'] = row['id']
                session['user_type'] = row['user_type']
                session['jwt_token'] = password
                return redirect(url_for('admin'))
        return redirect(url_for('login'))
    except Exception as e:
        return redirect(url_for('login', next=request.url))


@app.route('/cache/clear')
def clear_cache():
    session.clear()
    return redirect(url_for('admin'))
