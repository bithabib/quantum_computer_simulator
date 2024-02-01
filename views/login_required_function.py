from functools import wraps
from flask import request, session, render_template, redirect, url_for
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if session.get('jwt_token') is None:
            # return redirect(url_for('login'))
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function