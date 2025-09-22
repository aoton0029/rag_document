from flask import request, jsonify
from functools import wraps
import logging

logger = logging.getLogger(__name__)

def log_request(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Request: {request.method} {request.path}")
        return func(*args, **kwargs)
    return wrapper

def validate_unified_id(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        unified_id = request.args.get('unified_id')
        if not unified_id:
            logger.error("Missing unified_id in request")
            return jsonify({"error": "unified_id is required"}), 400
        return func(*args, **kwargs)
    return wrapper

def handle_exceptions(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            return jsonify({"error": "An internal error occurred"}), 500
    return wrapper

# Example of how to use the middleware
# @app.route('/some_endpoint', methods=['GET'])
# @log_request
# @validate_unified_id
# @handle_exceptions
# def some_endpoint():
#     return jsonify({"message": "Success"})