import json
import logging
import time
import traceback

from flask import Flask, jsonify, request

from app.model_handler import predict, validate_input

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger('api')

app = Flask(__name__)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    data = request.get_json(silent=True)

    if data is None:
        return jsonify({'error': 'Request body must be JSON'}), 400

    version = data.pop('version', None) or request.args.get('version')

    try:
        validate_input(data)
        prediction, probability, used_version = predict(data, version=version)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    return jsonify({
        'prediction': prediction,
        'probability': probability,
        'model_version': used_version,
    }), 200


@app.after_request
def log_request(response):
    entry = {
        'timestamp': time.time(),
        'level': 'info',
        'endpoint': request.path,
        'method': request.method,
        'status': response.status_code,
    }

    if request.path == '/predict' and response.is_json:
        body = response.get_json(silent=True) or {}
        entry['model_version'] = body.get('model_version')
        entry['prediction'] = body.get('prediction')
        entry['probability'] = body.get('probability')

    logger.info(json.dumps(entry))

    return response


@app.errorhandler(Exception)
def log_unhandled(error):
    logger.error(json.dumps({
        'timestamp': time.time(),
        'level': 'error',
        'endpoint': request.path,
        'method': request.method,
        'error_type': type(error).__name__,
        'error_message': str(error),
        'traceback': traceback.format_exc().splitlines()[-5:],
    }))

    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)