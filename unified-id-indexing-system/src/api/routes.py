from flask import Blueprint, request, jsonify
from src.core.unified_id import UnifiedID
from src.indexing.index_registry import IndexRegistry

api = Blueprint('api', __name__)

unified_id_generator = UnifiedID()
index_registry = IndexRegistry()

@api.route('/generate_unified_id', methods=['POST'])
def generate_unified_id():
    new_id = unified_id_generator.generate()
    return jsonify({"unified_id": new_id}), 201

@api.route('/register_index', methods=['POST'])
def register_index():
    data = request.json
    unified_id = data.get('unified_id')
    index_type = data.get('index_type')
    
    if not unified_id or not index_type:
        return jsonify({"error": "unified_id and index_type are required"}), 400
    
    index_registry.register(unified_id, index_type)
    return jsonify({"message": "Index registered successfully"}), 201

@api.route('/index_status/<unified_id>', methods=['GET'])
def get_index_status(unified_id):
    status = index_registry.get_status(unified_id)
    if status is None:
        return jsonify({"error": "Index not found"}), 404
    return jsonify({"unified_id": unified_id, "status": status}), 200