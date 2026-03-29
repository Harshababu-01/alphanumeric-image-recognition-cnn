from flask import Flask
from app.routes import main_bp
import os

# Initialize Flask application with explicit folder references
app = Flask(__name__, template_folder='templates', static_folder='static')

# Configuration for File Uploads
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 # 2MB max file size to prevent abuse

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Register application routes
app.register_blueprint(main_bp)

if __name__ == '__main__':
    # Start the production-style development server
    app.run(debug=True, host='0.0.0.0', port=5000)
