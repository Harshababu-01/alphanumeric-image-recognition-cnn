from src.predict import predict

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def process_and_predict(source_data, mode="digit"):
    """
    Ingests physical filepath OR raw Numpy array directly from the Web Canvas stream.
    Delegates inference to the core predict module.
    """
    prediction, confidence, top_3, error = predict(source_data, mode)
    
    if error:
        return None, error
        
    return {
        "prediction": prediction,
        "confidence": confidence,
        "top_3": top_3
    }, None

