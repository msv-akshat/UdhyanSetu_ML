import { useState, useRef } from "react";
import { Container, Row, Col, Card, Button, Form, Spinner, Badge, ProgressBar } from "react-bootstrap";
import { CloudArrowUpFill, ArrowRight, XCircleFill, CheckCircleFill } from 'react-bootstrap-icons';
import './style.css';

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:5000";

const CLASS_NAMES = [
  "Healthy",
  "Bacterial Leaf Spot",
  "Early Blight",
  "Late Blight",
  "Leaf Mold",
  "Septoria Leaf Spot",
  "Yellow Leaf Curl Virus",
  "Powdery Mildew",
  "Downy Mildew"
];

const LeafDiseaseDetector = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [top3, setTop3] = useState([]);
  const [loading, setLoading] = useState(false);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPrediction(null);
      setTop3([]);
      const reader = new FileReader();
      reader.onloadend = () => setPreview(reader.result);
      reader.readAsDataURL(file);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) {
      setSelectedFile(file);
      setPrediction(null);
      setTop3([]);
      const reader = new FileReader();
      reader.onloadend = () => setPreview(reader.result);
      reader.readAsDataURL(file);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append("image", selectedFile);

    setLoading(true);
    setPrediction(null);
    setTop3([]);

    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      if (res.ok) {
        setPrediction(data.prediction);
        setTop3(data.top_3);
      } else {
        alert(data.error || "Something went wrong!");
      }
    } catch (err) {
      console.error(err);
      alert("Error connecting to backend");
    } finally {
      setLoading(false);
    }
  };

  const removeFile = () => {
    setSelectedFile(null);
    setPreview(null);
    setPrediction(null);
    setTop3([]);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  return (
    <div className="main-container">
      <Card className="app-card shadow-lg">
        <Card.Body className="p-5">
          <h1 className="text-center mb-2 fw-bold text-success">UdhyanSetu 🌿</h1>
          <p className="text-center mb-5 text-muted">
            Upload a leafy vegetable image to get a quick diagnosis!
          </p>

          <Form onSubmit={handleSubmit}>
            <div
              className="file-drop-area mb-4"
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current.click()}
            >
              <input
                type="file"
                ref={fileInputRef}
                accept="image/*"
                onChange={handleFileChange}
                className="d-none"
              />
              <div className="file-drop-content">
                {preview ? (
                  <div className="preview-container">
                    <img src={preview} alt="Leaf Preview" className="preview-image" />
                    <Button
                      variant="light"
                      className="remove-btn"
                      onClick={(e) => { e.stopPropagation(); removeFile(); }}
                    >
                      <XCircleFill size={20} />
                    </Button>
                  </div>
                ) : (
                  <>
                    <CloudArrowUpFill size={50} className="mb-3 text-muted" />
                    <p className="mb-0 text-muted">
                      Drag & drop an image here or <span className="browse-link">browse</span>
                    </p>
                  </>
                )}
              </div>
            </div>

            <Button
              variant="primary"
              type="submit"
              className="w-100 fw-bold submit-btn"
              disabled={loading || !selectedFile}
            >
              {loading ? (
                <Spinner animation="border" size="sm" role="status" />
              ) : (
                <>
                  Predict Disease <ArrowRight size={18} className="ms-2" />
                </>
              )}
            </Button>
          </Form>

          {prediction && (
            <div className="prediction-results mt-5 p-4 rounded shadow-sm">
              <div className="d-flex align-items-center mb-3">
                <h4 className="mb-0 text-primary">Prediction: <span className="text-secondary">{prediction}</span></h4>
              </div>

              {top3.length > 0 && (
                <div className="top-results mt-3">
                  <h6 className="text-muted mb-2">Top 3 Probabilities:</h6>
                  {top3.map(([cls, score], idx) => (
                    <div className="d-flex align-items-center mb-2" key={idx}>
                      <span className="me-2 fw-bold text-muted" style={{ width: '150px' }}>{cls}</span>
                      <ProgressBar
                        now={score * 100}
                        label={`${(score * 100).toFixed(2)}%`}
                        variant={idx === 0 ? "success" : "info"}
                        className="flex-grow-1"
                        style={{ height: '20px' }}
                      />
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </Card.Body>
      </Card>
    </div>
  );
};

export default LeafDiseaseDetector;