import { useState, useRef } from 'react'
import './App.css'

interface ClassificationResult {
  defect_type: string;
  confidence: number;
}

interface Detections {
  defect_type: string;
  confidence: number;
}

interface DetectionResult {
  detected_defects: Detections[];
  image_base64: string;
}

interface VideoSummary {
  defect_type: string;
  confidence: number;
}

interface VideoResult {
  video_name: string;
  total_frames: number;
  frames_processed: number;
  detected_defects: VideoSummary[];
  overall_status: string;
  video_url: string;
}

function App() {
  const [activeTab, setActiveTab] = useState<'classification' | 'detection' | 'video'>('classification');
  const [hasVideoResult, setHasVideoResult] = useState(false);

  return (
    <div className={`dashboard ${activeTab === 'video' && hasVideoResult ? 'video-mode' : ''}`}>
      <header className="header" style={{ marginBottom: '1rem' }}>
        <h1>MetalGuard AI</h1>
        <p>Precision Cosmetic Defect Detection for Industrial Metals</p>
      </header>

      <nav className="navbar">
        <div 
          className={`nav-item ${activeTab === 'classification' ? 'active' : ''}`}
          onClick={() => setActiveTab('classification')}
        >
          Surface Classification
        </div>
        <div 
          className={`nav-item ${activeTab === 'detection' ? 'active' : ''}`}
          onClick={() => setActiveTab('detection')}
        >
          Object Detection
        </div>
        <div 
          className={`nav-item ${activeTab === 'video' ? 'active' : ''}`}
          onClick={() => setActiveTab('video')}
        >
          Video Analysis
        </div>
      </nav>

      {activeTab === 'classification' && <ClassificationView />}
      {activeTab === 'detection' && <DetectionView />}
      {activeTab === 'video' && <VideoDetectionView onResultChange={setHasVideoResult} />}
    </div>
  )
}

function ClassificationView() {
  const [image, setImage] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ClassificationResult | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setImage(file);
      setPreview(URL.createObjectURL(file));
      setResult(null);
    }
  };

  const handleUpload = async () => {
    if (!image) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('file', image);

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (data.status === 'success') {
        setResult({
          defect_type: data.defect_type,
          confidence: data.confidence
        });
      } else {
        alert("Error: " + data.message);
      }
    } catch (error) {
      console.error("Upload failed", error);
      alert("Could not connect to the backend. Is FastAPI running?");
    } finally {
      setLoading(false);
    }
  };

  const isDefective = result ? result.defect_type.toLowerCase() !== 'scratches' && result.defect_type.toLowerCase() !== 'normal' : false;

  return (
    <>
      <main className="glass upload-area" onClick={() => fileInputRef.current?.click()}>
        <input
          type="file"
          ref={fileInputRef}
          style={{ display: 'none' }}
          accept="image/*"
          onChange={handleImageChange}
        />

        {preview ? (
          <div className="preview-container">
            <img src={preview} alt="Metal Surface Preview" className="preview-img" />
          </div>
        ) : (
          <>
            <div className="upload-icon">󰄵</div>
            <h2>Upload Metal Image</h2>
            <p>Drag and drop or click to select a sample image for Classification</p>
          </>
        )}
      </main>

      <aside className="glass results-card">
        <div className="status-header">
          <div className={`status-badge ${!result ? 'status-detecting' : (result.defect_type.toLowerCase() === 'normal' ? 'status-clear' : 'status-defective')}`}>
            {!result ? 'READY TO SCAN' : (result.defect_type.toLowerCase() === 'normal' ? 'NO DEFECTS' : 'DEFECT DETECTED')}
          </div>
        </div>

        <div className="metric">
          <span className="metric-label">Analysis Result</span>
          <span className="metric-value">{result ? result.defect_type.toUpperCase() : '---'}</span>
        </div>

        <div className="metric">
          <span className="metric-label">Confidence Score</span>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <span className="metric-value">{result ? `${result.confidence}%` : '0%'}</span>
          </div>
          <div className="confidence-bg">
            <div
              className="confidence-fill"
              style={{
                width: result ? `${result.confidence}%` : '0%',
                background: isDefective ? 'var(--danger)' : 'linear-gradient(to right, var(--accent-primary), var(--accent-secondary))'
              }}
            ></div>
          </div>
        </div>

        <div className="metric">
          <span className="metric-label">System Recommendation</span>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', margin: '0' }}>
            {!result
              ? 'Please upload an image to begin automated surface classification.'
              : result.defect_type.toLowerCase() === 'normal'
                ? 'Surface integrity verified. Component is safe for next assembly stage.'
                : `Action Required: Surface shows ${result.defect_type}. Recommend secondary manual inspection.`}
          </p>
        </div>

        <button
          className="primary"
          onClick={(e) => { e.stopPropagation(); handleUpload(); }}
          disabled={!image || loading}
        >
          {loading ? <span className="loader"></span> : 'RUN CLASSIFICATION SCAN'}
        </button>
      </aside>
    </>
  )
}

function DetectionView() {
  const [image, setImage] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<DetectionResult | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setImage(file);
      setPreview(URL.createObjectURL(file));
      setResult(null);
    }
  };

  const handleUpload = async () => {
    if (!image) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('file', image);

    try {
      const response = await fetch('http://localhost:8000/predict-detection', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (data.status === 'success') {
        setResult({
          detected_defects: data.detected_defects,
          image_base64: `data:image/jpeg;base64,${data.image_base64}`
        });
        // Override preview with the annotated image
        setPreview(`data:image/jpeg;base64,${data.image_base64}`);
      } else {
        alert("Error: " + data.message);
      }
    } catch (error) {
      console.error("Upload failed", error);
      alert("Could not connect to the backend. Is FastAPI running?");
    } finally {
      setLoading(false);
    }
  };

  const hasDefects = result ? result.detected_defects.length > 0 : false;

  return (
    <>
      <main className="glass upload-area" onClick={() => fileInputRef.current?.click()}>
        <input
          type="file"
          ref={fileInputRef}
          style={{ display: 'none' }}
          accept="image/*"
          onChange={handleImageChange}
        />

        {preview ? (
          <div className="preview-container">
            <img src={preview} alt="Detection Preview" className="preview-img" style={{ maxWidth: '100%', maxHeight: '600px' }} />
          </div>
        ) : (
          <>
            <div className="upload-icon">󰄵</div>
            <h2>Upload Metal Image</h2>
            <p>Drag and drop or click to select a sample image for Object Detection</p>
          </>
        )}
      </main>

      <aside className="glass results-card">
        <div className="status-header">
          <div className={`status-badge ${!result ? 'status-detecting' : (!hasDefects ? 'status-clear' : 'status-defective')}`}>
            {!result ? 'READY TO DETECT' : (!hasDefects ? 'NO DEFECTS DETECTED' : `${result.detected_defects.length} DEFECTS FOUND`)}
          </div>
        </div>

        <div className="metric">
          <span className="metric-label">Detected Objects</span>
          {!result ? (
            <span className="metric-value" style={{ fontSize: '1rem', color: 'var(--text-muted)' }}>Waiting for image...</span>
          ) : !hasDefects ? (
            <span className="metric-value">None</span>
          ) : (
            <div className="defect-list">
              {result.detected_defects.map((defect, idx) => (
                <div key={idx} className="defect-item">
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
                    <span style={{ fontWeight: 600, color: 'white', textTransform: 'capitalize' }}>
                      {defect.defect_type}
                    </span>
                    <span style={{ fontSize: '0.9rem', color: 'var(--accent-primary)' }}>
                      {defect.confidence}%
                    </span>
                  </div>
                  <div className="confidence-bg" style={{ height: '4px' }}>
                    <div
                      className="confidence-fill"
                      style={{
                        width: `${defect.confidence}%`,
                        background: 'var(--danger)' // all defects are danger
                      }}
                    ></div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="metric">
          <span className="metric-label">System Recommendation</span>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', margin: '0' }}>
            {!result
              ? 'Upload an image to identify and locate defects using our YOLO model.'
              : !hasDefects
                ? 'No visible defects detected. Surface passes quality check.'
                : `Action Required: Multiple defects located. Recommend targeted maintenance or disposal.`}
          </p>
        </div>

        <button
          className="primary"
          onClick={(e) => { e.stopPropagation(); handleUpload(); }}
          disabled={!image || loading}
        >
          {loading ? <span className="loader"></span> : 'RUN DETECTION SCAN'}
        </button>
      </aside>
    </>
  )
}

interface VideoDetectionProps {
  onResultChange: (hasResult: boolean) => void;
}

function VideoDetectionView({ onResultChange }: VideoDetectionProps) {
  const [video, setVideo] = useState<File | null>(null);
  const [videoName, setVideoName] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<VideoResult | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleVideoChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setVideo(file);
      setVideoName(file.name);
      setResult(null);
      onResultChange(false);
    }
  };

  const handleUpload = async () => {
    if (!video) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('file', video);

    try {
      const response = await fetch('http://localhost:8000/predict-video-detection', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (data.status === 'success') {
        setResult({
          video_name: data.video_name,
          total_frames: data.total_frames,
          frames_processed: data.frames_processed,
          detected_defects: data.detected_defects,
          overall_status: data.overall_status,
          video_url: data.video_url,
        });
        onResultChange(true);
      } else {
        alert("Error: " + data.message);
      }
    } catch (error) {
      console.error("Upload failed", error);
      alert("Could not connect to the backend. Is FastAPI running?");
    } finally {
      setLoading(false);
    }
  };

  const hasDefects = result ? result.detected_defects.length > 0 : false;

  return (
    <>
      <main className="glass upload-area" onClick={() => fileInputRef.current?.click()}>
        <input
          type="file"
          ref={fileInputRef}
          style={{ display: 'none' }}
          accept="video/mp4,video/x-m4v,video/*"
          onChange={handleVideoChange}
        />

        {videoName ? (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem' }}>
            <div className="upload-icon" style={{ fontSize: '3rem' }}>🎥</div>
            <h2 style={{ color: 'var(--accent-primary)' }}>{videoName}</h2>
            <p>Ready for analysis</p>
          </div>
        ) : (
          <>
            <div className="upload-icon">󰄵</div>
            <h2>Upload Video File</h2>
            <p>Drag and drop or click to select a video for Analysis</p>
          </>
        )}
      </main>

      <aside className="glass results-card">
        <div className="status-header">
          <div className={`status-badge ${!result ? 'status-detecting' : (!hasDefects ? 'status-clear' : 'status-defective')}`}>
            {!result ? 'READY TO ANALYZE' : (!hasDefects ? 'NO DEFECTS DETECTED' : 'DEFECTS FOUND')}
          </div>
        </div>

        <div className="metric">
          <span className="metric-label">Analysis Summary</span>
          {!result ? (
            <span className="metric-value" style={{ fontSize: '1rem', color: 'var(--text-muted)' }}>Waiting for video...</span>
          ) : !hasDefects ? (
            <span className="metric-value">0 Defects</span>
          ) : (
            <div className="defect-list">
              {result.detected_defects.map((defect, idx) => (
                <div key={idx} className="defect-item">
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
                    <span style={{ fontWeight: 600, color: 'white', textTransform: 'capitalize' }}>
                      {defect.defect_type}
                    </span>
                    <span style={{ fontSize: '0.9rem', color: 'var(--accent-primary)' }}>
                      {defect.confidence}%
                    </span>
                  </div>
                  <div className="confidence-bg" style={{ height: '4px' }}>
                    <div
                      className="confidence-fill"
                      style={{
                        width: `${defect.confidence}%`,
                        background: 'var(--danger)' // all defects are danger
                      }}
                    ></div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="metric">
          <span className="metric-label">Processing Stats</span>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', margin: '0' }}>
            {!result
              ? 'Upload a video to begin frame-by-frame analysis.'
              : `Successfully scanned ${result.frames_processed} frames (optimizing by reducing the frame rate 5x) out of ${result.total_frames} total frames.`}
          </p>
        </div>

        <button
          className="primary"
          onClick={(e) => { e.stopPropagation(); handleUpload(); }}
          disabled={!video || loading}
        >
          {loading ? <span className="loader"></span> : 'RUN VIDEO SCAN'}
        </button>
      </aside>

      {result?.video_url && (
        <main className="glass upload-area" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'flex-start', padding: '2rem' }}>
          <h2 style={{ marginBottom: '1rem', color: 'var(--accent-primary)', fontSize: '1.5rem', fontWeight: 'bold' }}>Analyzed Video Playback</h2>
          <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', width: '100%' }}>
            <video 
              src={result.video_url} 
              controls 
              autoPlay 
              loop
              style={{ width: '100%', borderRadius: '12px', backgroundColor: '#000', boxShadow: '0 20px 40px rgba(0,0,0,0.4)', maxHeight: '600px', objectFit: 'contain' }} 
            />
          </div>
        </main>
      )}
    </>
  )
}

export default App
