import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './ImageUploader.css';

const API_URL = 'http://localhost:5000';

const ImageUploader = () => {
    const [file, setFile] = useState(null);
    const [fileInfo, setFileInfo] = useState(null);
    const [originalImage, setOriginalImage] = useState('');
    const [denoisedImage, setDenoisedImage] = useState('');
    const [loading, setLoading] = useState({
        status: false,
        message: '',
        progress: 0
    });
    const [error, setError] = useState('');
    const [serverStatus, setServerStatus] = useState('checking');
    const [serverDetails, setServerDetails] = useState(null);
    const [noiseSettings, setNoiseSettings] = useState({
        applyNoise: false,
        noiseType: 'gaussian',
        noiseLevel: 0.1
    });
    const [comparisonView, setComparisonView] = useState(false);

    // Check server status periodically
    useEffect(() => {
        checkServerStatus();
        
        const interval = setInterval(() => {
            checkServerStatus();
        }, 30000);
        
        return () => clearInterval(interval);
    }, []);

    const checkServerStatus = async () => {
        try {
            const response = await axios.get(`${API_URL}/status`, {
                timeout: 5000
            });
            if (response.data.status === 'online') {
                setServerStatus('online');
                setServerDetails(response.data);
            } else {
                setServerStatus('offline');
            }
        } catch (error) {
            console.error('Server status check failed:', error);
            setServerStatus('offline');
        }
    };

    const handleFileChange = (event) => {
        const selectedFile = event.target.files[0];
        if (!selectedFile) return;

        // Validate file type
        const validTypes = ['image/jpeg', 'image/png', 'image/jpg'];
        if (!validTypes.includes(selectedFile.type)) {
            setError('Please select a valid image file (JPEG or PNG)');
            return;
        }

        // Validate file size (max 10MB)
        if (selectedFile.size > 10 * 1024 * 1024) {
            setError('File size too large (max 10MB)');
            return;
        }

        setFile(selectedFile);
        setFileInfo({
            name: selectedFile.name,
            size: (selectedFile.size / (1024 * 1024)).toFixed(2) + ' MB',
            type: selectedFile.type,
            dimensions: 'Loading...'
        });
        setError('');
        setDenoisedImage('');

        // Create preview and get dimensions
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                setFileInfo(prev => ({
                    ...prev,
                    dimensions: `${img.width} × ${img.height} px`
                }));
                setOriginalImage(e.target.result);
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(selectedFile);
    };

    const handleNoiseSettingsChange = (event) => {
        const { name, value, type, checked } = event.target;
        setNoiseSettings(prev => ({
            ...prev,
            [name]: type === 'checkbox' ? checked : value
        }));
    };

    const handleDragOver = (e) => {
        e.preventDefault();
        e.stopPropagation();
    };

    const handleDrop = (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            const fileInput = document.createElement('input');
            fileInput.type = 'file';
            fileInput.files = e.dataTransfer.files;
            const event = { target: fileInput };
            handleFileChange(event);
        }
    };

    const handleUpload = async () => {
        if (!file) {
            setError('Please select a file first');
            return;
        }

        setLoading({
            status: true,
            message: 'Uploading image...',
            progress: 0
        });
        setError('');
        setDenoisedImage('');
        
        try {
            const formData = new FormData();
            formData.append('file', file);
            
            if (noiseSettings.applyNoise) {
                formData.append('noise_type', noiseSettings.noiseType);
                formData.append('noise_level', noiseSettings.noiseLevel);
            }

            const response = await axios.post(`${API_URL}/upload`, formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
                timeout: 60000,
                onUploadProgress: (progressEvent) => {
                    const progress = Math.round(
                        (progressEvent.loaded * 100) / progressEvent.total
                    );
                    setLoading(prev => ({
                        ...prev,
                        progress,
                        message: progress < 100 ? 'Uploading image...' : 'Processing image...'
                    }));
                }
            });
            
            setDenoisedImage(`${API_URL}${response.data.denoised_image_path}`);
        } catch (error) {
            let errorMessage = 'An error occurred while processing the image.';
            
            if (error.code === 'ECONNABORTED') {
                errorMessage = 'Request timed out. Please try again with a smaller image.';
            } else if (error.response) {
                errorMessage = error.response.data.error || 
                             `Server error (${error.response.status})`;
            } else if (error.request) {
                errorMessage = 'No response from server. Please check your connection.';
            }
            
            setError(errorMessage);
        } finally {
            setLoading({ status: false, message: '', progress: 0 });
        }
    };

    return (
        <div className="image-uploader-container">
            <h1>Image Denoising Tool</h1>
            
            <div className={`server-status ${serverStatus}`}>
                Server Status: {serverStatus === 'online' ? '🟢 Online' : 
                              serverStatus === 'checking' ? '🟡 Checking...' : '🔴 Offline'}
                {serverStatus === 'online' && serverDetails?.model_loaded && ' (Model Loaded)'}
                {serverStatus === 'online' && !serverDetails?.model_loaded && ' (No Model)'}
                <button onClick={checkServerStatus} className="refresh-btn">⟳</button>
            </div>
            
            {serverStatus === 'offline' && (
                <div className="error-message">
                    <p>Server appears to be offline. Make sure the Flask backend is running on port 5000.</p>
                    <div className="setup-instructions">
                        <h3>Setup Instructions:</h3>
                        <ol>
                            <li>Make sure you have Python installed</li>
                            <li>Install required packages: <code>pip install flask flask-cors tensorflow opencv-python numpy</code></li>
                            <li>Run the Flask server: <code>python app.py</code></li>
                        </ol>
                    </div>
                </div>
            )}
            
            {serverStatus === 'online' && (
                <>
                    <div className="upload-section">
                        <div 
                            className="dropzone"
                            onDragOver={handleDragOver}
                            onDrop={handleDrop}
                            onClick={() => document.querySelector('.file-input').click()}
                        >
                            <input 
                                type="file" 
                                accept=".jpg,.jpeg,.png" 
                                onChange={handleFileChange} 
                                className="file-input"
                                hidden
                            />
                            {!file ? (
                                <>
                                    <div className="dropzone-icon">📁</div>
                                    <p>Drag & drop an image here or click to browse</p>
                                    <p className="file-requirements">Supports: JPEG, PNG (Max 10MB)</p>
                                </>
                            ) : (
                                <div className="file-preview">
                                    <div className="file-info">
                                        <strong>{fileInfo?.name}</strong>
                                        <div>Size: {fileInfo?.size}</div>
                                        <div>Dimensions: {fileInfo?.dimensions}</div>
                                    </div>
                                    <button 
                                        className="change-file-btn"
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            setFile(null);
                                            setOriginalImage('');
                                        }}
                                    >
                                        Change File
                                    </button>
                                </div>
                            )}
                        </div>
                        
                        <div className="noise-settings">
                            <h3>Noise Settings (For Testing)</h3>
                            <label className="checkbox-label">
                                <input 
                                    type="checkbox" 
                                    name="applyNoise" 
                                    checked={noiseSettings.applyNoise}
                                    onChange={handleNoiseSettingsChange}
                                />
                                Apply artificial noise
                            </label>
                            
                            {noiseSettings.applyNoise && (
                                <>
                                    <div className="setting-group">
                                        <label>Noise Type:</label>
                                        <select 
                                            name="noiseType" 
                                            value={noiseSettings.noiseType}
                                            onChange={handleNoiseSettingsChange}
                                        >
                                            <option value="gaussian">Gaussian</option>
                                            <option value="poisson">Poisson</option>
                                            <option value="salt_pepper">Salt & Pepper</option>
                                        </select>
                                    </div>
                                    
                                    <div className="setting-group">
                                        <label>
                                            Noise Level: {parseFloat(noiseSettings.noiseLevel).toFixed(2)}
                                        </label>
                                        <input 
                                            type="range" 
                                            name="noiseLevel" 
                                            min="0.05" 
                                            max="0.3" 
                                            step="0.05"
                                            value={noiseSettings.noiseLevel}
                                            onChange={handleNoiseSettingsChange}
                                        />
                                    </div>
                                </>
                            )}
                        </div>
                        
                        <button 
                            onClick={handleUpload} 
                            disabled={loading.status || !file}
                            className="upload-button"
                        >
                            {loading.status ? (
                                <>
                                    <span className="spinner"></span>
                                    {loading.message} {loading.progress}%
                                </>
                            ) : (
                                'Upload and Denoise'
                            )}
                        </button>
                    </div>
                    
                    {error && (
                        <div className="error-message">
                            <h3>Error</h3>
                            <p>{error}</p>
                        </div>
                    )}
                    
                    {(originalImage || denoisedImage) && (
                        <div className="results-container">
                            {originalImage && (
                                <div className="image-container">
                                    <h3>Original Image</h3>
                                    <img 
                                        src={originalImage} 
                                        alt="Original" 
                                        className={`preview-image ${comparisonView ? 'comparison-view' : ''}`}
                                    />
                                    {fileInfo && (
                                        <div className="image-meta">
                                            <div>Name: {fileInfo.name}</div>
                                            <div>Size: {fileInfo.size}</div>
                                            <div>Dimensions: {fileInfo.dimensions}</div>
                                        </div>
                                    )}
                                </div>
                            )}
                            
                            {denoisedImage && (
                                <div className="image-container">
                                    <div className="denoised-header">
                                        <h3>Denoised Image</h3>
                                        <label className="comparison-toggle">
                                            <input 
                                                type="checkbox" 
                                                checked={comparisonView}
                                                onChange={() => setComparisonView(!comparisonView)}
                                            />
                                            Comparison Slider
                                        </label>
                                    </div>
                                    <div className={`image-comparison ${comparisonView ? 'active' : ''}`}>
                                        {comparisonView && originalImage && (
                                            <img 
                                                src={originalImage} 
                                                alt="Original" 
                                                className="comparison-original"
                                            />
                                        )}
                                        <img 
                                            src={denoisedImage} 
                                            alt="Denoised" 
                                            className={`preview-image ${comparisonView ? 'comparison-view' : ''}`}
                                        />
                                    </div>
                                    <a 
                                        href={denoisedImage} 
                                        download={`denoised_${fileInfo?.name || 'image'}`} 
                                        className="download-link"
                                    >
                                        Download Denoised Image
                                    </a>
                                </div>
                            )}
                        </div>
                    )}
                </>
            )}
        </div>
    );
};

export default ImageUploader;