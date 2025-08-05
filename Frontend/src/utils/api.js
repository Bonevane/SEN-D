const API_BASE_URL = 'http://localhost:8000';

export const healthCheck = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    return response.ok;
  } catch (error) {
    console.error('Health check failed:', error);
    return false;
  }
};

export const predictImage = async (imageFile) => {
  const formData = new FormData();
  formData.append('file', imageFile);
  
  try {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Prediction failed:', error);
    throw error;
  }
};