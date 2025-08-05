export const validateImageFile = (file) => {
  const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
  const maxSize = 10 * 1024 * 1024; // 10MB
  
  if (!validTypes.includes(file.type)) {
    return { valid: false, error: 'Please upload a JPG, JPEG, or PNG image.' };
  }
  
  if (file.size > maxSize) {
    return { valid: false, error: 'Image size must be less than 10MB.' };
  }
  
  return { valid: true };
};

export const createImagePreview = (file) => {
  return URL.createObjectURL(file);
};

export const downloadImage = (base64Data, filename) => {
  const link = document.createElement('a');
  link.href = `data:image/png;base64,${base64Data}`;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};