import React, { useState, useRef } from "react";
import { FiUpload, FiX, FiImage } from "react-icons/fi";
import { validateImageFile, createImagePreview } from "../utils/imageUtils";

const ImageUpload = ({
  onImageSelect,
  selectedImage,
  onClearImage,
  isLoading,
}) => {
  const [dragActive, setDragActive] = useState(false);
  const [error, setError] = useState("");
  const fileInputRef = useRef(null);

  const handleFiles = (files) => {
    const file = files[0];
    if (!file) return;

    const validation = validateImageFile(file);
    if (!validation.valid) {
      setError(validation.error);
      return;
    }

    setError("");
    onImageSelect(file);
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFiles(e.dataTransfer.files);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFiles(e.target.files);
    }
  };

  const openFileDialog = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className="w-full max-w-2xl mx-auto">
      {!selectedImage ? (
        <div
          className={`relative border-2 border-dashed rounded-2xl p-8 text-center transition-all duration-300 cursor-pointer
            ${
              dragActive
                ? "border-[#e91e4d] bg-[#FCE4EC]/50"
                : "border-gray-300 hover:border-[#e91e4d] hover:bg-[#FCE4EC]/20"
            }
            ${isLoading && "pointer-events-none opacity-50"}
          `}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={openFileDialog}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/jpeg,image/jpg,image/png"
            onChange={handleChange}
            className="hidden"
            disabled={isLoading}
          />

          <div className="space-y-4">
            <div className="w-20 h-20 mx-auto bg-[#FCE4EC] rounded-full flex items-center justify-center">
              <FiUpload className="text-3xl text-[#e91e4d]" />
            </div>

            <div>
              <h3 className="text-xl font-semibold text-[#212121] mb-2">
                Upload Medical Image
              </h3>
              <p className="text-gray-600 mb-4">
                Drag and drop your CT scan image here, or click to browse
              </p>
              <p className="text-sm text-gray-500">
                Supports JPG, JPEG, PNG (max 10MB)
              </p>
            </div>

            <button
              type="button"
              className="px-6 py-3 bg-[#e91e4d] text-white font-medium rounded-full hover:bg-[#ad1442] transition-colors disabled:opacity-50 cursor-pointer"
              disabled={isLoading}
            >
              Choose File
            </button>
          </div>
        </div>
      ) : (
        <div className="relative bg-white rounded-2xl shadow-lg overflow-hidden">
          <div className="p-4 border-b border-gray-200 flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <FiImage className="text-[#e91e4d]" />
              <span className="font-medium text-[#212121]">Selected Image</span>
            </div>
            {!isLoading && (
              <button
                onClick={onClearImage}
                className="p-2 text-gray-400 hover:text-[#e91e4d] transition-colors cursor-pointer"
              >
                <FiX size={20} />
              </button>
            )}
          </div>

          <div className="p-4">
            <img
              src={createImagePreview(selectedImage)}
              alt="Selected medical image"
              className="w-full h-64 object-contain bg-gray-50 rounded-lg"
            />
            <div className="mt-3 text-sm text-gray-600">
              <span className="font-medium">{selectedImage.name}</span>
              <span className="mx-2">•</span>
              <span>{(selectedImage.size / 1024 / 1024).toFixed(2)} MB</span>
            </div>
          </div>
        </div>
      )}

      {error && (
        <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-red-600 text-sm">{error}</p>
        </div>
      )}
    </div>
  );
};

export default ImageUpload;
