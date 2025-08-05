import React, { useState } from "react";
import { FiActivity, FiAlertCircle, FiCheck } from "react-icons/fi";
import AnimatedSection from "./AnimatedSection";
import AnimatedCard from "./AnimatedCard";
import ImageUpload from "./ImageUpload";
import LoadingSpinner from "./LoadingSpinner";
import ResultTabs from "./ResultTabs";
import { predictImage } from "../utils/api";

const PredictionSection = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState("");

  const handleImageSelect = (file) => {
    setSelectedImage(file);
    setResults(null);
    setError("");
  };

  const handleClearImage = () => {
    setSelectedImage(null);
    setResults(null);
    setError("");
  };

  const handlePredict = async () => {
    if (!selectedImage) return;

    setIsLoading(true);
    setError("");

    try {
      const result = await predictImage(selectedImage);
      setResults(result);
      console.log("Prediction results:", result);
    } catch (err) {
      setError(
        "Failed to analyze image. Please ensure the backend server is running on localhost:8000 and try again."
      );
      console.error("Prediction error:", err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <section id="prediction" className="py-20 bg-[#FAFAFA]">
      <div className="container mx-auto px-4">
        <AnimatedSection className="text-center mb-12">
          <div className="inline-flex items-center space-x-2 bg-[#FCE4EC] px-4 py-2 rounded-full mb-4">
            <FiActivity className="text-[#e91e4d]" />
            <span className="text-sm font-medium text-[#ad1442]">
              AI Analysis
            </span>
          </div>
          <h2 className="text-4xl md:text-5xl font-bold text-[#212121] mb-4">
            Kidney Stone Detection
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Upload a medical CT scan image to receive instant analysis using our
            advanced ensemble AI model
          </p>
        </AnimatedSection>

        <div className="max-w-4xl mx-auto space-y-8">
          {/* Image Upload */}
          <AnimatedCard delay={200} animation="scaleIn">
            <ImageUpload
              onImageSelect={handleImageSelect}
              selectedImage={selectedImage}
              onClearImage={handleClearImage}
              isLoading={isLoading}
            />
          </AnimatedCard>

          {/* Predict Button */}
          {selectedImage && !isLoading && !results && (
            <AnimatedCard
              className="text-center"
              animation="fadeIn"
              delay={100}
            >
              <button
                onClick={handlePredict}
                disabled={isLoading}
                className="px-8 py-4 bg-[#e91e4d] text-white font-semibold rounded-full hover:bg-[#ad1442] transition-all duration-300 transform hover:scale-105 shadow-lg disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
              >
                Analyze Image
              </button>
            </AnimatedCard>
          )}

          {/* Loading State */}
          {isLoading && (
            <AnimatedCard
              className="bg-white rounded-2xl shadow-lg"
              animation="fadeIn"
            >
              <LoadingSpinner message="Analyzing medical image..." />
            </AnimatedCard>
          )}

          {/* Error State */}
          {error && (
            <AnimatedCard
              className="bg-red-50 border border-red-200 rounded-2xl p-6"
              animation="slideUp"
            >
              <div className="flex items-center space-x-3">
                <FiAlertCircle className="text-red-500 text-xl flex-shrink-0" />
                <div>
                  <h3 className="text-red-800 font-medium mb-1">
                    Analysis Failed
                  </h3>
                  <p className="text-red-600 text-sm">{error}</p>
                </div>
              </div>
            </AnimatedCard>
          )}

          {/* Results */}
          {results && !isLoading && (
            <AnimatedSection
              className="space-y-6"
              animation="slideUp"
              delay={200}
            >
              <AnimatedCard className="text-center pt-2" delay={0}>
                <div className="flex items-center justify-center mb-4">
                  <div className="w-8 h-8 bg-green-100 rounded-full flex items-center justify-center mr-3">
                    <FiCheck className="text-green-600" />
                  </div>
                  <h3 className="text-2xl font-semibold text-[#212121]">
                    Analysis Complete
                  </h3>
                </div>
                <p className="text-gray-600">
                  Review the results from our ensemble AI model and individual
                  model predictions
                </p>
              </AnimatedCard>
              <AnimatedCard delay={300} animation="scaleIn">
                <ResultTabs results={results} />
              </AnimatedCard>
            </AnimatedSection>
          )}
        </div>
      </div>
    </section>
  );
};

export default PredictionSection;
