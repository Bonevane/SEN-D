import React, { useState } from "react";
import { FiLayers, FiEye, FiDownload, FiClock } from "react-icons/fi";
import AnimatedCard from "./AnimatedCard";
import { downloadImage } from "../utils/imageUtils";

const ResultCard = ({
  title,
  prediction,
  confidence,
  probabilities,
  gradcam,
  overlay,
  processingTime,
}) => {
  const isPositive = prediction?.toLowerCase().includes("stone");

  return (
    <div className="bg-white rounded-xl shadow-lg overflow-hidden">
      <div className="p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xl font-semibold text-[#212121]">{title}</h3>
          {processingTime && (
            <div className="flex items-center space-x-1 text-gray-500 text-sm">
              <FiClock size={14} />
              <span>{processingTime.toFixed(1)}s</span>
            </div>
          )}
        </div>

        <div className="mb-6">
          <div
            className={`inline-flex items-center px-4 py-2 rounded-full text-sm font-medium ${
              isPositive
                ? "bg-red-100 text-red-800"
                : "bg-green-100 text-green-800"
            }`}
          >
            {prediction}
          </div>

          <div className="mt-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-gray-700">
                Confidence
              </span>
              <span className="text-sm font-bold text-[#212121]">
                {confidence}%
              </span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-[#e91e4d] h-2 rounded-full transition-all duration-500"
                style={{ width: `${confidence}%` }}
              ></div>
            </div>
          </div>
        </div>

        {probabilities && (
          <div className="mb-6">
            <h4 className="text-sm font-medium text-gray-700 mb-3">
              Probability Breakdown
            </h4>
            <div className="space-y-2">
              {Object.entries(probabilities).map(([key, value]) => (
                <div
                  key={key}
                  className="flex items-center justify-between text-sm"
                >
                  <span className="capitalize">{key.replace("_", " ")}</span>
                  <span className="font-medium">
                    {(value * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {(gradcam || overlay) && (
          <div className="space-y-4">
            <h4 className="text-sm font-medium text-gray-700 flex items-center space-x-2">
              <FiEye size={16} />
              <span>Grad-CAM Visualization</span>
            </h4>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {overlay && (
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-medium text-gray-600">
                      Overlay
                    </span>
                    <button
                      onClick={() =>
                        downloadImage(
                          overlay,
                          `${title.toLowerCase()}_overlay.png`
                        )
                      }
                      className="p-1 text-gray-400 hover:text-[#e91e4d] transition-colors"
                    >
                      <FiDownload size={14} />
                    </button>
                  </div>
                  <img
                    src={`data:image/png;base64,${overlay}`}
                    alt="Grad-CAM Overlay"
                    className="w-full h-32 object-contain bg-gray-50 rounded-lg"
                  />
                </div>
              )}

              {gradcam && (
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-medium text-gray-600">
                      Heatmap
                    </span>
                    <button
                      onClick={() =>
                        downloadImage(
                          gradcam,
                          `${title.toLowerCase()}_heatmap.png`
                        )
                      }
                      className="p-1 text-gray-400 hover:text-[#e91e4d] transition-colors"
                    >
                      <FiDownload size={14} />
                    </button>
                  </div>
                  <img
                    src={`data:image/png;base64,${gradcam}`}
                    alt="Grad-CAM Heatmap"
                    className="w-full h-32 object-contain bg-gray-50 rounded-lg"
                  />
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

const ResultTabs = ({ results }) => {
  const [activeTab, setActiveTab] = useState("ensemble");

  const tabs = [
    {
      id: "ensemble",
      label: "Ensemble Results",
      icon: FiLayers,
      data: results.ensemble,
    },
    {
      id: "inception_v3",
      label: "InceptionV3",
      icon: FiEye,
      data: results.individual_models?.inception_v3,
    },
    {
      id: "inception_resnet_v2",
      label: "InceptionResNetV2",
      icon: FiEye,
      data: results.individual_models?.inception_resnet_v2,
    },
    {
      id: "xception",
      label: "Xception",
      icon: FiEye,
      data: results.individual_models?.xception,
    },
  ];

  return (
    <div className="w-full max-w-6xl mx-auto">
      {/* Tab Navigation */}
      <div className="flex flex-wrap border-b border-gray-200 mb-6">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center space-x-2 px-4 py-3 font-medium text-sm transition-colors border-b-2 cursor-pointer ${
                activeTab === tab.id
                  ? "text-[#e91e4d] border-[#e91e4d]"
                  : "text-gray-500 border-transparent hover:text-gray-700"
              }`}
            >
              <Icon size={16} />
              <span>{tab.label}</span>
            </button>
          );
        })}
      </div>

      {/* Tab Content */}
      <div className="min-h-[400px]">
        {tabs.map((tab) => (
          <AnimatedCard
            key={tab.id}
            className={`${activeTab === tab.id ? "block" : "hidden"}`}
            animation="fadeIn"
            duration={400}
          >
            {tab.data && (
              <ResultCard
                title={tab.label}
                prediction={tab.data.prediction}
                confidence={Math.round(tab.data.confidence * 100)}
                probabilities={tab.data.probabilities}
                gradcam={tab.data.gradcam_heatmap}
                overlay={tab.data.gradcam_overlay}
                processingTime={
                  tab.id === "ensemble" ? results.processing_time : null
                }
              />
            )}
          </AnimatedCard>
        ))}
      </div>
    </div>
  );
};

export default ResultTabs;
