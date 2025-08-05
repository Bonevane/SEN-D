import React, { useState } from "react";
import {
  FiClock,
  FiDownload,
  FiEye,
  FiBarChart,
  FiLayers,
} from "react-icons/fi";

const ResultCard = ({
  title,
  prediction,
  confidence,
  probabilities,
  processingTime,
}) => {
  const isPositive = prediction === "Kidney_stone";

  return (
    <div className="bg-white rounded-xl shadow-lg overflow-hidden">
      <div className="p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xl font-semibold text-[#212121]">{title}</h3>
          {processingTime && (
            <div className="flex items-center space-x-1 text-gray-500 text-sm">
              <FiClock size={14} />
              <span>{parseFloat(processingTime).toFixed(2)}s</span>
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
      </div>
    </div>
  );
};

const ComparisonTable = ({ results }) => {
  const downloadImage = (base64Data, filename) => {
    const link = document.createElement("a");
    link.href = `data:image/png;base64,${base64Data}`;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const tableData = [
    {
      model: "Ensemble",
      prediction: results.ensemble.prediction,
      confidence: (results.ensemble.confidence * 100).toFixed(1),
      kidneyStoneProb: (
        results.ensemble.probabilities.Kidney_stone * 100
      ).toFixed(1),
      normalProb: (results.ensemble.probabilities.Normal * 100).toFixed(1),
    },
    ...Object.entries(results.individual_models).map(([modelName, data]) => ({
      model: modelName
        .replace(/_/g, " ")
        .replace(/\b\w/g, (l) => l.toUpperCase()),
      prediction: data.prediction,
      confidence: (data.confidence * 100).toFixed(1),
      kidneyStoneProb: (data.probabilities.Kidney_stone * 100).toFixed(1),
      normalProb: (data.probabilities.Normal * 100).toFixed(1),
    })),
  ];

  return (
    <div className="space-y-6">
      {/* Results Table */}
      <div className="bg-white rounded-xl shadow-lg overflow-hidden">
        <div className="p-6">
          <div className="flex items-center space-x-2 mb-4">
            <FiBarChart className="text-[#e91e4d]" size={20} />
            <h3 className="text-xl font-semibold text-[#212121]">
              Model Comparison Results
            </h3>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-200">
                  <th className="text-left py-3 px-4 font-semibold text-gray-700">
                    Model
                  </th>
                  <th className="text-left py-3 px-4 font-semibold text-gray-700">
                    Prediction
                  </th>
                  <th className="text-left py-3 px-4 font-semibold text-gray-700">
                    Confidence
                  </th>
                  <th className="text-left py-3 px-4 font-semibold text-gray-700">
                    Kidney Stone
                  </th>
                  <th className="text-left py-3 px-4 font-semibold text-gray-700">
                    Normal
                  </th>
                </tr>
              </thead>
              <tbody>
                {tableData.map((row, index) => (
                  <tr
                    key={row.model}
                    className={`border-b border-gray-100 ${
                      index === 0 ? "bg-[#FCE4EC]" : "hover:bg-gray-50"
                    }`}
                  >
                    <td className="py-3 px-4">
                      <div className="flex items-center space-x-2">
                        {index === 0 && (
                          <FiLayers className="text-[#e91e4d]" size={16} />
                        )}
                        <span
                          className={`font-medium ${
                            index === 0 ? "text-[#ad1442]" : "text-gray-900"
                          }`}
                        >
                          {row.model}
                        </span>
                      </div>
                    </td>
                    <td className="py-3 px-4">
                      <span
                        className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
                          row.prediction === "Kidney_stone"
                            ? "bg-red-100 text-red-800"
                            : "bg-green-100 text-green-800"
                        }`}
                      >
                        {row.prediction.replace("_", " ")}
                      </span>
                    </td>
                    <td className="py-3 px-4 font-semibold text-gray-900">
                      {row.confidence}%
                    </td>
                    <td className="py-3 px-4 text-red-600 font-medium">
                      {row.kidneyStoneProb}%
                    </td>
                    <td className="py-3 px-4 text-green-600 font-medium">
                      {row.normalProb}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="mt-4 flex items-center justify-between text-sm text-gray-500">
            <span>
              Processing Time: {parseFloat(results.processing_time).toFixed(2)}s
            </span>
            <span>Models Used: {results.num_models}</span>
          </div>
        </div>
      </div>
    </div>
  );
};

const ResultTabs = ({ results }) => {
  const [activeTab, setActiveTab] = useState("ensemble");

  const tabs = [
    {
      id: "ensemble",
      label: "Ensemble",
      icon: FiLayers,
      data: results.ensemble,
    },
    ...Object.entries(results.individual_models).map(([key, data]) => ({
      id: key,
      label: key.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase()),
      icon: FiBarChart,
      data,
    })),
  ];

  const activeTabData = tabs.find((tab) => tab.id === activeTab);

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-2xl shadow-lg overflow-hidden">
        {/* Tab Navigation */}
        <div className="border-b border-gray-200">
          <nav className="flex space-x-8 px-6">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center space-x-2 py-4 px-1 border-b-2 font-medium text-sm transition-colors cursor-pointer ${
                    activeTab === tab.id
                      ? "border-[#e91e4d] text-[#e91e4d]"
                      : "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300"
                  }`}
                >
                  <Icon size={16} />
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </nav>
        </div>

        {/* Tab Content */}
        <div>
          {activeTab === "ensemble" ? (
            <ComparisonTable results={results} />
          ) : (
            <ResultCard
              title={activeTabData.label}
              prediction={activeTabData.data.prediction}
              confidence={Math.round(activeTabData.data.confidence * 100)}
              probabilities={activeTabData.data.probabilities}
            />
          )}
        </div>
      </div>

      {/* Grad-CAM Visualization Section */}
      <div className="bg-white rounded-xl shadow-lg overflow-hidden">
        <div className="p-6">
          <div className="flex items-center space-x-2 mb-6">
            <FiEye className="text-[#e91e4d]" size={20} />
            <h3 className="text-xl font-semibold text-[#212121]">
              Grad-CAM Visualizations
            </h3>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {Object.entries(results.individual_models).map(
              ([modelName, data]) => (
                <div key={modelName} className="space-y-3">
                  <div className="flex items-center justify-between">
                    <h4 className="font-semibold text-gray-800 capitalize">
                      {modelName.replace(/_/g, " ")}
                    </h4>
                    {data.gradcam_overlay && (
                      <button
                        onClick={() =>
                          downloadImage(
                            data.gradcam_overlay,
                            `${modelName}_gradcam.png`
                          )
                        }
                        className="p-1 text-gray-400 hover:text-[#e91e4d] transition-colors cursor-pointer"
                        title="Download Grad-CAM"
                      >
                        <FiDownload size={16} />
                      </button>
                    )}
                  </div>

                  {data.gradcam_overlay ? (
                    <div className="relative">
                      <img
                        src={`data:image/png;base64,${data.gradcam_overlay}`}
                        alt={`${modelName} Grad-CAM`}
                        className="w-full h-full object-contain bg-gray-50 rounded-lg"
                      />
                      <div className="absolute top-2 left-2">
                        <span
                          className={`inline-flex items-center px-2 py-1 rounded text-xs font-medium ${
                            data.prediction === "Kidney_stone"
                              ? "bg-red-100 text-red-800"
                              : "bg-green-100 text-green-800"
                          }`}
                        >
                          {(data.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  ) : (
                    <div className="w-full h-48 bg-gray-100 rounded-lg flex items-center justify-center text-gray-500">
                      No Grad-CAM available
                    </div>
                  )}
                </div>
              )
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ResultTabs;
