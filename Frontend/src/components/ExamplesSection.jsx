import React from "react";
import { FiCheck, FiX, FiInfo } from "react-icons/fi";
import AnimatedSection from "./AnimatedSection";
import AnimatedCard from "./AnimatedCard";

const ExampleCard = ({ title, description, isGood, tips }) => (
  <div className="bg-white rounded-xl shadow-lg overflow-hidden hover:shadow-xl transition-all duration-300 hover:scale-102">
    <div className="h-48 bg-gradient-to-br from-gray-100 to-gray-200 flex items-center justify-center">
      <div className="text-center">
        <div
          className={`w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-3 ${
            isGood ? "bg-green-100" : "bg-red-100"
          }`}
        >
          {isGood ? (
            <FiCheck className="text-2xl text-green-600" />
          ) : (
            <FiX className="text-2xl text-red-600" />
          )}
        </div>
        <p className="text-gray-600 font-medium">{title}</p>
      </div>
    </div>

    <div className="p-6">
      <div
        className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-medium mb-3 ${
          isGood ? "bg-green-100 text-green-800" : "bg-red-100 text-red-800"
        }`}
      >
        {isGood ? "Good Example" : "Poor Example"}
      </div>

      <p className="text-gray-600 text-sm mb-4">{description}</p>

      <div className="space-y-2">
        {tips.map((tip, index) => (
          <div key={index} className="flex items-start space-x-2 text-sm">
            <FiInfo className="text-[#e91e4d] mt-0.5 flex-shrink-0" size={12} />
            <span className="text-gray-600">{tip}</span>
          </div>
        ))}
      </div>
    </div>
  </div>
);

const ExamplesSection = () => {
  const goodExamples = [
    {
      title: "Clear CT Scan",
      description:
        "High-resolution abdominal CT scan with clear kidney structures visible",
      tips: [
        "Sharp image quality with good contrast",
        "Kidney regions clearly visible",
        "Appropriate medical imaging format",
      ],
    },
    {
      title: "Proper Orientation",
      description:
        "Correctly oriented medical image showing kidney cross-section",
      tips: [
        "Standard radiological orientation",
        "Clear anatomical structures",
        "Sufficient image resolution (299x299 min)",
      ],
    },
  ];

  const poorExamples = [
    {
      title: "Blurry Image",
      description: "Low-quality or motion-blurred medical images",
      tips: [
        "Image quality too poor for analysis",
        "Motion artifacts present",
        "Insufficient detail for AI processing",
      ],
    },
    {
      title: "Wrong Image Type",
      description: "Non-medical images or incorrect body regions",
      tips: [
        "Not a medical CT scan",
        "Wrong anatomical region",
        "Incompatible file format or size",
      ],
    },
  ];

  return (
    <section className="py-20 bg-white">
      <div className="container mx-auto px-4">
        <AnimatedSection className="text-center mb-12">
          <h2 className="text-4xl md:text-5xl font-bold text-[#212121] mb-4">
            Image Examples
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Understanding what types of images work best with our AI system
          </p>
        </AnimatedSection>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 max-w-6xl mx-auto">
          {/* Good Examples */}
          <AnimatedCard animation="slideRight" delay={200}>
            <h3 className="text-2xl font-semibold text-[#212121] mb-6 flex items-center space-x-2">
              <div className="w-8 h-8 bg-green-100 rounded-full flex items-center justify-center">
                <FiCheck className="text-green-600" />
              </div>
              <span>Good Examples</span>
            </h3>
            <div className="space-y-6">
              {goodExamples.map((example, index) => (
                <AnimatedCard
                  key={index}
                  delay={400 + index * 150}
                  animation="slideUp"
                >
                  <ExampleCard {...example} isGood={true} />
                </AnimatedCard>
              ))}
            </div>
          </AnimatedCard>

          {/* Poor Examples */}
          <AnimatedCard animation="slideLeft" delay={300}>
            <h3 className="text-2xl font-semibold text-[#212121] mb-6 flex items-center space-x-2">
              <div className="w-8 h-8 bg-red-100 rounded-full flex items-center justify-center">
                <FiX className="text-red-600" />
              </div>
              <span>Poor Examples</span>
            </h3>
            <div className="space-y-6">
              {poorExamples.map((example, index) => (
                <AnimatedCard
                  key={index}
                  delay={500 + index * 150}
                  animation="slideUp"
                >
                  <ExampleCard {...example} isGood={false} />
                </AnimatedCard>
              ))}
            </div>
          </AnimatedCard>
        </div>

        {/* Technical Requirements */}
        <AnimatedSection
          className="mt-16 bg-[#FCE4EC]/20 rounded-2xl p-8 max-w-4xl mx-auto"
          animation="scaleIn"
          delay={600}
        >
          <h3 className="text-xl font-semibold text-[#212121] mb-4 text-center">
            Technical Requirements
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-center">
            <div>
              <div className="w-12 h-12 bg-[#e91e4d] rounded-full flex items-center justify-center mx-auto mb-3">
                <span className="text-white font-bold">299</span>
              </div>
              <h4 className="font-medium text-[#212121] mb-1">Image Size</h4>
              <p className="text-sm text-gray-600">Minimum 299x299 pixels</p>
            </div>
            <div>
              <div className="w-12 h-12 bg-[#e91e4d] rounded-full flex items-center justify-center mx-auto mb-3">
                <span className="text-white font-bold">RGB</span>
              </div>
              <h4 className="font-medium text-[#212121] mb-1">Color Format</h4>
              <p className="text-sm text-gray-600">RGB color channels</p>
            </div>
            <div>
              <div className="w-12 h-12 bg-[#e91e4d] rounded-full flex items-center justify-center mx-auto mb-3">
                <span className="text-white font-bold">10MB</span>
              </div>
              <h4 className="font-medium text-[#212121] mb-1">File Size</h4>
              <p className="text-sm text-gray-600">Maximum file size</p>
            </div>
          </div>
        </AnimatedSection>
      </div>
    </section>
  );
};

export default ExamplesSection;
