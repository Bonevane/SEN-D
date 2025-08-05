import React from "react";
import {
  FiLayers,
  FiBookOpen,
  FiShield,
  FiTrendingUp,
  FiDatabase,
  FiCpu,
} from "react-icons/fi";
import { MdScience, MdBiotech } from "react-icons/md";
import AnimatedSection from "./AnimatedSection";
import AnimatedCard from "./AnimatedCard";

const FeatureCard = ({ icon: Icon, title, description, details }) => (
  <div className="bg-white rounded-xl shadow-lg p-6 hover:shadow-xl transition-all duration-300 hover:scale-102">
    <div className="w-12 h-12 bg-[#FCE4EC] rounded-lg flex items-center justify-center mb-4">
      <Icon className="text-2xl text-[#e91e4d]" />
    </div>
    <h3 className="text-xl font-semibold text-[#212121] mb-2">{title}</h3>
    <p className="text-gray-600 mb-4">{description}</p>
    <ul className="space-y-1 text-sm text-gray-500">
      {details.map((detail, index) => (
        <li key={index} className="flex items-center space-x-2">
          <div className="w-1 h-1 bg-[#e91e4d] rounded-full"></div>
          <span>{detail}</span>
        </li>
      ))}
    </ul>
  </div>
);

const StatCard = ({ value, label, description }) => (
  <div className="text-center">
    <div className="text-3xl font-bold text-[#e91e4d] mb-2">{value}</div>
    <div className="text-lg font-medium text-[#212121] mb-1">{label}</div>
    <div className="text-sm text-gray-600">{description}</div>
  </div>
);

const AboutSection = () => {
  const architectureFeatures = [
    {
      icon: FiLayers,
      title: "StackedEnsembleNet",
      description:
        "Custom ensemble network combining three state-of-the-art CNN architectures with meta-learner fusion",
      details: [
        "InceptionV3 with auxiliary classifiers",
        "InceptionResNetV2 with residual connections",
        "Xception with depthwise separable convolutions",
        "Meta-learner for final classification",
      ],
    },
    {
      icon: MdScience,
      title: "Explainable AI",
      description:
        "Grad-CAM visualizations provide interpretable insights into model predictions for clinical validation",
      details: [
        "Gradient-based Class Activation Maps",
        "Visual attention highlighting on CT scans",
        "Transparent decision making process",
        "Assists radiologists in diagnosis verification",
      ],
    },
    {
      icon: FiShield,
      title: "Medical Grade Performance",
      description:
        "Rigorously tested on a plethora of clinical CT imaging data with proper hospital-grade validation",
      details: [
        "98.74% classification accuracy",
        "98.57% precision rate",
        "98.96% recall sensitivity",
        "Clinical dataset validated",
      ],
    },
  ];

  const technicalSpecs = [
    {
      icon: FiDatabase,
      title: "Hybrid Dataset",
      description:
        "Comprehensive medical imaging dataset from multiple clinical sources",
      details: [
        "Kaggle Axial CT Imaging Dataset",
        "Elazığ Fethi Sekin City Hospital data",
        "Extensive offline data augmentation",
        "Balanced Kidney Stone vs Normal classes",
      ],
    },
    {
      icon: FiCpu,
      title: "CNN Architecture",
      description:
        "Advanced deep learning pipeline with custom classifier heads and optimization",
      details: [
        "Custom classifier: 256 → 128 → 2 neurons",
        "Batch normalization & ReLU activation",
        "Dropout regularization",
        "Adam optimizer with Cross Entropy Loss",
      ],
    },
    {
      icon: FiTrendingUp,
      title: "Performance Metrics",
      description:
        "Comprehensive evaluation demonstrating strong diagnostic performance",
      details: [
        "Accuracy: 98.74%",
        "Precision: 98.57%",
        "Recall: 98.96%",
        "F1-Score: 98.76%",
      ],
    },
  ];

  return (
    <section id="about" className="py-20 bg-[#FAFAFA]">
      <div className="container mx-auto px-4">
        {/* Header */}
        <AnimatedSection className="text-center mb-16">
          <div className="inline-flex items-center space-x-2 bg-[#FCE4EC] px-4 py-2 rounded-full mb-4">
            <MdBiotech className="text-[#e91e4d]" />
            <span className="text-sm font-medium text-[#ad1442]">
              Technology
            </span>
          </div>
          <h2 className="text-4xl md:text-5xl font-bold text-[#212121] mb-4">
            About SEN-D
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Deep Learning Classification using CNNs and Ensemble Methods for
            automated kidney stone detection from axial CT scan images
          </p>
        </AnimatedSection>

        {/* Statistics */}
        <AnimatedSection
          className="grid grid-cols-2 md:grid-cols-4 gap-8 mb-16 max-w-4xl mx-auto"
          delay={200}
        >
          <AnimatedCard delay={0}>
            <StatCard
              value="98.7%"
              label="Accuracy"
              description="Clinical validation"
            />
          </AnimatedCard>
          <AnimatedCard delay={100}>
            <StatCard
              value="~ 3s"
              label="Processing"
              description="Real-time analysis"
            />
          </AnimatedCard>
          <AnimatedCard delay={200}>
            <StatCard
              value="3"
              label="CNN Models"
              description="Ensemble fusion"
            />
          </AnimatedCard>
          <AnimatedCard delay={300}>
            <StatCard
              value="10K+"
              label="Dataset"
              description="Medical images"
            />
          </AnimatedCard>
        </AnimatedSection>

        {/* Architecture Features */}
        <div className="mb-16">
          <AnimatedSection>
            <h3 className="text-3xl font-bold text-[#212121] text-center mb-8">
              AI Architecture
            </h3>
          </AnimatedSection>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-6xl mx-auto">
            {architectureFeatures.map((feature, index) => (
              <AnimatedCard key={index} delay={index * 150} animation="scaleIn">
                <FeatureCard {...feature} />
              </AnimatedCard>
            ))}
          </div>
        </div>

        {/* Technical Specifications */}
        <div className="mb-16">
          <AnimatedSection>
            <h3 className="text-3xl font-bold text-[#212121] text-center mb-8">
              Technical Specifications
            </h3>
          </AnimatedSection>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-6xl mx-auto">
            {technicalSpecs.map((spec, index) => (
              <AnimatedCard key={index} delay={index * 150} animation="slideUp">
                <FeatureCard {...spec} />
              </AnimatedCard>
            ))}
          </div>
        </div>

        {/* Research Background */}
        <AnimatedSection delay={300}>
          <div className="bg-white rounded-2xl shadow-lg p-8 max-w-5xl mx-auto hover:scale-102 transition-transform duration-300">
            <div className="flex items-center justify-center space-x-3 mb-6">
              <div className="w-12 h-12 bg-[#FCE4EC] rounded-lg flex items-center justify-center">
                <FiBookOpen className="text-2xl text-[#e91e4d]" />
              </div>
              <h3 className="text-2xl font-semibold text-[#212121]">
                Research Foundation
              </h3>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div>
                <h4 className="font-semibold text-[#212121] mb-3">
                  Reference Research
                </h4>
                <div className="bg-gray-50 rounded-lg p-4 mb-4">
                  <p className="text-sm text-gray-700 italic mb-2">
                    "An Optimized Fusion of Deep Learning Models for Kidney
                    Stone Detection from CT Images"
                  </p>
                  <p className="text-xs text-gray-600">
                    Computers in Biology and Medicine, 2024 - ScienceDirect
                  </p>
                </div>
                <h4 className="font-semibold text-[#212121] mb-3">
                  Clinical Objective
                </h4>
                <ul className="space-y-2 text-sm text-gray-600">
                  <li>• Binary classification: Kidney Stone vs Normal</li>
                  <li>• Support early detection and diagnosis</li>
                  <li>• Assist radiologists in CT scan analysis</li>
                  <li>• Improve diagnostic efficiency and accuracy</li>
                </ul>
              </div>
              <div>
                <h4 className="font-semibold text-[#212121] mb-3">
                  Data Sources
                </h4>
                <ul className="space-y-2 text-sm text-gray-600">
                  <li>• Kaggle Axial CT Imaging Dataset</li>
                  <li>• Elazığ Fethi Sekin City Hospital, Turkey</li>
                  <li>• Clinical-grade CT imaging data</li>
                  <li>• Extensive offline augmentation techniques</li>
                </ul>
                <h4 className="font-semibold text-[#212121] mb-3 mt-4">
                  Implementation Details
                </h4>
                <ul className="space-y-2 text-sm text-gray-600">
                  <li>• PyTorch & timm library implementation</li>
                  <li>• Modular, extensible platform design</li>
                  <li>• Optimized for medical image analysis</li>
                  <li>• Ready for clinical deployment</li>
                </ul>
              </div>
            </div>
          </div>
        </AnimatedSection>
      </div>
    </section>
  );
};

export default AboutSection;
