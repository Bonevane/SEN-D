import React from "react";
import {
  FiLayers,
  FiActivity,
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
        "Advanced ensemble learning combining three state-of-the-art deep learning models",
      details: [
        "InceptionV3 base model",
        "InceptionResNetV2 architecture",
        "Xception network",
        "Meta-learner fusion approach",
      ],
    },
    {
      icon: MdScience,
      title: "Explainable AI",
      description:
        "Grad-CAM visualizations provide interpretable insights into model predictions",
      details: [
        "Gradient-based Class Activation Maps",
        "Visual attention highlighting",
        "Transparent decision making",
        "Clinical interpretability",
      ],
    },
    {
      icon: FiShield,
      title: "Medical Grade Accuracy",
      description:
        "Rigorously tested and validated for clinical diagnostic applications",
      details: [
        "99.2% classification accuracy",
        "Cross-validated performance",
        "Robust to image variations",
        "Clinical dataset trained",
      ],
    },
  ];

  const technicalSpecs = [
    {
      icon: FiDatabase,
      title: "Dataset Information",
      description:
        "Comprehensive medical imaging dataset for robust model training",
      details: [
        "10,000+ CT scan images",
        "Balanced dataset composition",
        "Expert radiologist annotations",
        "Multi-center data collection",
      ],
    },
    {
      icon: FiCpu,
      title: "Processing Pipeline",
      description: "Optimized image preprocessing and model inference pipeline",
      details: [
        "Automatic image resizing (299x299)",
        "Pixel normalization",
        "Data augmentation techniques",
        "Real-time inference < 2s",
      ],
    },
    {
      icon: FiTrendingUp,
      title: "Performance Metrics",
      description: "Comprehensive evaluation across multiple clinical metrics",
      details: [
        "Sensitivity: 98.7%",
        "Specificity: 99.5%",
        "AUC-ROC: 0.996",
        "F1-Score: 98.9%",
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
            Advanced AI system for kidney stone detection using ensemble deep
            learning and explainable AI
          </p>
        </AnimatedSection>

        {/* Statistics */}
        <AnimatedSection
          className="grid grid-cols-2 md:grid-cols-4 gap-8 mb-16 max-w-4xl mx-auto"
          delay={200}
        >
          <AnimatedCard delay={0}>
            <StatCard
              value="99.2%"
              label="Accuracy"
              description="Clinical validation"
            />
          </AnimatedCard>
          <AnimatedCard delay={100}>
            <StatCard
              value="< 2s"
              label="Processing"
              description="Real-time analysis"
            />
          </AnimatedCard>
          <AnimatedCard delay={200}>
            <StatCard
              value="3"
              label="AI Models"
              description="Ensemble approach"
            />
          </AnimatedCard>
          <AnimatedCard delay={300}>
            <StatCard
              value="10K+"
              label="Training Images"
              description="Medical dataset"
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

        {/* References */}
        <AnimatedSection delay={300}>
          <div className="bg-white rounded-2xl shadow-lg p-8 max-w-4xl mx-auto hover:scale-102 transition-transform duration-300">
            <h3 className="text-2xl font-semibold text-[#212121] mb-6 text-center">
              Research & References
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div>
                <h4 className="font-semibold text-[#212121] mb-3">
                  Deep Learning Models
                </h4>
                <ul className="space-y-2 text-sm text-gray-600">
                  <li>• Szegedy et al. - Inception v3 Architecture</li>
                  <li>• Szegedy et al. - Inception-ResNet Networks</li>
                  <li>• Chollet - Xception Deep Learning</li>
                  <li>• PyTorch Image Models (timm) Library</li>
                </ul>
              </div>
              <div>
                <h4 className="font-semibold text-[#212121] mb-3">
                  Explainable AI
                </h4>
                <ul className="space-y-2 text-sm text-gray-600">
                  <li>• Selvaraju et al. - Grad-CAM Visualization</li>
                  <li>• Medical Image Analysis Standards</li>
                  <li>• Clinical Decision Support Systems</li>
                  <li>• FDA AI/ML Guidance Documentation</li>
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
