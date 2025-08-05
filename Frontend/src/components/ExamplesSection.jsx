import { FiCheck, FiX, FiInfo } from "react-icons/fi";
import AnimatedSection from "./AnimatedSection";
import AnimatedCard from "./AnimatedCard";

const ExampleCard = ({
  title,
  description,
  isGood,
  tips,
  imageUrl,
  imageAlt,
}) => (
  <div className="bg-white rounded-3xl shadow-lg overflow-hidden hover:shadow-xl transition-all duration-300 hover:scale-102 will-change-transform">
    <div className="h-auto flex items-center justify-center relative overflow-hidden">
      <img src={imageUrl} alt={imageAlt} className="w-full h-full" />

      {/* Overlay badge */}
      <div
        className="absolute top-0 left-0 right-0 flex items-center justify-center"
        style={{ bottom: "35%" }}
      >
        <div
          className={`inline-flex items-center p-3 rounded-full text-sm font-medium ${
            isGood ? "bg-green-100 text-green-800" : "bg-red-100 text-red-800"
          } border border-white/30 shadow-lg backdrop-blur-sm bg-opacity-95`}
        >
          {isGood ? <FiCheck size={32} /> : <FiX size={32} />}
        </div>
      </div>

      <div className="absolute p-6 bottom-0 left-0 right-0 bg-white/90 backdrop-blur-md will-change-transform">
        <h4 className="font-semibold text-[#212121] mb-2">{title}</h4>
        <p className="text-gray-600 text-sm mb-4">{description}</p>

        <div className="space-y-2">
          {tips.map((tip, index) => (
            <div key={index} className="flex items-start space-x-2 text-sm">
              <FiInfo
                className="text-[#e91e4d] mt-0.5 flex-shrink-0"
                size={16}
              />
              <span className="text-gray-600">{tip}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  </div>
);

const ExamplesSection = () => {
  const goodExamples = [
    {
      title: "Clear Frontal CT Scan",
      description:
        "High-resolution frontal view CT scan with clear kidney structures and good contrast",
      imageUrl: "good_1.png",
      imageAlt: "Clear frontal medical CT scan showing kidney structures",
      tips: [
        "Sharp image quality with excellent contrast",
        "Kidney regions clearly visible in frontal view",
        "Single, focused CT slice without distractions",
      ],
    },
    {
      title: "Cross-Section CT View",
      description:
        "Properly oriented cross-sectional CT scan showing detailed kidney anatomy",
      imageUrl: "good_2.png",
      imageAlt: "Cross-sectional CT scan showing kidney anatomy",
      tips: [
        "Standard cross-sectional orientation",
        "Clear anatomical structures visible",
        "Appropriate DICOM windowing and contrast",
      ],
    },
  ];

  const poorExamples = [
    {
      title: "Multi-Panel CT Report",
      description:
        "Complete CT scan report with multiple views and annotations - too busy for AI analysis",
      imageUrl: "bad_1.jpg",
      imageAlt: "Multi-panel CT report with multiple views",
      tips: [
        "Contains multiple CT slices in one image",
        "Text annotations and measurements interfere",
        "Crop to single view for better analysis",
      ],
    },
    {
      title: "Non-Medical Equipment",
      description:
        "Medical equipment images like stethoscopes are not CT scans",
      imageUrl: "bad_2.jpeg",
      imageAlt: "Stethoscope - non-medical imaging example",
      tips: [
        "Not a medical scan or imaging study",
        "Equipment photos cannot be analyzed",
        "Upload actual CT scan images only",
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
            <div className="space-y-8">
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
            <div className="space-y-8">
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
          className="mt-16 bg-[#FCE4EC]/20 rounded-2xl p-8 max-w-4xl mx-auto border border-[#e91e4d]/30"
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

        {/* Additional Tips Section */}
        <AnimatedSection
          className="mt-12 max-w-4xl mx-auto"
          animation="scaleIn"
          delay={700}
        >
          <div className="bg-blue-50 border border-blue-200 rounded-xl p-6">
            <div className="flex items-start space-x-3">
              <FiInfo
                className="text-blue-600 mt-[2px] flex-shrink-0"
                size={20}
              />
              <div>
                <h4 className="font-semibold text-blue-900 mb-2">
                  Best Practices for Image Upload
                </h4>
                <ul className="text-sm text-blue-800 space-y-1">
                  <li>• Upload single CT scan slices, not full reports</li>
                  <li>• Ensure images show abdominal/kidney region clearly</li>
                  <li>• Use images with good contrast and minimal noise</li>
                  <li>• Avoid multi-panel reports with annotations</li>
                  <li>
                    • Check that anatomical structures are clearly visible
                  </li>
                  <li>• Crop images to show only the relevant CT slice</li>
                </ul>
              </div>
            </div>
          </div>
        </AnimatedSection>
      </div>
    </section>
  );
};

export default ExamplesSection;
