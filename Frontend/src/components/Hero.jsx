import React from "react";
import { FiActivity, FiZap, FiShield, FiTrendingUp } from "react-icons/fi";
import { MdScience, MdBiotech, MdHealthAndSafety } from "react-icons/md";

const FloatingIcon = ({ Icon, className, delay = 0 }) => (
  <div
    className={`absolute animate-bounce ${className}`}
    style={{
      animationDelay: `${delay}s`,
      animationDuration: "3s",
    }}
  >
    <Icon className="text-white/20 text-4xl" />
  </div>
);

const Hero = () => {
  const scrollToPrediction = () => {
    const element = document.getElementById("prediction");
    if (element) {
      element.scrollIntoView({ behavior: "smooth" });
    }
  };

  return (
    <section
      id="hero"
      className="relative min-h-screen flex items-center justify-center overflow-hidden"
    >
      {/* Background Gradient */}
      <div className="absolute inset-0 bg-gradient-to-br from-[#e91e4d] via-[#ad1442] to-[#8b0a2e]" />

      {/* Medical Pattern Overlay */}
      <div className="absolute inset-0 opacity-10">
        <div
          className="absolute inset-0"
          style={{
            backgroundImage: `url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.1'%3E%3Cpath d='M30 30c0-11.046-8.954-20-20-20s-20 8.954-20 20 8.954 20 20 20 20-8.954 20-20zm0 0c0 11.046 8.954 20 20 20s20-8.954 20-20-8.954-20-20-20-20 8.954-20 20z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")`,
          }}
        />
      </div>

      {/* Floating Medical Icons */}
      <FloatingIcon Icon={FiActivity} className="top-20 left-20" delay={0} />
      <FloatingIcon Icon={MdScience} className="top-32 right-32" delay={1} />
      <FloatingIcon Icon={FiShield} className="bottom-40 left-16" delay={2} />
      <FloatingIcon
        Icon={MdBiotech}
        className="bottom-20 right-20"
        delay={0.5}
      />
      <FloatingIcon Icon={FiZap} className="top-1/4 left-1/2" delay={1.5} />
      <FloatingIcon
        Icon={MdHealthAndSafety}
        className="bottom-1/3 right-1/4"
        delay={2.5}
      />

      {/* Main Content */}
      <div className="relative z-10 text-center px-4 max-w-5xl mx-auto">
        <div className="mb-8 animate-fade-in">
          <div className="inline-flex items-center space-x-3 mb-6">
            <div className="w-16 h-16 bg-white/20 backdrop-blur-sm rounded-full flex items-center justify-center transition-all duration-300 transform hover:scale-105 hover:shadow-2xl">
              <FiActivity className="text-3xl text-white" />
            </div>
            <div className="w-16 h-16 bg-white/20 backdrop-blur-sm rounded-full flex items-center justify-center transition-all duration-300 transform hover:scale-105 hover:shadow-2xl">
              <MdScience className="text-3xl text-white" />
            </div>
            <div className="w-16 h-16 bg-white/20 backdrop-blur-sm rounded-full flex items-center justify-center transition-all duration-300 transform hover:scale-105 hover:shadow-2xl">
              <FiTrendingUp className="text-3xl text-white" />
            </div>
          </div>
        </div>

        <h1 className="text-5xl md:text-7xl font-bold text-white mb-6 leading-tight animate-slide-up">
          <span className="block">SEN-D</span>
          <span className="text-3xl md:text-4xl font-normal text-white/90 block mt-2">
            AI-Powered Kidney Stone Detection
          </span>
        </h1>

        <p className="text-xl md:text-2xl text-white/90 mb-8 max-w-3xl mx-auto leading-relaxed animate-slide-up-delay">
          Advanced medical imaging analysis using ensemble deep learning with
          explainable AI through Grad-CAM visualizations
        </p>

        <div className="flex flex-col sm:flex-row gap-4 justify-center items-center animate-slide-up-delay-2">
          <button
            onClick={scrollToPrediction}
            className="group px-8 py-4 bg-white text-[#e91e4d] font-semibold rounded-full hover:bg-white/90 transition-all duration-300 transform hover:scale-105 hover:shadow-2xl flex items-center space-x-2 cursor-pointer"
          >
            <span>Start Predicting</span>
            <FiActivity className="group-hover:animate-pulse" />
          </button>

          <div className="flex items-center space-x-6 text-white/80">
            <div className="text-center">
              <div className="text-2xl font-bold text-white">99.2%</div>
              <div className="text-sm">Accuracy</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-white">2s</div>
              <div className="text-sm">Processing</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-white">3</div>
              <div className="text-sm">AI Models</div>
            </div>
          </div>
        </div>
      </div>

      {/* Scroll Indicator */}
      <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2 animate-bounce">
        <div className="w-6 h-10 border-2 border-white/50 rounded-full flex justify-center">
          <div className="w-1 h-3 bg-white/70 rounded-full animate-pulse mt-2"></div>
        </div>
      </div>

      <style jsx>{`
        @keyframes fade-in {
          from {
            opacity: 0;
          }
          to {
            opacity: 1;
          }
        }

        @keyframes slide-up {
          from {
            opacity: 0;
            transform: translateY(30px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        .animate-fade-in {
          animation: fade-in 1s ease-out;
        }

        .animate-slide-up {
          animation: slide-up 1s ease-out;
        }

        .animate-slide-up-delay {
          animation: slide-up 1s ease-out 0.3s both;
        }

        .animate-slide-up-delay-2 {
          animation: slide-up 1s ease-out 0.6s both;
        }
      `}</style>
    </section>
  );
};

export default Hero;
