import React, { useState, useEffect } from "react";
import { FiActivity, FiHome, FiInfo, FiImage } from "react-icons/fi";

const Navigation = () => {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      const scrolled = window.scrollY > 100;
      setIsVisible(scrolled);
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const scrollToSection = (sectionId) => {
    const element = document.getElementById(sectionId);
    if (element) {
      element.scrollIntoView({ behavior: "smooth" });
    }
  };

  return (
    <nav
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        isVisible
          ? "bg-white/80 backdrop-blur-md shadow-lg translate-y-0"
          : "bg-transparent -translate-y-full"
      }`}
    >
      <div className="container mx-auto px-8 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <FiActivity className="text-2xl text-[#e91e4d]" />
            <span className="text-xl font-bold text-[#212121]">SEN-D</span>
          </div>

          <div className="hidden md:flex items-center space-x-6">
            <button
              onClick={() => scrollToSection("hero")}
              className="flex items-center space-x-2 text-[#212121] hover:text-[#e91e4d] transition-colors cursor-pointer"
            >
              <FiHome size={16} />
              <span>Home</span>
            </button>
            <button
              onClick={() => scrollToSection("prediction")}
              className="flex items-center space-x-2 text-[#212121] hover:text-[#e91e4d] transition-colors cursor-pointer"
            >
              <FiImage size={16} />
              <span>Predict</span>
            </button>
            <button
              onClick={() => scrollToSection("about")}
              className="flex items-center space-x-2 text-[#212121] hover:text-[#e91e4d] transition-colors cursor-pointer"
            >
              <FiInfo size={16} />
              <span>About</span>
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navigation;
