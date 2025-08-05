import React, { useEffect, useState } from 'react';
import Navigation from './components/Navigation';
import Hero from './components/Hero';
import PredictionSection from './components/PredictionSection';
import ExamplesSection from './components/ExamplesSection';
import AboutSection from './components/AboutSection';
import { healthCheck } from './utils/api';
import { FiWifi, FiWifiOff } from 'react-icons/fi';

function App() {
  const [apiStatus, setApiStatus] = useState('checking');

  useEffect(() => {
    const checkAPI = async () => {
      try {
        const isHealthy = await healthCheck();
        setApiStatus(isHealthy ? 'connected' : 'disconnected');
      } catch (error) {
        setApiStatus('disconnected');
      }
    };

    checkAPI();
    const interval = setInterval(checkAPI, 30000); // Check every 30 seconds
    
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-white">
      <Navigation />
      
      {/* API Status Indicator */}
      <div className={`fixed top-4 right-4 z-50 flex items-center space-x-2 px-3 py-2 rounded-full text-sm font-medium transition-all duration-300 ${
        apiStatus === 'connected' 
          ? 'bg-green-100 text-green-800' 
          : apiStatus === 'disconnected'
          ? 'bg-red-100 text-red-800'
          : 'bg-yellow-100 text-yellow-800'
      }`}>
        {apiStatus === 'connected' ? (
          <>
            <FiWifi size={16} />
            <span>API Connected</span>
          </>
        ) : apiStatus === 'disconnected' ? (
          <>
            <FiWifiOff size={16} />
            <span>API Offline</span>
          </>
        ) : (
          <>
            <div className="w-4 h-4 border-2 border-yellow-600 rounded-full animate-spin border-t-transparent"></div>
            <span>Checking API</span>
          </>
        )}
      </div>

      <Hero />
      <PredictionSection />
      <ExamplesSection />
      <AboutSection />
      
      {/* Footer */}
      <footer className="bg-[#212121] text-white py-12">
        <div className="container mx-auto px-4 text-center">
          <div className="flex items-center justify-center space-x-2 mb-4">
            <div className="w-8 h-8 bg-[#e91e4d] rounded-full flex items-center justify-center">
              <span className="text-white font-bold text-sm">S</span>
            </div>
            <span className="text-xl font-bold">SEN-D</span>
          </div>
          <p className="text-gray-400 mb-4">
            AI-Powered Kidney Stone Detection System
          </p>
          <p className="text-sm text-gray-500">
            Built with advanced ensemble deep learning for medical imaging analysis
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;