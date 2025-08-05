import React from 'react';
import { FiActivity } from 'react-icons/fi';

const LoadingSpinner = ({ message = "Analyzing image..." }) => {
  return (
    <div className="flex flex-col items-center justify-center p-8 space-y-4">
      <div className="relative">
        <div className="w-16 h-16 border-4 border-[#FCE4EC] rounded-full animate-spin border-t-[#e91e4d]"></div>
        <div className="absolute inset-0 flex items-center justify-center">
          <FiActivity className="text-[#e91e4d] text-xl animate-pulse" />
        </div>
      </div>
      
      <div className="text-center">
        <p className="text-lg font-medium text-[#212121] mb-2">{message}</p>
        <p className="text-sm text-gray-600">
          Running ensemble analysis with 3 AI models...
        </p>
      </div>
      
      <div className="flex space-x-2">
        <div className="w-2 h-2 bg-[#e91e4d] rounded-full animate-bounce" style={{ animationDelay: '0s' }}></div>
        <div className="w-2 h-2 bg-[#e91e4d] rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
        <div className="w-2 h-2 bg-[#e91e4d] rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></div>
      </div>
    </div>
  );
};

export default LoadingSpinner;