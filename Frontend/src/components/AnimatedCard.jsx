import React from 'react';
import { useScrollAnimation } from '../hooks/useScrollAnimation';

const AnimatedCard = ({ 
  children, 
  className = '', 
  delay = 0,
  duration = 600,
  animation = 'slideUp'
}) => {
  const [ref, isVisible] = useScrollAnimation();

  const getAnimationClasses = () => {
    const baseClasses = 'transform transition-all ease-out';
    
    switch (animation) {
      case 'slideUp':
        return `${baseClasses} ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}`;
      case 'slideLeft':
        return `${baseClasses} ${isVisible ? 'opacity-100 translate-x-0' : 'opacity-0 translate-x-8'}`;
      case 'slideRight':
        return `${baseClasses} ${isVisible ? 'opacity-100 translate-x-0' : 'opacity-0 -translate-x-8'}`;
      case 'scaleIn':
        return `${baseClasses} ${isVisible ? 'opacity-100 scale-100' : 'opacity-0 scale-95'}`;
      case 'fadeIn':
        return `${baseClasses} ${isVisible ? 'opacity-100' : 'opacity-0'}`;
      default:
        return `${baseClasses} ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}`;
    }
  };

  return (
    <div
      ref={ref}
      className={`${getAnimationClasses()} ${className}`}
      style={{
        transitionDuration: `${duration}ms`,
        transitionDelay: `${delay}ms`
      }}
    >
      {children}
    </div>
  );
};

export default AnimatedCard;