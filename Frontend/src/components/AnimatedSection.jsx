import React from 'react';
import { useScrollAnimation } from '../hooks/useScrollAnimation';

const AnimatedSection = ({ 
  children, 
  className = '', 
  animation = 'slideUp',
  delay = 0,
  duration = 600 
}) => {
  const [ref, isVisible] = useScrollAnimation();

  const animations = {
    slideUp: {
      initial: 'opacity-0 translate-y-8',
      animate: 'opacity-100 translate-y-0'
    },
    slideLeft: {
      initial: 'opacity-0 translate-x-8',
      animate: 'opacity-100 translate-x-0'
    },
    slideRight: {
      initial: 'opacity-0 -translate-x-8',
      animate: 'opacity-100 translate-x-0'
    },
    fadeIn: {
      initial: 'opacity-0',
      animate: 'opacity-100'
    },
    scaleIn: {
      initial: 'opacity-0 scale-95',
      animate: 'opacity-100 scale-100'
    }
  };

  const selectedAnimation = animations[animation] || animations.slideUp;

  return (
    <div
      ref={ref}
      className={`transform transition-all ease-out ${
        isVisible ? selectedAnimation.animate : selectedAnimation.initial
      } ${className}`}
      style={{
        transitionDuration: `${duration}ms`,
        transitionDelay: `${delay}ms`
      }}
    >
      {children}
    </div>
  );
};

export default AnimatedSection;