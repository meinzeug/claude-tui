// Source file for coverage demonstration

class Calculator {
  add(a, b) {
    return a + b;
  }
  
  subtract(a, b) {
    return a - b;
  }
  
  multiply(a, b) {
    return a * b;
  }
  
  divide(a, b) {
    if (b === 0) {
      throw new Error('Division by zero is not allowed');
    }
    return a / b;
  }
  
  power(base, exponent) {
    if (exponent === 0) {
      return 1;
    }
    
    if (exponent < 0) {
      return 1 / this.power(base, Math.abs(exponent));
    }
    
    let result = 1;
    for (let i = 0; i < exponent; i++) {
      result *= base;
    }
    return result;
  }
  
  sqrt(number) {
    if (number < 0) {
      throw new Error('Cannot calculate square root of negative number');
    }
    
    if (number === 0 || number === 1) {
      return number;
    }
    
    // Newton's method for square root
    let guess = number / 2;
    while (Math.abs(guess * guess - number) > 0.000001) {
      guess = (guess + number / guess) / 2;
    }
    
    return guess;
  }
  
  factorial(n) {
    if (n < 0) {
      throw new Error('Factorial is not defined for negative numbers');
    }
    
    if (n === 0 || n === 1) {
      return 1;
    }
    
    return n * this.factorial(n - 1);
  }
  
  // This method intentionally has low test coverage
  complexOperation(a, b, operation) {
    switch (operation) {
      case 'add':
        return this.add(a, b);
      case 'subtract':
        return this.subtract(a, b);
      case 'multiply':
        return this.multiply(a, b);
      case 'divide':
        return this.divide(a, b);
      case 'power':
        return this.power(a, b);
      case 'sqrt':
        return this.sqrt(a);
      case 'factorial':
        return this.factorial(a);
      default:
        throw new Error('Unknown operation');
    }
  }
}

module.exports = Calculator;