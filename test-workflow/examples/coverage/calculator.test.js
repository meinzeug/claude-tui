// Test file for coverage demonstration
// Note: This file intentionally doesn't test all methods to show coverage gaps

const Calculator = require('./calculator');

describe('Calculator Coverage Demo', () => {
  let calc;
  
  beforeEach(() => {
    calc = new Calculator();
  });
  
  describe('Basic Operations', () => {
    it('should add numbers correctly', () => {
      expect(calc.add(2, 3)).toBe(5);
      expect(calc.add(-1, 1)).toBe(0);
    });
    
    it('should subtract numbers correctly', () => {
      expect(calc.subtract(5, 3)).toBe(2);
      expect(calc.subtract(0, 5)).toBe(-5);
    });
    
    it('should multiply numbers correctly', () => {
      expect(calc.multiply(4, 3)).toBe(12);
      expect(calc.multiply(0, 5)).toBe(0);
    });
    
    it('should divide numbers correctly', () => {
      expect(calc.divide(10, 2)).toBe(5);
      expect(calc.divide(7, 2)).toBe(3.5);
    });
    
    it('should throw error for division by zero', () => {
      expect(() => calc.divide(5, 0)).toThrow('Division by zero is not allowed');
    });
  });
  
  describe('Power Operations', () => {
    it('should calculate power correctly', () => {
      expect(calc.power(2, 3)).toBe(8);
      expect(calc.power(5, 0)).toBe(1);
    });
    
    it('should handle negative exponents', () => {
      expect(calc.power(2, -2)).toBe(0.25);
    });
  });
  
  describe('Square Root', () => {
    it('should calculate square root correctly', () => {
      expect(calc.sqrt(4)).toBeCloseTo(2);
      expect(calc.sqrt(9)).toBeCloseTo(3);
      expect(calc.sqrt(0)).toBe(0);
      expect(calc.sqrt(1)).toBe(1);
    });
    
    it('should throw error for negative numbers', () => {
      expect(() => calc.sqrt(-4)).toThrow('Cannot calculate square root of negative number');
    });
  });
  
  // Note: factorial method is not tested, showing coverage gap
  
  describe('Complex Operations', () => {
    it('should perform add operation through complex method', () => {
      expect(calc.complexOperation(2, 3, 'add')).toBe(5);
    });
    
    it('should perform multiply operation through complex method', () => {
      expect(calc.complexOperation(4, 3, 'multiply')).toBe(12);
    });
    
    // Note: Many operations are not tested through complexOperation, showing branch coverage gaps
    
    it('should throw error for unknown operation', () => {
      expect(() => calc.complexOperation(1, 2, 'unknown')).toThrow('Unknown operation');
    });
  });
});