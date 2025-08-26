// Advanced test example demonstrating setup/teardown and mocking

class Database {
  constructor() {
    this.connected = false;
    this.data = new Map();
  }
  
  async connect() {
    // Simulate connection delay
    await new Promise(resolve => setTimeout(resolve, 50));
    this.connected = true;
  }
  
  async disconnect() {
    await new Promise(resolve => setTimeout(resolve, 25));
    this.connected = false;
    this.data.clear();
  }
  
  async save(key, value) {
    if (!this.connected) {
      throw new Error('Database not connected');
    }
    this.data.set(key, value);
    return true;
  }
  
  async find(key) {
    if (!this.connected) {
      throw new Error('Database not connected');
    }
    return this.data.get(key);
  }
  
  getConnectionStatus() {
    return this.connected;
  }
}

class UserService {
  constructor(database) {
    this.db = database;
  }
  
  async createUser(userData) {
    if (!userData.email || !userData.name) {
      throw new Error('Email and name are required');
    }
    
    const user = {
      id: Date.now(),
      ...userData,
      createdAt: new Date().toISOString()
    };
    
    await this.db.save(`user:${user.id}`, user);
    return user;
  }
  
  async getUser(id) {
    const user = await this.db.find(`user:${id}`);
    if (!user) {
      throw new Error('User not found');
    }
    return user;
  }
}

describe('User Service with Database', () => {
  let database;
  let userService;
  
  // Setup before all tests
  beforeAll(async () => {
    database = new Database();
    await database.connect();
    userService = new UserService(database);
  });
  
  // Cleanup after all tests
  afterAll(async () => {
    await database.disconnect();
  });
  
  // Reset data before each test
  beforeEach(() => {
    database.data.clear();
  });
  
  describe('User Creation', () => {
    it('should create a user successfully', async () => {
      const userData = {
        name: 'John Doe',
        email: 'john@example.com'
      };
      
      const user = await userService.createUser(userData);
      
      expect(user.name).toBe('John Doe');
      expect(user.email).toBe('john@example.com');
      expect(user.id).toBeTruthy();
      expect(user.createdAt).toBeTruthy();
    });
    
    it('should validate required fields', async () => {
      await expect(userService.createUser({})).rejects.toThrow('Email and name are required');
      await expect(userService.createUser({ name: 'John' })).rejects.toThrow('Email and name are required');
      await expect(userService.createUser({ email: 'john@example.com' })).rejects.toThrow('Email and name are required');
    });
  });
  
  describe('User Retrieval', () => {
    let createdUser;
    
    beforeEach(async () => {
      createdUser = await userService.createUser({
        name: 'Jane Doe',
        email: 'jane@example.com'
      });
    });
    
    it('should retrieve an existing user', async () => {
      const user = await userService.getUser(createdUser.id);
      
      expect(user).toEqual(createdUser);
    });
    
    it('should throw error for non-existent user', async () => {
      await expect(userService.getUser(999999)).rejects.toThrow('User not found');
    });
  });
  
  describe('Database Connection', () => {
    it('should be connected during tests', () => {
      expect(database.getConnectionStatus()).toBe(true);
    });
    
    it('should handle database disconnection gracefully', async () => {
      const tempDb = new Database();
      const tempService = new UserService(tempDb);
      
      // Database is not connected
      await expect(tempService.createUser({
        name: 'Test User',
        email: 'test@example.com'
      })).rejects.toThrow('Database not connected');
    });
  });
});