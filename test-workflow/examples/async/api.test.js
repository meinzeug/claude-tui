// Async test examples demonstrating promise and callback testing

class ApiClient {
  async fetchUser(id) {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 100));
    
    if (!id || id < 1) {
      throw new Error('Invalid user ID');
    }
    
    return {
      id,
      name: `User ${id}`,
      email: `user${id}@example.com`
    };
  }
  
  fetchUserCallback(id, callback) {
    setTimeout(() => {
      if (!id || id < 1) {
        callback(new Error('Invalid user ID'));
        return;
      }
      
      callback(null, {
        id,
        name: `User ${id}`,
        email: `user${id}@example.com`
      });
    }, 50);
  }
}

const api = new ApiClient();

describe('API Client', () => {
  describe('Promise-based methods', () => {
    it('should fetch user data successfully', async () => {
      const user = await api.fetchUser(1);
      
      expect(user).toEqual({
        id: 1,
        name: 'User 1',
        email: 'user1@example.com'
      });
    });

    it('should handle invalid user ID', async () => {
      await expect(api.fetchUser(0)).rejects.toThrow('Invalid user ID');
    });

    it('should handle missing user ID', async () => {
      await expect(api.fetchUser()).rejects.toThrow('Invalid user ID');
    });
  });

  describe('Callback-based methods', () => {
    it('should fetch user data with callback', (done) => {
      api.fetchUserCallback(1, (error, user) => {
        if (error) {
          done(error);
          return;
        }
        
        try {
          expect(user).toEqual({
            id: 1,
            name: 'User 1',
            email: 'user1@example.com'
          });
          done();
        } catch (assertionError) {
          done(assertionError);
        }
      });
    });

    it('should handle callback errors', (done) => {
      api.fetchUserCallback(0, (error, user) => {
        try {
          expect(error).toBeTruthy();
          expect(error.message).toBe('Invalid user ID');
          expect(user).toBeFalsy();
          done();
        } catch (assertionError) {
          done(assertionError);
        }
      });
    });
  });

  describe('Multiple async operations', () => {
    it('should fetch multiple users concurrently', async () => {
      const userPromises = [1, 2, 3].map(id => api.fetchUser(id));
      const users = await Promise.all(userPromises);
      
      expect(users).toHaveLength(3);
      expect(users[0].name).toBe('User 1');
      expect(users[1].name).toBe('User 2');
      expect(users[2].name).toBe('User 3');
    });
  });
});