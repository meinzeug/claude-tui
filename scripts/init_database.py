#!/usr/bin/env python3
"""
Database Initialization Script

Production-ready database initialization with:
- Environment detection (development/staging/production)
- Database connection validation
- Migration execution
- Initial data seeding
- Health checks and validation
- Rollback capabilities
- Comprehensive logging
"""

import asyncio
import argparse
import sys
import os
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.database.service import DatabaseService, create_database_service
from src.database.session import DatabaseConfig
from src.core.logger import get_logger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = get_logger(__name__)


class DatabaseInitializer:
    """Database initialization and management."""
    
    def __init__(self, environment: str = 'development', database_url: str = None):
        """
        Initialize database initializer.
        
        Args:
            environment: Target environment (development/staging/production)
            database_url: Database connection URL (optional)
        """
        self.environment = environment
        self.database_url = database_url or os.getenv('DATABASE_URL')
        self.service = None
        
        logger.info(f"Initializing database for {environment} environment")
    
    async def initialize(self) -> bool:
        """
        Initialize database service.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            # Create database service with environment-specific configuration
            config_kwargs = self._get_environment_config()
            
            self.service = await create_database_service(
                database_url=self.database_url,
                **config_kwargs
            )
            
            logger.info("Database service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize database service: {e}")
            return False
    
    def _get_environment_config(self) -> dict:
        """
        Get environment-specific database configuration.
        
        Returns:
            dict: Database configuration parameters
        """
        config = {
            'development': {
                'pool_size': 5,
                'max_overflow': 2,
                'pool_timeout': 30,
                'pool_pre_ping': True,
                'echo': False
            },
            'staging': {
                'pool_size': 10,
                'max_overflow': 5,
                'pool_timeout': 60,
                'pool_pre_ping': True,
                'echo': False
            },
            'production': {
                'pool_size': 20,
                'max_overflow': 10,
                'pool_timeout': 30,
                'pool_pre_ping': True,
                'echo': False
            }
        }
        
        return config.get(self.environment, config['development'])
    
    async def validate_connection(self) -> bool:
        """
        Validate database connection.
        
        Returns:
            bool: True if connection is valid
        """
        logger.info("Validating database connection...")
        
        try:
            async with self.service.get_session() as session:
                await session.execute("SELECT 1")
            
            logger.info("Database connection validated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Database connection validation failed: {e}")
            return False
    
    async def run_migrations(self, target_revision: str = 'head') -> bool:
        """
        Run database migrations.
        
        Args:
            target_revision: Target migration revision
            
        Returns:
            bool: True if migrations successful
        """
        logger.info(f"Running database migrations to {target_revision}...")
        
        try:
            # Check current migration status
            migration_status = await self.service.get_migration_status()
            
            if migration_status.get('status') == 'configuration_missing':
                logger.warning("Alembic configuration missing, skipping migrations")
                return True
            
            current_rev = migration_status.get('current_revision')
            head_rev = migration_status.get('head_revision')
            
            logger.info(f"Current revision: {current_rev}")
            logger.info(f"Head revision: {head_rev}")
            
            if migration_status.get('is_up_to_date'):
                logger.info("Database is already up to date")
                return True
            
            # Run migrations
            result = await self.service.run_migrations(target_revision)
            
            if result:
                logger.info("Database migrations completed successfully")
            else:
                logger.error("Database migrations failed")
            
            return result
            
        except Exception as e:
            logger.error(f"Migration execution failed: {e}")
            return False
    
    async def seed_initial_data(self) -> bool:
        """
        Seed initial data for the environment.
        
        Returns:
            bool: True if seeding successful
        """
        logger.info(f"Seeding initial data for {self.environment} environment...")
        
        try:
            async with self.service.get_repositories() as repos:
                user_repo = repos.get_user_repository()
                
                # Check if admin user already exists
                admin_user = await user_repo.get_by_email("admin@claude-tui.com")
                
                if admin_user:
                    logger.info("Admin user already exists, skipping seed")
                    return True
                
                # Create admin user
                if self.environment == 'development':
                    # Get admin password from environment variable
                    admin_password = os.getenv('CLAUDE_TIU_ADMIN_PASSWORD', 'changeme123!')
                    if admin_password == 'changeme123!':
                        logger.warning("⚠️  Using default admin password! Set CLAUDE_TIU_ADMIN_PASSWORD env var")
                    
                    admin_user = await user_repo.create_user(
                        email="admin@claude-tui.com",
                        username="admin",
                        password=admin_password,
                        full_name="Development Admin"
                    )
                    logger.info("Development admin user created")
                
                elif self.environment in ['staging', 'production']:
                    # For staging/production, create user with strong random password
                    import secrets
                    import string
                    
                    alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
                    random_password = ''.join(secrets.choice(alphabet) for _ in range(16))
                    
                    admin_user = await user_repo.create_user(
                        email="admin@claude-tui.com",
                        username="admin",
                        password=random_password,
                        full_name="System Administrator"
                    )
                    
                    logger.info("Production admin user created")
                    logger.warning(f"IMPORTANT: Admin password is: {random_password}")
                    logger.warning("Please save this password securely and change it after first login")
                
                # Log the admin user creation
                audit_repo = repos.get_audit_repository()
                await audit_repo.log_action(
                    user_id=admin_user.id,
                    action="admin_user_created",
                    resource_type="user",
                    resource_id=str(admin_user.id),
                    result="success",
                    metadata={"environment": self.environment}
                )
            
            logger.info("Initial data seeding completed")
            return True
            
        except Exception as e:
            logger.error(f"Initial data seeding failed: {e}")
            return False
    
    async def run_health_checks(self) -> bool:
        """
        Run comprehensive health checks.
        
        Returns:
            bool: True if all health checks pass
        """
        logger.info("Running comprehensive health checks...")
        
        try:
            # Get health status
            health = await self.service.health_check()
            
            # Check overall status
            overall_status = health.get('status')
            logger.info(f"Overall health status: {overall_status}")
            
            if overall_status != 'healthy':
                logger.error("Health check failed - service is not healthy")
                return False
            
            # Check repository health
            repo_health = health.get('repository_health', {})
            failed_repos = [name for name, status in repo_health.items() if status != 'healthy']
            
            if failed_repos:
                logger.error(f"Unhealthy repositories: {', '.join(failed_repos)}")
                return False
            
            # Check connection pool
            pool_test = health.get('connection_pool_test', {})
            if not pool_test.get('test_passed', False):
                logger.error("Connection pool health check failed")
                return False
            
            # Get database statistics
            stats = await self.service.get_database_statistics()
            logger.info(f"Database statistics: {stats}")
            
            logger.info("All health checks passed")
            return True
            
        except Exception as e:
            logger.error(f"Health checks failed: {e}")
            return False
    
    async def create_backup(self, backup_path: str = None) -> bool:
        """
        Create database backup.
        
        Args:
            backup_path: Path for backup file (optional)
            
        Returns:
            bool: True if backup successful
        """
        if not backup_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"backup_claude_tui_{self.environment}_{timestamp}.sql"
        
        logger.info(f"Creating database backup: {backup_path}")
        
        try:
            result = await self.service.backup_database(backup_path)
            
            if result:
                logger.info(f"Database backup completed: {backup_path}")
            else:
                logger.error("Database backup failed")
            
            return result
            
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            return False
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.service:
            await self.service.close()
            logger.info("Database service closed")


async def main():
    """Main initialization workflow."""
    parser = argparse.ArgumentParser(description="Initialize Claude-TIU database")
    
    parser.add_argument(
        '--environment', '-e',
        choices=['development', 'staging', 'production'],
        default='development',
        help='Target environment'
    )
    
    parser.add_argument(
        '--database-url', '-d',
        help='Database connection URL'
    )
    
    parser.add_argument(
        '--skip-migrations',
        action='store_true',
        help='Skip running database migrations'
    )
    
    parser.add_argument(
        '--skip-seed',
        action='store_true',
        help='Skip seeding initial data'
    )
    
    parser.add_argument(
        '--create-backup',
        action='store_true',
        help='Create database backup before initialization'
    )
    
    parser.add_argument(
        '--backup-path',
        help='Path for backup file'
    )
    
    parser.add_argument(
        '--target-revision',
        default='head',
        help='Target migration revision'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate connection and run health checks'
    )
    
    args = parser.parse_args()
    
    # Initialize database initializer
    initializer = DatabaseInitializer(
        environment=args.environment,
        database_url=args.database_url
    )
    
    try:
        # Step 1: Initialize database service
        if not await initializer.initialize():
            logger.error("Database service initialization failed")
            return 1
        
        # Step 2: Validate connection
        if not await initializer.validate_connection():
            logger.error("Database connection validation failed")
            return 1
        
        # If validate-only mode, skip initialization steps
        if args.validate_only:
            logger.info("Validation completed successfully")
            if not await initializer.run_health_checks():
                return 1
            return 0
        
        # Step 3: Create backup if requested
        if args.create_backup:
            if not await initializer.create_backup(args.backup_path):
                logger.error("Database backup failed")
                return 1
        
        # Step 4: Run migrations
        if not args.skip_migrations:
            if not await initializer.run_migrations(args.target_revision):
                logger.error("Database migration failed")
                return 1
        
        # Step 5: Seed initial data
        if not args.skip_seed:
            if not await initializer.seed_initial_data():
                logger.error("Initial data seeding failed")
                return 1
        
        # Step 6: Run health checks
        if not await initializer.run_health_checks():
            logger.error("Health checks failed")
            return 1
        
        logger.info("Database initialization completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Database initialization interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return 1
        
    finally:
        await initializer.cleanup()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))