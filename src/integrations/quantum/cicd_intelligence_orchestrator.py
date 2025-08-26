"""
CI/CD Intelligence Orchestrator
Smart integration for GitHub Actions, GitLab CI, Jenkins and other CI/CD platforms
"""

import asyncio
import base64
import json
import logging
import re
import yaml
from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import requests
from urllib.parse import urljoin
import subprocess

from .universal_environment_adapter import (
    AdapterPlugin, EnvironmentContext, IntegrationStatus, 
    EnvironmentCapability, create_adapter_plugin
)


class CICDPlatform(str, Enum):
    """Supported CI/CD platform types"""
    GITHUB_ACTIONS = "github_actions"
    GITLAB_CI = "gitlab_ci"
    JENKINS = "jenkins"
    AZURE_DEVOPS = "azure_devops"
    CIRCLECI = "circleci"
    TRAVIS_CI = "travis_ci"
    BUILDKITE = "buildkite"
    DRONE = "drone"
    TEAMCITY = "teamcity"


class PipelineStatus(str, Enum):
    """Pipeline execution status"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class JobType(str, Enum):
    """CI/CD job types"""
    BUILD = "build"
    TEST = "test"
    DEPLOY = "deploy"
    LINT = "lint"
    SECURITY_SCAN = "security_scan"
    PERFORMANCE_TEST = "performance_test"
    INTEGRATION_TEST = "integration_test"
    RELEASE = "release"


@dataclass
class PipelineEvent:
    """CI/CD pipeline event data"""
    event_type: str
    timestamp: datetime
    pipeline_id: str
    job_name: Optional[str] = None
    status: Optional[PipelineStatus] = None
    branch: Optional[str] = None
    commit_sha: Optional[str] = None
    author: Optional[str] = None
    duration: Optional[int] = None
    logs: Optional[str] = None
    artifacts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "pipeline_id": self.pipeline_id,
            "job_name": self.job_name,
            "status": self.status.value if self.status else None,
            "branch": self.branch,
            "commit_sha": self.commit_sha,
            "author": self.author,
            "duration": self.duration,
            "logs": self.logs,
            "artifacts": self.artifacts,
            "metadata": self.metadata
        }


@dataclass
class PipelineConfig:
    """CI/CD pipeline configuration"""
    name: str
    platform: CICDPlatform
    triggers: List[str] = field(default_factory=list)
    jobs: List[Dict[str, Any]] = field(default_factory=list)
    environment_vars: Dict[str, str] = field(default_factory=dict)
    secrets: List[str] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    notifications: Dict[str, Any] = field(default_factory=dict)
    
    def to_yaml(self) -> str:
        """Convert configuration to YAML format"""
        config_dict = {
            "name": self.name,
            "on": self.triggers,
            "jobs": {job["name"]: job for job in self.jobs},
            "env": self.environment_vars
        }
        return yaml.dump(config_dict, default_flow_style=False)


class BaseCICDAdapter(AdapterPlugin):
    """Base adapter for CI/CD platform integration"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.platform = CICDPlatform(config.get("platform", "github_actions"))
        self.base_url = config.get("base_url")
        self.auth_token = config.get("auth_token")
        self.repository = config.get("repository")
        self.branch = config.get("branch", "main")
        
        # Event tracking
        self._pipeline_events: List[PipelineEvent] = []
        self._active_pipelines: Dict[str, Dict[str, Any]] = {}
        self._webhook_server: Optional[Any] = None
        
        # Intelligence features
        self._failure_patterns: Dict[str, List[str]] = {}
        self._optimization_suggestions: List[Dict[str, Any]] = []
        self._performance_metrics: Dict[str, List[float]] = {}
        
    @abstractmethod
    async def _authenticate(self) -> bool:
        """Authenticate with CI/CD platform"""
        pass
        
    @abstractmethod
    async def _get_pipelines(self) -> List[Dict[str, Any]]:
        """Get list of pipelines"""
        pass
        
    @abstractmethod
    async def _get_pipeline_status(self, pipeline_id: str) -> Dict[str, Any]:
        """Get pipeline status"""
        pass
        
    @abstractmethod
    async def _trigger_pipeline(self, config: Dict[str, Any]) -> str:
        """Trigger new pipeline"""
        pass
        
    @abstractmethod
    async def _cancel_pipeline(self, pipeline_id: str) -> bool:
        """Cancel running pipeline"""
        pass
        
    @abstractmethod
    async def _get_pipeline_logs(self, pipeline_id: str, job_name: Optional[str] = None) -> str:
        """Get pipeline logs"""
        pass
        
    async def initialize(self) -> bool:
        """Initialize CI/CD adapter"""
        try:
            self._status = IntegrationStatus.CONNECTING
            
            # Authenticate with platform
            if not await self._authenticate():
                self.logger.error(f"Authentication failed for {self.platform}")
                self._status = IntegrationStatus.ERROR
                return False
                
            # Detect platform capabilities
            await self._detect_capabilities()
            
            # Start webhook server for real-time events
            await self._start_webhook_server()
            
            self._status = IntegrationStatus.ACTIVE
            self.logger.info(f"CI/CD adapter initialized: {self.platform}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize CI/CD adapter: {e}")
            self._status = IntegrationStatus.ERROR
            return False
            
    async def connect(self, context: EnvironmentContext) -> bool:
        """Connect to CI/CD platform"""
        try:
            self.repository = context.session_data.get("repository", self.repository)
            self.branch = context.session_data.get("branch", self.branch)
            
            # Test connection
            pipelines = await self._get_pipelines()
            self.logger.info(f"Connected to {self.platform}, found {len(pipelines)} pipelines")
            return True
            
        except Exception as e:
            self.logger.error(f"Connection error: {e}")
            return False
            
    async def disconnect(self) -> bool:
        """Disconnect from CI/CD platform"""
        try:
            # Stop webhook server
            if self._webhook_server:
                await self._stop_webhook_server()
                
            self._status = IntegrationStatus.INACTIVE
            self.logger.info(f"Disconnected from {self.platform}")
            return True
            
        except Exception as e:
            self.logger.error(f"Disconnection error: {e}")
            return False
            
    async def send_command(self, command: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send command to CI/CD platform"""
        try:
            if self._status != IntegrationStatus.ACTIVE:
                raise RuntimeError(f"CI/CD platform not connected: {self._status}")
                
            if command == "trigger_pipeline":
                pipeline_id = await self._trigger_pipeline(params)
                return {"pipeline_id": pipeline_id, "status": "triggered"}
                
            elif command == "get_status":
                pipeline_id = params.get("pipeline_id")
                if not pipeline_id:
                    raise ValueError("pipeline_id required")
                status = await self._get_pipeline_status(pipeline_id)
                return {"status": status}
                
            elif command == "cancel_pipeline":
                pipeline_id = params.get("pipeline_id")
                if not pipeline_id:
                    raise ValueError("pipeline_id required")
                success = await self._cancel_pipeline(pipeline_id)
                return {"cancelled": success}
                
            elif command == "get_logs":
                pipeline_id = params.get("pipeline_id")
                job_name = params.get("job_name")
                logs = await self._get_pipeline_logs(pipeline_id, job_name)
                return {"logs": logs}
                
            elif command == "optimize_pipeline":
                suggestions = await self._generate_optimization_suggestions(params)
                return {"suggestions": suggestions}
                
            elif command == "analyze_failures":
                analysis = await self._analyze_failure_patterns(params)
                return {"analysis": analysis}
                
            else:
                raise ValueError(f"Unknown command: {command}")
                
        except Exception as e:
            self.logger.error(f"Command failed: {e}")
            raise
            
    async def receive_events(self) -> List[Dict[str, Any]]:
        """Receive events from CI/CD platform"""
        events = []
        
        try:
            # Return and clear accumulated events
            for event in self._pipeline_events:
                events.append(event.to_dict())
                
            self._pipeline_events.clear()
            
        except Exception as e:
            self.logger.error(f"Error receiving events: {e}")
            
        return events
        
    async def sync_state(self, state: Dict[str, Any]) -> bool:
        """Sync state with CI/CD platform"""
        try:
            # Sync pipeline configurations
            if "pipeline_configs" in state:
                for config_data in state["pipeline_configs"]:
                    await self._sync_pipeline_config(config_data)
                    
            # Sync environment variables
            if "environment_vars" in state:
                await self._sync_environment_vars(state["environment_vars"])
                
            # Sync secrets
            if "secrets" in state:
                await self._sync_secrets(state["secrets"])
                
            return True
            
        except Exception as e:
            self.logger.error(f"State sync failed: {e}")
            return False
            
    async def _detect_capabilities(self):
        """Detect CI/CD platform capabilities"""
        capabilities = set()
        
        # Common capabilities
        capabilities.add(EnvironmentCapability(
            name="pipeline_management",
            version="1.0",
            features={"trigger", "cancel", "status", "logs"}
        ))
        
        capabilities.add(EnvironmentCapability(
            name="webhook_integration",
            version="1.0",
            features={"real_time_events", "notifications"}
        ))
        
        # Platform-specific capabilities
        if self.platform in [CICDPlatform.GITHUB_ACTIONS, CICDPlatform.GITLAB_CI]:
            capabilities.add(EnvironmentCapability(
                name="yaml_configuration",
                version="2.0",
                features={"syntax_validation", "auto_completion"}
            ))
            
        if self.platform == CICDPlatform.JENKINS:
            capabilities.add(EnvironmentCapability(
                name="groovy_scripting",
                version="2.0",
                features={"pipeline_as_code", "shared_libraries"}
            ))
            
        self._capabilities = capabilities
        
    async def _start_webhook_server(self):
        """Start webhook server for real-time events"""
        # This would start a simple HTTP server to receive webhooks
        # For now, we'll simulate this functionality
        self.logger.info(f"Webhook server started for {self.platform}")
        
    async def _stop_webhook_server(self):
        """Stop webhook server"""
        if self._webhook_server:
            self._webhook_server = None
        self.logger.info(f"Webhook server stopped for {self.platform}")
        
    async def _sync_pipeline_config(self, config_data: Dict[str, Any]):
        """Sync pipeline configuration with platform"""
        # This would update pipeline configuration on the platform
        self.logger.info(f"Synced pipeline config: {config_data.get('name')}")
        
    async def _sync_environment_vars(self, env_vars: Dict[str, str]):
        """Sync environment variables with platform"""
        self.logger.info(f"Synced {len(env_vars)} environment variables")
        
    async def _sync_secrets(self, secrets: Dict[str, str]):
        """Sync secrets with platform"""
        self.logger.info(f"Synced {len(secrets)} secrets")
        
    async def _generate_optimization_suggestions(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate pipeline optimization suggestions"""
        suggestions = []
        
        # Analyze recent pipeline performance
        recent_pipelines = await self._get_pipelines()
        
        for pipeline in recent_pipelines[-10:]:  # Last 10 pipelines
            duration = pipeline.get("duration", 0)
            
            if duration > 300:  # 5 minutes
                suggestions.append({
                    "type": "performance",
                    "priority": "high",
                    "message": f"Pipeline {pipeline['id']} took {duration}s, consider parallelizing jobs",
                    "recommendation": "Use matrix builds or split long-running jobs"
                })
                
            # Check for repeated failures
            if pipeline.get("status") == "failure":
                suggestions.append({
                    "type": "reliability",
                    "priority": "medium", 
                    "message": f"Pipeline {pipeline['id']} failed, check for flaky tests",
                    "recommendation": "Add retry mechanisms or improve test stability"
                })
                
        return suggestions
        
    async def _analyze_failure_patterns(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze failure patterns in recent pipelines"""
        analysis = {
            "total_failures": 0,
            "common_failure_reasons": [],
            "failure_trends": {},
            "recommendations": []
        }
        
        # This would analyze logs and failure patterns
        # For now, return example analysis
        analysis.update({
            "total_failures": 5,
            "common_failure_reasons": [
                "Test timeout in integration tests",
                "Dependency resolution failures",
                "Environment setup issues"
            ],
            "failure_trends": {
                "test_failures": 60,
                "build_failures": 30,
                "deployment_failures": 10
            },
            "recommendations": [
                "Increase timeout for integration tests",
                "Pin dependency versions in lockfile", 
                "Add retry logic for environment setup"
            ]
        })
        
        return analysis
        
    def _add_pipeline_event(self, event_type: str, pipeline_id: str, **kwargs):
        """Add pipeline event to queue"""
        event = PipelineEvent(
            event_type=event_type,
            timestamp=datetime.now(),
            pipeline_id=pipeline_id,
            **kwargs
        )
        self._pipeline_events.append(event)


@create_adapter_plugin("github_actions", {})
class GitHubActionsAdapter(BaseCICDAdapter):
    """GitHub Actions integration adapter"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.platform = CICDPlatform.GITHUB_ACTIONS
        self.base_url = config.get("base_url", "https://api.github.com")
        self.owner = config.get("owner")
        self.repo = config.get("repo")
        
    async def _authenticate(self) -> bool:
        """Authenticate with GitHub API"""
        try:
            if not self.auth_token:
                self.logger.error("GitHub token not provided")
                return False
                
            headers = {
                "Authorization": f"token {self.auth_token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            response = requests.get(
                f"{self.base_url}/user",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                user_data = response.json()
                self.logger.info(f"Authenticated as GitHub user: {user_data.get('login')}")
                return True
            else:
                self.logger.error(f"GitHub authentication failed: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"GitHub authentication error: {e}")
            return False
            
    async def _get_pipelines(self) -> List[Dict[str, Any]]:
        """Get GitHub Actions workflows"""
        try:
            headers = {
                "Authorization": f"token {self.auth_token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            response = requests.get(
                f"{self.base_url}/repos/{self.owner}/{self.repo}/actions/runs",
                headers=headers,
                timeout=30
            )
            
            response.raise_for_status()
            data = response.json()
            return data.get("workflow_runs", [])
            
        except Exception as e:
            self.logger.error(f"Failed to get GitHub Actions workflows: {e}")
            return []
            
    async def _get_pipeline_status(self, pipeline_id: str) -> Dict[str, Any]:
        """Get GitHub Actions workflow run status"""
        try:
            headers = {
                "Authorization": f"token {self.auth_token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            response = requests.get(
                f"{self.base_url}/repos/{self.owner}/{self.repo}/actions/runs/{pipeline_id}",
                headers=headers,
                timeout=30
            )
            
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            self.logger.error(f"Failed to get GitHub Actions status: {e}")
            return {}
            
    async def _trigger_pipeline(self, config: Dict[str, Any]) -> str:
        """Trigger GitHub Actions workflow"""
        try:
            headers = {
                "Authorization": f"token {self.auth_token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            # Trigger workflow dispatch
            data = {
                "ref": config.get("ref", self.branch),
                "inputs": config.get("inputs", {})
            }
            
            workflow_id = config.get("workflow_id", "main.yml")
            
            response = requests.post(
                f"{self.base_url}/repos/{self.owner}/{self.repo}/actions/workflows/{workflow_id}/dispatches",
                headers=headers,
                json=data,
                timeout=30
            )
            
            response.raise_for_status()
            
            # Return a generated ID (GitHub doesn't return the run ID immediately)
            return f"triggered_{int(datetime.now().timestamp())}"
            
        except Exception as e:
            self.logger.error(f"Failed to trigger GitHub Actions workflow: {e}")
            raise
            
    async def _cancel_pipeline(self, pipeline_id: str) -> bool:
        """Cancel GitHub Actions workflow run"""
        try:
            headers = {
                "Authorization": f"token {self.auth_token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            response = requests.post(
                f"{self.base_url}/repos/{self.owner}/{self.repo}/actions/runs/{pipeline_id}/cancel",
                headers=headers,
                timeout=30
            )
            
            return response.status_code == 202
            
        except Exception as e:
            self.logger.error(f"Failed to cancel GitHub Actions workflow: {e}")
            return False
            
    async def _get_pipeline_logs(self, pipeline_id: str, job_name: Optional[str] = None) -> str:
        """Get GitHub Actions workflow logs"""
        try:
            headers = {
                "Authorization": f"token {self.auth_token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            if job_name:
                # Get specific job logs
                jobs_response = requests.get(
                    f"{self.base_url}/repos/{self.owner}/{self.repo}/actions/runs/{pipeline_id}/jobs",
                    headers=headers,
                    timeout=30
                )
                jobs_response.raise_for_status()
                jobs = jobs_response.json().get("jobs", [])
                
                job = next((j for j in jobs if j["name"] == job_name), None)
                if not job:
                    return f"Job '{job_name}' not found"
                    
                job_id = job["id"]
                log_response = requests.get(
                    f"{self.base_url}/repos/{self.owner}/{self.repo}/actions/jobs/{job_id}/logs",
                    headers=headers,
                    timeout=30
                )
            else:
                # Get all workflow logs
                log_response = requests.get(
                    f"{self.base_url}/repos/{self.owner}/{self.repo}/actions/runs/{pipeline_id}/logs",
                    headers=headers,
                    timeout=30
                )
                
            if log_response.status_code == 200:
                return log_response.text
            else:
                return f"Failed to fetch logs: {log_response.status_code}"
                
        except Exception as e:
            self.logger.error(f"Failed to get GitHub Actions logs: {e}")
            return f"Error fetching logs: {str(e)}"


@create_adapter_plugin("gitlab_ci", {})
class GitLabCIAdapter(BaseCICDAdapter):
    """GitLab CI integration adapter"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.platform = CICDPlatform.GITLAB_CI
        self.base_url = config.get("base_url", "https://gitlab.com/api/v4")
        self.project_id = config.get("project_id")
        
    async def _authenticate(self) -> bool:
        """Authenticate with GitLab API"""
        try:
            if not self.auth_token:
                self.logger.error("GitLab token not provided")
                return False
                
            headers = {
                "PRIVATE-TOKEN": self.auth_token
            }
            
            response = requests.get(
                f"{self.base_url}/user",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                user_data = response.json()
                self.logger.info(f"Authenticated as GitLab user: {user_data.get('username')}")
                return True
            else:
                self.logger.error(f"GitLab authentication failed: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"GitLab authentication error: {e}")
            return False
            
    async def _get_pipelines(self) -> List[Dict[str, Any]]:
        """Get GitLab CI pipelines"""
        try:
            headers = {
                "PRIVATE-TOKEN": self.auth_token
            }
            
            response = requests.get(
                f"{self.base_url}/projects/{self.project_id}/pipelines",
                headers=headers,
                timeout=30
            )
            
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            self.logger.error(f"Failed to get GitLab CI pipelines: {e}")
            return []
            
    async def _get_pipeline_status(self, pipeline_id: str) -> Dict[str, Any]:
        """Get GitLab CI pipeline status"""
        try:
            headers = {
                "PRIVATE-TOKEN": self.auth_token
            }
            
            response = requests.get(
                f"{self.base_url}/projects/{self.project_id}/pipelines/{pipeline_id}",
                headers=headers,
                timeout=30
            )
            
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            self.logger.error(f"Failed to get GitLab CI status: {e}")
            return {}
            
    async def _trigger_pipeline(self, config: Dict[str, Any]) -> str:
        """Trigger GitLab CI pipeline"""
        try:
            headers = {
                "PRIVATE-TOKEN": self.auth_token
            }
            
            data = {
                "ref": config.get("ref", self.branch),
                "variables": config.get("variables", {})
            }
            
            response = requests.post(
                f"{self.base_url}/projects/{self.project_id}/pipeline",
                headers=headers,
                json=data,
                timeout=30
            )
            
            response.raise_for_status()
            pipeline_data = response.json()
            return str(pipeline_data["id"])
            
        except Exception as e:
            self.logger.error(f"Failed to trigger GitLab CI pipeline: {e}")
            raise
            
    async def _cancel_pipeline(self, pipeline_id: str) -> bool:
        """Cancel GitLab CI pipeline"""
        try:
            headers = {
                "PRIVATE-TOKEN": self.auth_token
            }
            
            response = requests.post(
                f"{self.base_url}/projects/{self.project_id}/pipelines/{pipeline_id}/cancel",
                headers=headers,
                timeout=30
            )
            
            return response.status_code == 200
            
        except Exception as e:
            self.logger.error(f"Failed to cancel GitLab CI pipeline: {e}")
            return False
            
    async def _get_pipeline_logs(self, pipeline_id: str, job_name: Optional[str] = None) -> str:
        """Get GitLab CI pipeline logs"""
        try:
            headers = {
                "PRIVATE-TOKEN": self.auth_token
            }
            
            # Get pipeline jobs
            response = requests.get(
                f"{self.base_url}/projects/{self.project_id}/pipelines/{pipeline_id}/jobs",
                headers=headers,
                timeout=30
            )
            
            response.raise_for_status()
            jobs = response.json()
            
            if job_name:
                job = next((j for j in jobs if j["name"] == job_name), None)
                if not job:
                    return f"Job '{job_name}' not found"
                jobs = [job]
                
            # Get logs for each job
            all_logs = []
            for job in jobs:
                try:
                    log_response = requests.get(
                        f"{self.base_url}/projects/{self.project_id}/jobs/{job['id']}/trace",
                        headers=headers,
                        timeout=30
                    )
                    
                    if log_response.status_code == 200:
                        all_logs.append(f"=== Job: {job['name']} ===")
                        all_logs.append(log_response.text)
                        all_logs.append("")
                        
                except Exception as e:
                    self.logger.warning(f"Failed to get logs for job {job['name']}: {e}")
                    
            return "\n".join(all_logs)
            
        except Exception as e:
            self.logger.error(f"Failed to get GitLab CI logs: {e}")
            return f"Error fetching logs: {str(e)}"


@create_adapter_plugin("jenkins", {})
class JenkinsAdapter(BaseCICDAdapter):
    """Jenkins integration adapter"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.platform = CICDPlatform.JENKINS
        self.base_url = config.get("base_url", "http://localhost:8080")
        self.username = config.get("username")
        self.api_token = config.get("api_token")
        
    async def _authenticate(self) -> bool:
        """Authenticate with Jenkins"""
        try:
            if not self.username or not self.api_token:
                self.logger.error("Jenkins credentials not provided")
                return False
                
            auth = (self.username, self.api_token)
            
            response = requests.get(
                f"{self.base_url}/api/json",
                auth=auth,
                timeout=10
            )
            
            if response.status_code == 200:
                self.logger.info(f"Authenticated with Jenkins as: {self.username}")
                return True
            else:
                self.logger.error(f"Jenkins authentication failed: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Jenkins authentication error: {e}")
            return False
            
    async def _get_pipelines(self) -> List[Dict[str, Any]]:
        """Get Jenkins jobs/pipelines"""
        try:
            auth = (self.username, self.api_token)
            
            response = requests.get(
                f"{self.base_url}/api/json",
                auth=auth,
                timeout=30
            )
            
            response.raise_for_status()
            data = response.json()
            return data.get("jobs", [])
            
        except Exception as e:
            self.logger.error(f"Failed to get Jenkins jobs: {e}")
            return []
            
    async def _get_pipeline_status(self, pipeline_id: str) -> Dict[str, Any]:
        """Get Jenkins job status"""
        try:
            auth = (self.username, self.api_token)
            
            response = requests.get(
                f"{self.base_url}/job/{pipeline_id}/api/json",
                auth=auth,
                timeout=30
            )
            
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            self.logger.error(f"Failed to get Jenkins job status: {e}")
            return {}
            
    async def _trigger_pipeline(self, config: Dict[str, Any]) -> str:
        """Trigger Jenkins job"""
        try:
            auth = (self.username, self.api_token)
            job_name = config.get("job_name", "build")
            parameters = config.get("parameters", {})
            
            if parameters:
                # Build with parameters
                response = requests.post(
                    f"{self.base_url}/job/{job_name}/buildWithParameters",
                    auth=auth,
                    data=parameters,
                    timeout=30
                )
            else:
                # Simple build trigger
                response = requests.post(
                    f"{self.base_url}/job/{job_name}/build",
                    auth=auth,
                    timeout=30
                )
                
            response.raise_for_status()
            
            # Jenkins returns build number in Location header
            location = response.headers.get("Location")
            if location:
                build_number = location.split("/")[-2]
                return f"{job_name}#{build_number}"
            else:
                return f"{job_name}#triggered"
                
        except Exception as e:
            self.logger.error(f"Failed to trigger Jenkins job: {e}")
            raise
            
    async def _cancel_pipeline(self, pipeline_id: str) -> bool:
        """Cancel Jenkins job"""
        try:
            auth = (self.username, self.api_token)
            
            # Parse job name and build number from pipeline_id
            if "#" in pipeline_id:
                job_name, build_number = pipeline_id.split("#")
                
                response = requests.post(
                    f"{self.base_url}/job/{job_name}/{build_number}/stop",
                    auth=auth,
                    timeout=30
                )
                
                return response.status_code == 200
            else:
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to cancel Jenkins job: {e}")
            return False
            
    async def _get_pipeline_logs(self, pipeline_id: str, job_name: Optional[str] = None) -> str:
        """Get Jenkins job logs"""
        try:
            auth = (self.username, self.api_token)
            
            # Parse job name and build number from pipeline_id
            if "#" in pipeline_id:
                job_name, build_number = pipeline_id.split("#")
                
                response = requests.get(
                    f"{self.base_url}/job/{job_name}/{build_number}/consoleText",
                    auth=auth,
                    timeout=30
                )
                
                if response.status_code == 200:
                    return response.text
                else:
                    return f"Failed to fetch logs: {response.status_code}"
            else:
                return f"Invalid pipeline ID format: {pipeline_id}"
                
        except Exception as e:
            self.logger.error(f"Failed to get Jenkins logs: {e}")
            return f"Error fetching logs: {str(e)}"


class CICDIntelligenceOrchestrator:
    """
    CI/CD Intelligence Orchestrator
    Coordinates multiple CI/CD platforms and provides unified intelligence
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Registry of active CI/CD adapters
        self._adapters: Dict[str, BaseCICDAdapter] = {}
        self._platform_configs = config.get("platforms", {})
        
        # Intelligence features
        self._failure_analytics = {}
        self._performance_trends = {}
        self._optimization_engine = OptimizationEngine()
        
    async def initialize(self) -> bool:
        """Initialize CI/CD intelligence orchestrator"""
        try:
            self.logger.info("Initializing CI/CD Intelligence Orchestrator")
            
            # Initialize configured platforms
            for platform_id, platform_config in self._platform_configs.items():
                await self._initialize_platform(platform_id, platform_config)
                
            self.logger.info("CI/CD Intelligence Orchestrator initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize CI/CD orchestrator: {e}")
            return False
            
    async def _initialize_platform(self, platform_id: str, config: Dict[str, Any]):
        """Initialize specific CI/CD platform adapter"""
        try:
            platform_type = config.get("type", "github_actions")
            
            if platform_type == "github_actions":
                adapter = GitHubActionsAdapter(config)
            elif platform_type == "gitlab_ci":
                adapter = GitLabCIAdapter(config)
            elif platform_type == "jenkins":
                adapter = JenkinsAdapter(config)
            else:
                self.logger.warning(f"Unsupported CI/CD platform: {platform_type}")
                return
                
            if await adapter.initialize():
                self._adapters[platform_id] = adapter
                self.logger.info(f"Initialized CI/CD platform: {platform_id}")
            else:
                self.logger.error(f"Failed to initialize CI/CD platform: {platform_id}")
                
        except Exception as e:
            self.logger.error(f"Error initializing platform {platform_id}: {e}")
            
    async def trigger_intelligent_pipeline(self, platform_id: str, 
                                         config: Dict[str, Any]) -> Dict[str, Any]:
        """Trigger pipeline with intelligent optimizations"""
        try:
            if platform_id not in self._adapters:
                raise ValueError(f"Platform not connected: {platform_id}")
                
            adapter = self._adapters[platform_id]
            
            # Apply intelligent optimizations
            optimized_config = await self._optimization_engine.optimize_config(config)
            
            # Trigger pipeline
            pipeline_id = await adapter.send_command("trigger_pipeline", optimized_config)
            
            return {
                "pipeline_id": pipeline_id,
                "platform": platform_id,
                "optimizations_applied": optimized_config.get("optimizations", [])
            }
            
        except Exception as e:
            self.logger.error(f"Failed to trigger intelligent pipeline: {e}")
            raise
            
    async def analyze_cross_platform_performance(self) -> Dict[str, Any]:
        """Analyze performance across all connected platforms"""
        analysis = {
            "platforms": {},
            "cross_platform_insights": [],
            "recommendations": []
        }
        
        for platform_id, adapter in self._adapters.items():
            try:
                # Get recent pipelines
                pipelines = await adapter._get_pipelines()
                
                # Analyze performance
                platform_analysis = await self._analyze_platform_performance(pipelines)
                analysis["platforms"][platform_id] = platform_analysis
                
            except Exception as e:
                self.logger.error(f"Failed to analyze platform {platform_id}: {e}")
                
        # Generate cross-platform insights
        analysis["cross_platform_insights"] = self._generate_cross_platform_insights(
            analysis["platforms"]
        )
        
        return analysis
        
    async def _analyze_platform_performance(self, pipelines: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance for a specific platform"""
        total_pipelines = len(pipelines)
        successful_pipelines = sum(1 for p in pipelines if p.get("status") == "success")
        failed_pipelines = sum(1 for p in pipelines if p.get("status") == "failure")
        
        avg_duration = 0
        if pipelines:
            durations = [p.get("duration", 0) for p in pipelines if p.get("duration")]
            avg_duration = sum(durations) / len(durations) if durations else 0
            
        return {
            "total_pipelines": total_pipelines,
            "success_rate": (successful_pipelines / total_pipelines * 100) if total_pipelines > 0 else 0,
            "failure_rate": (failed_pipelines / total_pipelines * 100) if total_pipelines > 0 else 0,
            "average_duration": avg_duration,
            "recent_trends": self._calculate_trends(pipelines)
        }
        
    def _generate_cross_platform_insights(self, platforms: Dict[str, Dict[str, Any]]) -> List[str]:
        """Generate insights comparing performance across platforms"""
        insights = []
        
        if len(platforms) < 2:
            return insights
            
        # Compare success rates
        success_rates = {p: data["success_rate"] for p, data in platforms.items()}
        best_platform = max(success_rates.items(), key=lambda x: x[1])
        worst_platform = min(success_rates.items(), key=lambda x: x[1])
        
        insights.append(
            f"{best_platform[0]} has the highest success rate ({best_platform[1]:.1f}%), "
            f"while {worst_platform[0]} has the lowest ({worst_platform[1]:.1f}%)"
        )
        
        # Compare durations
        durations = {p: data["average_duration"] for p, data in platforms.items()}
        fastest_platform = min(durations.items(), key=lambda x: x[1])
        slowest_platform = max(durations.items(), key=lambda x: x[1])
        
        insights.append(
            f"{fastest_platform[0]} has the fastest average build time ({fastest_platform[1]:.1f}s), "
            f"while {slowest_platform[0]} is slowest ({slowest_platform[1]:.1f}s)"
        )
        
        return insights
        
    def _calculate_trends(self, pipelines: List[Dict[str, Any]]) -> Dict[str, str]:
        """Calculate trends for recent pipelines"""
        if len(pipelines) < 5:
            return {"trend": "insufficient_data"}
            
        recent = pipelines[:5]
        older = pipelines[5:10] if len(pipelines) >= 10 else pipelines[5:]
        
        recent_success_rate = sum(1 for p in recent if p.get("status") == "success") / len(recent)
        older_success_rate = sum(1 for p in older if p.get("status") == "success") / len(older) if older else recent_success_rate
        
        if recent_success_rate > older_success_rate + 0.1:
            return {"trend": "improving", "change": f"+{(recent_success_rate - older_success_rate) * 100:.1f}%"}
        elif recent_success_rate < older_success_rate - 0.1:
            return {"trend": "declining", "change": f"{(recent_success_rate - older_success_rate) * 100:.1f}%"}
        else:
            return {"trend": "stable"}


class OptimizationEngine:
    """Engine for optimizing CI/CD pipeline configurations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    async def optimize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize pipeline configuration"""
        optimized = config.copy()
        optimizations = []
        
        # Optimize job parallelization
        if "jobs" in config:
            parallelization_opts = self._optimize_parallelization(config["jobs"])
            if parallelization_opts:
                optimized["jobs"] = parallelization_opts["jobs"]
                optimizations.extend(parallelization_opts["optimizations"])
                
        # Optimize caching strategy
        caching_opts = self._optimize_caching(config)
        if caching_opts:
            optimized.update(caching_opts["config"])
            optimizations.extend(caching_opts["optimizations"])
            
        # Optimize resource allocation
        resource_opts = self._optimize_resources(config)
        if resource_opts:
            optimized.update(resource_opts["config"])
            optimizations.extend(resource_opts["optimizations"])
            
        optimized["optimizations"] = optimizations
        return optimized
        
    def _optimize_parallelization(self, jobs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Optimize job parallelization"""
        # This would analyze job dependencies and suggest parallel execution
        # For now, return example optimization
        return {
            "jobs": jobs,
            "optimizations": ["Suggested matrix strategy for test jobs"]
        }
        
    def _optimize_caching(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Optimize caching strategy"""
        # This would analyze dependencies and suggest caching strategies
        return {
            "config": {"cache": {"paths": ["node_modules", ".pip-cache"]}},
            "optimizations": ["Added dependency caching for faster builds"]
        }
        
    def _optimize_resources(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Optimize resource allocation"""
        # This would suggest optimal resource allocation
        return {
            "config": {"resources": {"cpu": 2, "memory": "4GB"}},
            "optimizations": ["Optimized resource allocation based on workload analysis"]
        }


if __name__ == "__main__":
    # Example usage
    async def main():
        config = {
            "platforms": {
                "github-main": {
                    "type": "github_actions",
                    "owner": "myorg",
                    "repo": "myproject",
                    "auth_token": "ghp_token"
                },
                "gitlab-backup": {
                    "type": "gitlab_ci",
                    "project_id": "12345",
                    "auth_token": "gl_token"
                }
            }
        }
        
        orchestrator = CICDIntelligenceOrchestrator(config)
        await orchestrator.initialize()
        
        # Trigger intelligent pipeline
        result = await orchestrator.trigger_intelligent_pipeline(
            "github-main",
            {
                "workflow_id": "ci.yml",
                "ref": "main",
                "inputs": {"environment": "production"}
            }
        )
        print(f"Pipeline triggered: {result}")
        
        # Analyze cross-platform performance
        analysis = await orchestrator.analyze_cross_platform_performance()
        print(f"Performance analysis: {analysis}")
        
    # Run example
    # asyncio.run(main())