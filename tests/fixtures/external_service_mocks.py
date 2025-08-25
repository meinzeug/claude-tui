"""
External Service Mocks for claude-tiu Testing.

Comprehensive mocks for external services including:
- Claude Code integration mocks
- Claude Flow integration mocks
- Database connection mocks
- Email service mocks
- OAuth provider mocks
- Payment processor mocks
- File storage mocks
- Third-party API mocks
"""

import asyncio
import uuid
import json
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Union
from unittest.mock import AsyncMock, MagicMock
import pytest


class MockClaudeCodeIntegration:
    """Mock for Claude Code integration service."""
    
    def __init__(self, simulate_errors: bool = False, response_delay: float = 0.1):
        self.simulate_errors = simulate_errors
        self.response_delay = response_delay
        self.call_count = 0
        self.last_request = None
        self.connection_status = "connected"
        
    async def test_connection(self) -> Dict[str, Any]:
        """Mock connection test."""
        await asyncio.sleep(self.response_delay)
        self.call_count += 1
        
        if self.simulate_errors and self.call_count % 5 == 0:
            raise ConnectionError("Mock connection failure")
        
        return {
            "status": self.connection_status,
            "version": "1.0.0",
            "capabilities": ["code_generation", "file_operations", "project_management"]
        }
    
    async def generate_code(self, prompt: str, language: str = "python", context: Dict = None) -> Dict[str, Any]:
        """Mock code generation."""
        await asyncio.sleep(self.response_delay)
        self.call_count += 1
        self.last_request = {"prompt": prompt, "language": language, "context": context}
        
        if self.simulate_errors and "error" in prompt.lower():
            raise Exception("Mock code generation error")
        
        # Generate mock code based on prompt
        mock_code = self._generate_mock_code(prompt, language)
        
        return {
            "code": mock_code,
            "language": language,
            "quality_score": 0.85 + (hash(prompt) % 15) / 100,  # 0.85-1.00
            "metadata": {
                "lines": len(mock_code.split('\n')),
                "functions": mock_code.count('def ') + mock_code.count('function '),
                "classes": mock_code.count('class '),
                "generated_at": datetime.utcnow().isoformat()
            },
            "suggestions": self._generate_suggestions(prompt, language),
            "performance_metrics": {
                "generation_time_ms": (self.response_delay * 1000) + (hash(prompt) % 50),
                "token_count": len(prompt.split()) * 3,
                "complexity_score": (hash(prompt) % 10) + 1
            }
        }
    
    async def execute_command(self, command: str, args: List[str] = None, cwd: str = None) -> Dict[str, Any]:
        """Mock command execution."""
        await asyncio.sleep(self.response_delay / 2)  # Commands are usually faster
        self.call_count += 1
        
        if self.simulate_errors and "fail" in command:
            return {
                "status": "error",
                "exit_code": 1,
                "stdout": "",
                "stderr": f"Mock command '{command}' failed",
                "execution_time": self.response_delay
            }
        
        return {
            "status": "success",
            "exit_code": 0,
            "stdout": f"Mock output for command: {command}",
            "stderr": "",
            "execution_time": self.response_delay,
            "cwd": cwd or "/mock/working/directory"
        }
    
    async def create_file(self, file_path: str, content: str, overwrite: bool = False) -> Dict[str, Any]:
        """Mock file creation."""
        await asyncio.sleep(self.response_delay / 4)
        self.call_count += 1
        
        if self.simulate_errors and "readonly" in file_path:
            raise PermissionError("Mock permission denied")
        
        return {
            "status": "success",
            "file_path": file_path,
            "size_bytes": len(content),
            "created_at": datetime.utcnow().isoformat(),
            "checksum": str(hash(content))
        }
    
    def _generate_mock_code(self, prompt: str, language: str) -> str:
        """Generate realistic mock code based on prompt and language."""
        prompt_lower = prompt.lower()
        
        if language == "python":
            if "function" in prompt_lower:
                function_name = "generated_function"
                if "fibonacci" in prompt_lower:
                    return """def fibonacci(n):
    \"\"\"Calculate fibonacci number.\"\"\"
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)"""
                elif "sort" in prompt_lower:
                    return """def bubble_sort(arr):
    \"\"\"Sort array using bubble sort.\"\"\"
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr"""
                else:
                    return f"""def {function_name}():
    \"\"\"Generated function based on: {prompt[:50]}...\"\"\"
    # TODO: Implement function logic
    pass"""
            elif "class" in prompt_lower:
                return """class GeneratedClass:
    \"\"\"Generated class based on prompt.\"\"\"
    
    def __init__(self):
        self.initialized = True
    
    def example_method(self):
        \"\"\"Example method.\"\"\"
        return "Hello from generated class\""""
            else:
                return f"""# Generated Python code
# Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}

def main():
    \"\"\"Main function.\"\"\"
    print("Generated code executed")

if __name__ == "__main__":
    main()"""
        
        elif language == "javascript":
            if "function" in prompt_lower:
                return """function generatedFunction() {
    // Generated JavaScript function
    console.log("Generated function executed");
    return true;
}"""
            else:
                return f"""// Generated JavaScript code
// Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}

const generatedCode = {
    execute: function() {
        console.log("Generated code executed");
        return { status: "success" };
    }
};

module.exports = generatedCode;"""
        
        else:  # Generic code for other languages
            return f"""// Generated {language} code
// Based on prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}

// TODO: Implement requested functionality
"""
    
    def _generate_suggestions(self, prompt: str, language: str) -> List[str]:
        """Generate mock suggestions."""
        suggestions = []
        
        if "TODO" in self._generate_mock_code(prompt, language):
            suggestions.append("Consider implementing the marked TODO items")
        
        if language == "python":
            suggestions.extend([
                "Add type hints for better code documentation",
                "Consider adding unit tests",
                "Add docstrings to functions and classes"
            ])
        elif language == "javascript":
            suggestions.extend([
                "Consider using TypeScript for better type safety",
                "Add JSDoc comments for documentation",
                "Consider using async/await for better readability"
            ])
        
        return suggestions[:3]  # Limit to 3 suggestions


class MockClaudeFlowIntegration:
    """Mock for Claude Flow integration service."""
    
    def __init__(self, simulate_errors: bool = False, response_delay: float = 0.2):
        self.simulate_errors = simulate_errors
        self.response_delay = response_delay
        self.call_count = 0
        self.active_tasks = {}
        self.agent_pool = {}
        self.swarms = {}
        
    async def test_connection(self) -> Dict[str, Any]:
        """Mock connection test."""
        await asyncio.sleep(self.response_delay)
        self.call_count += 1
        
        if self.simulate_errors and self.call_count % 7 == 0:
            raise ConnectionError("Mock Claude Flow connection failure")
        
        return {
            "status": "connected",
            "version": "2.0.0",
            "capabilities": [
                "task_orchestration",
                "agent_management", 
                "swarm_coordination",
                "neural_training",
                "performance_monitoring"
            ],
            "available_agents": len(self.agent_pool),
            "active_swarms": len(self.swarms)
        }
    
    async def initialize_swarm(self, topology: str = "mesh", max_agents: int = 10, strategy: str = "adaptive") -> Dict[str, Any]:
        """Mock swarm initialization."""
        await asyncio.sleep(self.response_delay)
        self.call_count += 1
        
        if self.simulate_errors and topology == "invalid":
            raise ValueError("Invalid topology specified")
        
        swarm_id = f"swarm_{uuid.uuid4().hex[:8]}"
        self.swarms[swarm_id] = {
            "id": swarm_id,
            "topology": topology,
            "max_agents": max_agents,
            "strategy": strategy,
            "created_at": datetime.utcnow().isoformat(),
            "status": "initialized",
            "agents": []
        }
        
        return self.swarms[swarm_id]
    
    async def orchestrate_task(self, task_request: Dict[str, Any]) -> Dict[str, Any]:
        """Mock task orchestration."""
        await asyncio.sleep(self.response_delay)
        self.call_count += 1
        
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        task_description = task_request.get("task", "Mock task")
        
        if self.simulate_errors and "fail" in task_description.lower():
            raise Exception("Mock task orchestration failure")
        
        # Simulate agent assignment
        required_agents = task_request.get("agents", ["general_agent"])
        assigned_agents = []
        
        for agent_type in required_agents:
            agent_id = f"{agent_type}_{uuid.uuid4().hex[:6]}"
            assigned_agents.append({
                "id": agent_id,
                "type": agent_type,
                "status": "assigned",
                "capabilities": self._get_agent_capabilities(agent_type)
            })
        
        task_data = {
            "task_id": task_id,
            "description": task_description,
            "status": "in_progress",
            "agents": assigned_agents,
            "strategy": task_request.get("strategy", "adaptive"),
            "estimated_completion": (datetime.utcnow() + timedelta(minutes=30)).isoformat(),
            "created_at": datetime.utcnow().isoformat(),
            "requirements": task_request.get("requirements", {}),
            "progress": 0,
            "metrics": {
                "agents_assigned": len(assigned_agents),
                "complexity_score": (hash(task_description) % 10) + 1,
                "estimated_duration_minutes": 15 + (hash(task_description) % 45)
            }
        }
        
        self.active_tasks[task_id] = task_data
        return task_data
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Mock task status retrieval."""
        await asyncio.sleep(self.response_delay / 4)  # Status checks are fast
        self.call_count += 1
        
        if task_id not in self.active_tasks:
            raise ValueError(f"Task {task_id} not found")
        
        task = self.active_tasks[task_id].copy()
        
        # Simulate progress
        elapsed_time = (datetime.utcnow() - datetime.fromisoformat(task["created_at"].replace('Z', '+00:00'))).total_seconds()
        estimated_duration = task["metrics"]["estimated_duration_minutes"] * 60
        
        progress = min(100, int((elapsed_time / estimated_duration) * 100))
        task["progress"] = progress
        
        if progress >= 100:
            task["status"] = "completed"
            task["completed_at"] = datetime.utcnow().isoformat()
            task["results"] = self._generate_task_results(task)
        
        return task
    
    async def spawn_agent(self, agent_type: str, capabilities: List[str] = None, config: Dict = None) -> Dict[str, Any]:
        """Mock agent spawning."""
        await asyncio.sleep(self.response_delay)
        self.call_count += 1
        
        agent_id = f"{agent_type}_{uuid.uuid4().hex[:8]}"
        
        agent_data = {
            "id": agent_id,
            "type": agent_type,
            "status": "active",
            "capabilities": capabilities or self._get_agent_capabilities(agent_type),
            "configuration": config or {},
            "spawned_at": datetime.utcnow().isoformat(),
            "performance_metrics": {
                "tasks_completed": 0,
                "success_rate": 1.0,
                "average_response_time_ms": 150 + (hash(agent_id) % 100)
            }
        }
        
        self.agent_pool[agent_id] = agent_data
        return agent_data
    
    async def train_neural_patterns(self, pattern_type: str, training_data: Dict, epochs: int = 50) -> Dict[str, Any]:
        """Mock neural pattern training."""
        training_delay = self.response_delay * (epochs / 10)  # Scale with epochs
        await asyncio.sleep(min(training_delay, 2.0))  # Cap at 2 seconds for tests
        self.call_count += 1
        
        if self.simulate_errors and pattern_type == "invalid_pattern":
            raise ValueError("Invalid pattern type")
        
        training_id = f"training_{uuid.uuid4().hex[:8]}"
        
        # Simulate training progress
        accuracy = 0.60 + (hash(pattern_type) % 35) / 100  # 0.60-0.95
        loss = 1.0 - accuracy + ((hash(training_id) % 20) / 100)  # Inverse correlation with noise
        
        return {
            "training_id": training_id,
            "pattern_type": pattern_type,
            "status": "completed",
            "epochs_completed": epochs,
            "final_accuracy": round(accuracy, 3),
            "final_loss": round(loss, 3),
            "training_time_seconds": training_delay,
            "model_path": f"/models/{pattern_type}_{training_id}.pkl",
            "metrics": {
                "convergence_epoch": max(1, epochs - (hash(training_id) % 10)),
                "best_accuracy": round(accuracy + 0.02, 3),
                "training_data_size": len(str(training_data)),
                "parameters_count": 1000 + (hash(pattern_type) % 9000)
            }
        }
    
    async def run_inference(self, model_path: str, input_data: Dict) -> Dict[str, Any]:
        """Mock neural inference."""
        await asyncio.sleep(self.response_delay / 10)  # Inference is very fast
        self.call_count += 1
        
        inference_id = f"inference_{uuid.uuid4().hex[:8]}"
        
        # Generate mock predictions
        predictions = []
        prediction_types = ["optimization_needed", "refactoring_suggested", "security_check_required", 
                          "performance_warning", "code_quality_issue", "dependency_update"]
        
        for i in range(3):  # Generate 3 predictions
            pred_type = prediction_types[(hash(inference_id) + i) % len(prediction_types)]
            confidence = 0.5 + ((hash(pred_type) + i) % 50) / 100  # 0.5-1.0
            
            predictions.append({
                "pattern": pred_type,
                "confidence": round(confidence, 3),
                "details": f"Mock prediction for {pred_type}",
                "suggested_action": f"Consider addressing {pred_type.replace('_', ' ')}"
            })
        
        return {
            "inference_id": inference_id,
            "model_path": model_path,
            "predictions": predictions,
            "execution_time_ms": (self.response_delay / 10) * 1000,
            "input_size": len(str(input_data)),
            "confidence_score": statistics.mean([p["confidence"] for p in predictions]) if predictions else 0.0
        }
    
    def _get_agent_capabilities(self, agent_type: str) -> List[str]:
        """Get capabilities for agent type."""
        capabilities_map = {
            "researcher": ["web_search", "data_analysis", "report_generation"],
            "coder": ["code_generation", "debugging", "refactoring", "testing"],
            "analyst": ["data_analysis", "pattern_recognition", "reporting"],
            "optimizer": ["performance_analysis", "code_optimization", "resource_management"],
            "coordinator": ["task_management", "agent_coordination", "workflow_orchestration"],
            "tester": ["test_generation", "test_execution", "quality_assurance"],
            "architect": ["system_design", "architecture_planning", "documentation"],
            "security_specialist": ["security_analysis", "vulnerability_assessment", "compliance_check"],
            "database_architect": ["schema_design", "query_optimization", "data_modeling"],
            "frontend_developer": ["ui_development", "user_experience", "responsive_design"],
            "backend_developer": ["api_development", "server_architecture", "database_integration"]
        }
        
        return capabilities_map.get(agent_type, ["general_purpose", "task_execution"])
    
    def _generate_task_results(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock task results."""
        task_type = task.get("description", "").lower()
        
        base_results = {
            "completion_status": "success",
            "quality_score": 0.8 + ((hash(task["task_id"]) % 20) / 100),
            "execution_time_minutes": 15 + (hash(task["task_id"]) % 30),
            "agents_utilized": len(task["agents"])
        }
        
        if "code" in task_type:
            base_results.update({
                "files_created": 3 + (hash(task["task_id"]) % 7),
                "lines_of_code": 150 + (hash(task["task_id"]) % 500),
                "test_coverage_percent": 75 + (hash(task["task_id"]) % 25)
            })
        elif "analysis" in task_type:
            base_results.update({
                "data_points_analyzed": 1000 + (hash(task["task_id"]) % 9000),
                "patterns_identified": 5 + (hash(task["task_id"]) % 15),
                "accuracy_score": 0.85 + ((hash(task["task_id"]) % 15) / 100)
            })
        elif "test" in task_type:
            base_results.update({
                "tests_created": 25 + (hash(task["task_id"]) % 75),
                "tests_passed": 90 + (hash(task["task_id"]) % 10),
                "coverage_increase_percent": 15 + (hash(task["task_id"]) % 25)
            })
        
        return base_results


class MockDatabaseService:
    """Mock for database service."""
    
    def __init__(self, simulate_errors: bool = False, connection_delay: float = 0.05):
        self.simulate_errors = simulate_errors
        self.connection_delay = connection_delay
        self.connected = False
        self.query_count = 0
        self.mock_data = {}
        
    async def connect(self) -> bool:
        """Mock database connection."""
        await asyncio.sleep(self.connection_delay)
        
        if self.simulate_errors and self.query_count % 20 == 0:
            raise ConnectionError("Mock database connection failed")
        
        self.connected = True
        return True
    
    async def execute_query(self, query: str, params: Dict = None) -> Dict[str, Any]:
        """Mock query execution."""
        await asyncio.sleep(self.connection_delay / 2)
        self.query_count += 1
        
        if not self.connected:
            raise ConnectionError("Database not connected")
        
        if self.simulate_errors and "DROP" in query.upper():
            raise Exception("Mock SQL error: Permission denied")
        
        # Simulate different query types
        query_upper = query.upper().strip()
        
        if query_upper.startswith("SELECT"):
            return {
                "rows": self._generate_mock_rows(query, params),
                "row_count": 1 + (hash(query) % 10),
                "execution_time_ms": self.connection_delay * 500,
                "query": query
            }
        elif query_upper.startswith("INSERT"):
            row_id = str(uuid.uuid4())
            return {
                "inserted_id": row_id,
                "rows_affected": 1,
                "execution_time_ms": self.connection_delay * 300,
                "query": query
            }
        elif query_upper.startswith("UPDATE"):
            return {
                "rows_affected": 1 + (hash(query) % 5),
                "execution_time_ms": self.connection_delay * 400,
                "query": query
            }
        elif query_upper.startswith("DELETE"):
            return {
                "rows_affected": hash(query) % 3,
                "execution_time_ms": self.connection_delay * 200,
                "query": query
            }
        else:
            return {
                "status": "executed",
                "execution_time_ms": self.connection_delay * 100,
                "query": query
            }
    
    async def close(self) -> None:
        """Mock database close."""
        await asyncio.sleep(self.connection_delay / 4)
        self.connected = False
    
    def _generate_mock_rows(self, query: str, params: Dict = None) -> List[Dict]:
        """Generate mock query results."""
        if "users" in query.lower():
            return [{
                "id": str(uuid.uuid4()),
                "email": "mock@example.com",
                "username": "mockuser",
                "created_at": datetime.utcnow().isoformat()
            }]
        elif "projects" in query.lower():
            return [{
                "id": str(uuid.uuid4()),
                "name": "Mock Project",
                "status": "active",
                "created_at": datetime.utcnow().isoformat()
            }]
        else:
            return [{
                "id": hash(query) % 1000,
                "value": f"mock_result_{hash(query) % 100}",
                "timestamp": datetime.utcnow().isoformat()
            }]


class MockEmailService:
    """Mock for email service."""
    
    def __init__(self, simulate_errors: bool = False, send_delay: float = 0.1):
        self.simulate_errors = simulate_errors
        self.send_delay = send_delay
        self.sent_emails = []
        self.bounce_rate = 0.02  # 2% bounce rate
        
    async def send_email(self, to: str, subject: str, content: str, template: str = None) -> Dict[str, Any]:
        """Mock email sending."""
        await asyncio.sleep(self.send_delay)
        
        if self.simulate_errors and "fail@" in to:
            raise Exception("Mock email sending failure")
        
        # Simulate bounce
        bounced = hash(to) % 100 < (self.bounce_rate * 100)
        
        email_id = f"email_{uuid.uuid4().hex[:8]}"
        email_data = {
            "email_id": email_id,
            "to": to,
            "subject": subject,
            "status": "bounced" if bounced else "sent",
            "sent_at": datetime.utcnow().isoformat(),
            "template": template,
            "content_length": len(content)
        }
        
        self.sent_emails.append(email_data)
        return email_data
    
    async def send_bulk_email(self, recipients: List[str], subject: str, content: str) -> Dict[str, Any]:
        """Mock bulk email sending."""
        total_delay = self.send_delay * min(len(recipients) / 10, 2.0)  # Bulk is more efficient
        await asyncio.sleep(total_delay)
        
        results = []
        for recipient in recipients:
            email_result = await self.send_email(recipient, subject, content)
            results.append(email_result)
        
        successful = sum(1 for r in results if r["status"] == "sent")
        bounced = len(results) - successful
        
        return {
            "bulk_id": f"bulk_{uuid.uuid4().hex[:8]}",
            "total_recipients": len(recipients),
            "successful_sends": successful,
            "bounced_emails": bounced,
            "send_rate": successful / len(recipients) if recipients else 0,
            "results": results
        }


class MockOAuthProvider:
    """Mock for OAuth providers (Google, GitHub, etc.)."""
    
    def __init__(self, provider_name: str = "mock_oauth", simulate_errors: bool = False):
        self.provider_name = provider_name
        self.simulate_errors = simulate_errors
        self.issued_tokens = {}
        
    async def authorize_url(self, client_id: str, redirect_uri: str, scope: str = "read") -> str:
        """Mock OAuth authorization URL generation."""
        if self.simulate_errors and "invalid" in client_id:
            raise ValueError("Invalid client ID")
        
        state = uuid.uuid4().hex
        return f"https://mock-oauth.com/authorize?client_id={client_id}&redirect_uri={redirect_uri}&scope={scope}&state={state}"
    
    async def exchange_code(self, code: str, client_id: str, client_secret: str) -> Dict[str, Any]:
        """Mock OAuth code exchange."""
        await asyncio.sleep(0.2)  # OAuth exchanges are typically slower
        
        if self.simulate_errors and code == "invalid_code":
            raise Exception("Invalid authorization code")
        
        access_token = f"mock_access_token_{uuid.uuid4().hex[:16]}"
        refresh_token = f"mock_refresh_token_{uuid.uuid4().hex[:16]}"
        
        token_data = {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "Bearer",
            "expires_in": 3600,
            "scope": "read write",
            "issued_at": datetime.utcnow().isoformat()
        }
        
        self.issued_tokens[access_token] = token_data
        return token_data
    
    async def get_user_info(self, access_token: str) -> Dict[str, Any]:
        """Mock user info retrieval."""
        await asyncio.sleep(0.1)
        
        if access_token not in self.issued_tokens:
            raise Exception("Invalid access token")
        
        # Generate consistent mock user data based on token
        user_id = hash(access_token) % 1000000
        
        return {
            "id": str(user_id),
            "email": f"mockuser{user_id}@{self.provider_name}.com",
            "name": f"Mock User {user_id}",
            "avatar_url": f"https://avatars.{self.provider_name}.com/u/{user_id}",
            "provider": self.provider_name,
            "verified": True,
            "locale": "en",
            "created_at": (datetime.utcnow() - timedelta(days=hash(access_token) % 365)).isoformat()
        }


class MockPaymentProcessor:
    """Mock for payment processing services."""
    
    def __init__(self, simulate_errors: bool = False, processing_delay: float = 0.5):
        self.simulate_errors = simulate_errors
        self.processing_delay = processing_delay
        self.transactions = {}
        self.decline_rate = 0.05  # 5% decline rate
        
    async def process_payment(self, amount: float, currency: str = "USD", payment_method: Dict = None) -> Dict[str, Any]:
        """Mock payment processing."""
        await asyncio.sleep(self.processing_delay)
        
        if self.simulate_errors and amount < 0:
            raise ValueError("Invalid payment amount")
        
        # Simulate payment decline
        declined = hash(str(amount) + currency) % 100 < (self.decline_rate * 100)
        
        transaction_id = f"txn_{uuid.uuid4().hex[:12]}"
        
        if declined:
            status = "declined"
            decline_reasons = ["insufficient_funds", "card_expired", "security_check_failed", "limit_exceeded"]
            decline_reason = decline_reasons[hash(transaction_id) % len(decline_reasons)]
        else:
            status = "completed"
            decline_reason = None
        
        transaction_data = {
            "transaction_id": transaction_id,
            "amount": amount,
            "currency": currency,
            "status": status,
            "decline_reason": decline_reason,
            "processed_at": datetime.utcnow().isoformat(),
            "processing_fee": round(amount * 0.029 + 0.30, 2) if status == "completed" else 0,
            "payment_method": payment_method or {"type": "card", "last4": "1234"}
        }
        
        self.transactions[transaction_id] = transaction_data
        return transaction_data
    
    async def refund_payment(self, transaction_id: str, amount: float = None) -> Dict[str, Any]:
        """Mock payment refund."""
        await asyncio.sleep(self.processing_delay / 2)
        
        if transaction_id not in self.transactions:
            raise ValueError("Transaction not found")
        
        original_txn = self.transactions[transaction_id]
        if original_txn["status"] != "completed":
            raise ValueError("Cannot refund non-completed transaction")
        
        refund_amount = amount or original_txn["amount"]
        if refund_amount > original_txn["amount"]:
            raise ValueError("Refund amount exceeds original transaction")
        
        refund_id = f"refund_{uuid.uuid4().hex[:12]}"
        
        refund_data = {
            "refund_id": refund_id,
            "original_transaction_id": transaction_id,
            "amount": refund_amount,
            "currency": original_txn["currency"],
            "status": "completed",
            "processed_at": datetime.utcnow().isoformat(),
            "refund_fee": 0  # Usually no fee for refunds
        }
        
        return refund_data


class MockFileStorageService:
    """Mock for cloud file storage services."""
    
    def __init__(self, simulate_errors: bool = False, upload_delay: float = 0.1):
        self.simulate_errors = simulate_errors
        self.upload_delay = upload_delay
        self.stored_files = {}
        
    async def upload_file(self, file_content: bytes, file_path: str, content_type: str = "application/octet-stream") -> Dict[str, Any]:
        """Mock file upload."""
        # Scale delay with file size (simulate real upload)
        scaled_delay = min(self.upload_delay * (len(file_content) / 1024 / 1024), 2.0)  # Max 2s for tests
        await asyncio.sleep(scaled_delay)
        
        if self.simulate_errors and "forbidden" in file_path:
            raise PermissionError("Upload forbidden")
        
        file_id = f"file_{uuid.uuid4().hex[:12]}"
        file_url = f"https://mock-storage.com/{file_id}/{file_path}"
        
        file_data = {
            "file_id": file_id,
            "file_path": file_path,
            "file_url": file_url,
            "size_bytes": len(file_content),
            "content_type": content_type,
            "uploaded_at": datetime.utcnow().isoformat(),
            "etag": str(hash(file_content)),
            "checksum": str(abs(hash(file_content)))
        }
        
        self.stored_files[file_id] = {**file_data, "content": file_content}
        return file_data
    
    async def download_file(self, file_id: str) -> Dict[str, Any]:
        """Mock file download."""
        await asyncio.sleep(self.upload_delay / 2)
        
        if file_id not in self.stored_files:
            raise FileNotFoundError("File not found")
        
        file_data = self.stored_files[file_id]
        
        return {
            "file_id": file_id,
            "content": file_data["content"],
            "metadata": {k: v for k, v in file_data.items() if k != "content"}
        }
    
    async def delete_file(self, file_id: str) -> Dict[str, Any]:
        """Mock file deletion."""
        await asyncio.sleep(self.upload_delay / 4)
        
        if file_id not in self.stored_files:
            raise FileNotFoundError("File not found")
        
        deleted_file = self.stored_files.pop(file_id)
        
        return {
            "file_id": file_id,
            "status": "deleted",
            "deleted_at": datetime.utcnow().isoformat(),
            "file_path": deleted_file["file_path"]
        }


# Pytest fixtures for external service mocks
@pytest.fixture
def mock_claude_code():
    """Provide mock Claude Code integration."""
    return MockClaudeCodeIntegration()


@pytest.fixture
def mock_claude_code_with_errors():
    """Provide mock Claude Code integration that simulates errors."""
    return MockClaudeCodeIntegration(simulate_errors=True)


@pytest.fixture
def mock_claude_flow():
    """Provide mock Claude Flow integration."""
    return MockClaudeFlowIntegration()


@pytest.fixture
def mock_claude_flow_with_errors():
    """Provide mock Claude Flow integration that simulates errors."""
    return MockClaudeFlowIntegration(simulate_errors=True)


@pytest.fixture
def mock_database():
    """Provide mock database service."""
    return MockDatabaseService()


@pytest.fixture
def mock_database_with_errors():
    """Provide mock database service that simulates errors."""
    return MockDatabaseService(simulate_errors=True)


@pytest.fixture
def mock_email_service():
    """Provide mock email service."""
    return MockEmailService()


@pytest.fixture
def mock_oauth_google():
    """Provide mock Google OAuth provider."""
    return MockOAuthProvider("google")


@pytest.fixture
def mock_oauth_github():
    """Provide mock GitHub OAuth provider."""
    return MockOAuthProvider("github")


@pytest.fixture
def mock_payment_processor():
    """Provide mock payment processor."""
    return MockPaymentProcessor()


@pytest.fixture
def mock_file_storage():
    """Provide mock file storage service."""
    return MockFileStorageService()


# Helper function to create multiple service mocks
def create_mock_service_suite(simulate_errors: bool = False) -> Dict[str, Any]:
    """Create a complete suite of mock services."""
    return {
        "claude_code": MockClaudeCodeIntegration(simulate_errors=simulate_errors),
        "claude_flow": MockClaudeFlowIntegration(simulate_errors=simulate_errors),
        "database": MockDatabaseService(simulate_errors=simulate_errors),
        "email": MockEmailService(simulate_errors=simulate_errors),
        "oauth_google": MockOAuthProvider("google", simulate_errors=simulate_errors),
        "oauth_github": MockOAuthProvider("github", simulate_errors=simulate_errors),
        "payment": MockPaymentProcessor(simulate_errors=simulate_errors),
        "file_storage": MockFileStorageService(simulate_errors=simulate_errors)
    }


# Import statistics for use in mock classes
try:
    import statistics
except ImportError:
    # Fallback implementation for basic statistics
    class MockStatistics:
        @staticmethod
        def mean(values):
            return sum(values) / len(values) if values else 0
    
    statistics = MockStatistics()