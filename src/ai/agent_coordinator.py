"""Agent Coordinator Module

Advanced multi-agent coordination and communication system with:
- Inter-agent communication and message routing
- Task assignment and load balancing
- Result aggregation and consensus mechanisms
- Agent lifecycle management and health monitoring
- Dynamic capability matching and resource allocation
"""

import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable, Set
import weakref
from concurrent.futures import ThreadPoolExecutor

from ..integrations.claude_flow import Agent, AgentType
from ..core.exceptions import CoordinationError, AgentError, CommunicationError

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of inter-agent messages"""
    TASK_ASSIGNMENT = "task_assignment"
    TASK_COMPLETION = "task_completion"
    TASK_UPDATE = "task_update"
    CAPABILITY_QUERY = "capability_query"
    CAPABILITY_RESPONSE = "capability_response"
    RESOURCE_REQUEST = "resource_request"
    RESOURCE_RESPONSE = "resource_response"
    STATUS_UPDATE = "status_update"
    ERROR_REPORT = "error_report"
    COORDINATION_REQUEST = "coordination_request"
    CONSENSUS_PROPOSAL = "consensus_proposal"
    CONSENSUS_VOTE = "consensus_vote"
    HEARTBEAT = "heartbeat"


class AgentStatus(Enum):
    """Enhanced agent status tracking"""
    INITIALIZING = "initializing"
    IDLE = "idle"
    ACTIVE = "active"
    BUSY = "busy"
    OVERLOADED = "overloaded"
    ERROR = "error"
    OFFLINE = "offline"
    TERMINATING = "terminating"


class ConsensusType(Enum):
    """Types of consensus mechanisms"""
    MAJORITY = "majority"
    UNANIMOUS = "unanimous"
    WEIGHTED = "weighted"
    LEADER_BASED = "leader_based"


@dataclass
class Message:
    """Inter-agent message structure"""
    id: str
    type: MessageType
    sender_id: str
    recipient_id: Optional[str]  # None for broadcast
    content: Dict[str, Any]
    priority: int = 5  # 1-10, higher is more priority
    timestamp: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    correlation_id: Optional[str] = None
    requires_response: bool = False
    response_timeout: int = 30  # seconds


@dataclass
class AgentCapability:
    """Agent capability definition"""
    name: str
    level: int  # 1-10 proficiency level
    resource_cost: float  # Resource units per operation
    average_time: float  # Average time in seconds
    success_rate: float  # Historical success rate 0-1
    last_used: Optional[datetime] = None
    usage_count: int = 0


@dataclass
class TaskAssignment:
    """Task assignment tracking"""
    id: str
    agent_id: str
    task_description: str
    assigned_at: datetime
    deadline: Optional[datetime] = None
    status: str = "assigned"
    priority: int = 5
    estimated_effort: float = 1.0
    actual_effort: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


@dataclass
class ConsensusProposal:
    """Consensus proposal for group decision making"""
    id: str
    proposer_id: str
    type: ConsensusType
    proposal: Dict[str, Any]
    votes: Dict[str, bool] = field(default_factory=dict)  # agent_id -> vote
    weights: Dict[str, float] = field(default_factory=dict)  # agent_id -> weight
    deadline: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(minutes=5))
    min_participants: int = 1
    status: str = "pending"  # pending, approved, rejected, expired
    created_at: datetime = field(default_factory=datetime.utcnow)


class AgentCoordinator:
    """
    Advanced Agent Coordinator for multi-agent systems
    
    Features:
    - Intelligent task assignment and load balancing
    - Inter-agent communication and message routing
    - Capability-based agent selection
    - Consensus mechanisms for group decisions
    - Resource allocation and conflict resolution
    - Performance monitoring and optimization
    - Fault tolerance and recovery
    """
    
    def __init__(
        self,
        max_message_queue_size: int = 1000,
        message_retention_hours: int = 24,
        enable_consensus: bool = True,
        heartbeat_interval: int = 30,
        load_balancing_algorithm: str = "capability_weighted"
    ):
        self.max_message_queue_size = max_message_queue_size
        self.message_retention_hours = message_retention_hours
        self.enable_consensus = enable_consensus
        self.heartbeat_interval = heartbeat_interval
        self.load_balancing_algorithm = load_balancing_algorithm
        
        # Agent tracking
        self.agents: Dict[str, Agent] = {}
        self.agent_status: Dict[str, AgentStatus] = {}
        self.agent_capabilities: Dict[str, List[AgentCapability]] = {}
        self.agent_metrics: Dict[str, Dict[str, Any]] = {}
        self.agent_last_seen: Dict[str, datetime] = {}
        
        # Communication system
        self.message_queues: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.max_message_queue_size))
        self.message_history: deque = deque(maxlen=self.max_message_queue_size * 10)
        self.broadcast_queue: deque = deque(maxlen=self.max_message_queue_size)
        self.pending_responses: Dict[str, Dict[str, Any]] = {}  # message_id -> response_info
        
        # Task management
        self.active_assignments: Dict[str, TaskAssignment] = {}
        self.assignment_history: List[TaskAssignment] = []
        self.task_queue: deque = deque()  # Pending task assignments
        
        # Consensus system
        self.active_proposals: Dict[str, ConsensusProposal] = {}
        self.consensus_history: List[ConsensusProposal] = []
        
        # Performance and monitoring
        self.coordination_metrics = {
            'messages_sent': 0,
            'messages_delivered': 0,
            'messages_failed': 0,
            'tasks_assigned': 0,
            'tasks_completed': 0,
            'consensus_proposals': 0,
            'consensus_successful': 0
        }
        
        # Background tasks
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.message_processor_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        
        # Thread pool for CPU-intensive operations
        self.thread_pool = ThreadPoolExecutor(max_workers=2)
        
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging for agent coordination"""
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    async def start(self):
        """Start the coordination system"""
        logger.info("Starting agent coordination system")
        
        # Start background tasks
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self.message_processor_task = asyncio.create_task(self._message_processor_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("Agent coordination system started")
    
    async def stop(self):
        """Stop the coordination system"""
        logger.info("Stopping agent coordination system")
        
        # Cancel background tasks
        for task in [self.heartbeat_task, self.message_processor_task, self.cleanup_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        logger.info("Agent coordination system stopped")
        
    def get_coordination_metrics(self) -> Dict[str, Any]:
        """Get overall coordination system metrics"""
        
        total_agents = len(self.agents)
        active_agents = len([s for s in self.agent_status.values() if s in [AgentStatus.ACTIVE, AgentStatus.BUSY]])
        
        # Calculate average metrics across agents
        avg_success_rate = 0.0
        avg_response_time = 0.0
        
        if self.agent_metrics:
            success_rates = [m.get('success_rate', 0.0) for m in self.agent_metrics.values()]
            response_times = [m.get('average_response_time', 0.0) for m in self.agent_metrics.values()]
            
            avg_success_rate = sum(success_rates) / len(success_rates) if success_rates else 0.0
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0.0
        
        return {
            **self.coordination_metrics,
            'total_agents': total_agents,
            'active_agents': active_agents,
            'idle_agents': len([s for s in self.agent_status.values() if s == AgentStatus.IDLE]),
            'busy_agents': len([s for s in self.agent_status.values() if s == AgentStatus.BUSY]),
            'overloaded_agents': len([s for s in self.agent_status.values() if s == AgentStatus.OVERLOADED]),
            'error_agents': len([s for s in self.agent_status.values() if s == AgentStatus.ERROR]),
            'active_assignments': len(self.active_assignments),
            'pending_messages': sum(len(q) for q in self.message_queues.values()),
            'active_proposals': len(self.active_proposals),
            'average_success_rate': avg_success_rate,
            'average_response_time': avg_response_time,
            'message_success_rate': (
                self.coordination_metrics['messages_delivered'] / 
                max(self.coordination_metrics['messages_sent'], 1)
            ) * 100
        }
    
    def _generate_message_id(self) -> str:
        """Generate unique message ID"""
        timestamp = int(time.time() * 1000000)  # microseconds
        return f"msg-{timestamp}-{uuid.uuid4().hex[:8]}"
    
    async def _heartbeat_loop(self):
        """Background heartbeat loop"""
        
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval)
            except Exception as e:
                logger.error(f"Heartbeat loop error: {e}")
                await asyncio.sleep(self.heartbeat_interval)
    
    async def _message_processor_loop(self):
        """Background message processing loop"""
        
        while True:
            try:
                await asyncio.sleep(5)  # Check every 5 seconds
            except Exception as e:
                logger.error(f"Message processor loop error: {e}")
                await asyncio.sleep(5)
    
    async def _cleanup_loop(self):
        """Background cleanup loop"""
        
        while True:
            try:
                await self._cleanup_expired_messages()
                await self._cleanup_completed_assignments()
                await self._cleanup_expired_proposals()
                await asyncio.sleep(300)  # Clean every 5 minutes
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(300)
    
    async def register_agent(self, agent: Agent) -> bool:
        """Register an agent with the coordinator"""
        
        try:
            agent_id = agent.id
            
            # Store agent reference
            self.agents[agent_id] = agent
            self.agent_status[agent_id] = AgentStatus.INITIALIZING
            self.agent_last_seen[agent_id] = datetime.utcnow()
            
            # Initialize capabilities
            capabilities = []
            for capability_name in agent.capabilities:
                capability = AgentCapability(
                    name=capability_name,
                    level=8,  # Default high proficiency
                    resource_cost=1.0,
                    average_time=10.0,
                    success_rate=0.95
                )
                capabilities.append(capability)
            
            self.agent_capabilities[agent_id] = capabilities
            
            # Initialize metrics
            self.agent_metrics[agent_id] = {
                'tasks_assigned': 0,
                'tasks_completed': 0,
                'tasks_failed': 0,
                'total_response_time': 0.0,
                'average_response_time': 0.0,
                'success_rate': 1.0,
                'last_activity': datetime.utcnow().isoformat()
            }
            
            # Send welcome message
            welcome_msg = Message(
                id=self._generate_message_id(),
                type=MessageType.STATUS_UPDATE,
                sender_id="coordinator",
                recipient_id=agent_id,
                content={'status': 'registered', 'welcome': True}
            )
            
            await self.send_message(welcome_msg)
            
            logger.info(f"Agent {agent_id} registered successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register agent {agent.id}: {e}")
            return False
    
    async def send_message(self, message: Message) -> bool:
        """Send message to agent(s)"""
        
        try:
            # Validate message
            if message.expires_at and datetime.utcnow() > message.expires_at:
                logger.warning(f"Message {message.id} expired before sending")
                return False
            
            # Store in history
            self.message_history.append(message)
            
            if message.recipient_id is None:
                # Broadcast message
                self.broadcast_queue.append(message)
                success_count = 0
                
                for agent_id in self.agents.keys():
                    if await self._deliver_message(agent_id, message):
                        success_count += 1
                
                self.coordination_metrics['messages_sent'] += 1
                self.coordination_metrics['messages_delivered'] += success_count
                
                return success_count > 0
            else:
                # Direct message
                self.coordination_metrics['messages_sent'] += 1
                success = await self._deliver_message(message.recipient_id, message)
                
                if success:
                    self.coordination_metrics['messages_delivered'] += 1
                else:
                    self.coordination_metrics['messages_failed'] += 1
                
                return success
                
        except Exception as e:
            logger.error(f"Failed to send message {message.id}: {e}")
            self.coordination_metrics['messages_failed'] += 1
            return False
    
    async def _deliver_message(self, agent_id: str, message: Message) -> bool:
        """Deliver message to specific agent"""
        
        try:
            if agent_id not in self.agents:
                return False
            
            # Add to agent's queue
            self.message_queues[agent_id].append(message)
            
            # Handle response requirement
            if message.requires_response:
                self.pending_responses[message.id] = {
                    'agent_id': agent_id,
                    'timeout': datetime.utcnow() + timedelta(seconds=message.response_timeout),
                    'correlation_id': message.correlation_id
                }
            
            # Update agent last seen
            self.agent_last_seen[agent_id] = datetime.utcnow()
            
            return True
            
        except Exception as e:
            logger.error(f"Message delivery failed to agent {agent_id}: {e}")
            return False
    
    async def assign_task(self, task_description: str, requirements: List[str], 
                         priority: int = 5) -> Optional[str]:
        """Assign task to best available agent"""
        
        try:
            # Find suitable agents
            suitable_agents = await self._find_suitable_agents(requirements)
            
            if not suitable_agents:
                logger.warning(f"No suitable agents found for task: {task_description[:50]}...")
                return None
            
            # Select best agent based on load balancing algorithm
            best_agent = await self._select_best_agent(suitable_agents, requirements)
            
            # Create task assignment
            assignment_id = f"task-{uuid.uuid4().hex[:8]}"
            assignment = TaskAssignment(
                id=assignment_id,
                agent_id=best_agent,
                task_description=task_description,
                assigned_at=datetime.utcnow(),
                deadline=datetime.utcnow() + timedelta(minutes=30),  # Default 30min deadline
                priority=priority,
                estimated_effort=self._estimate_task_effort(task_description, requirements)
            )
            
            # Store assignment
            self.active_assignments[assignment_id] = assignment
            
            # Send assignment message
            assignment_msg = Message(
                id=self._generate_message_id(),
                type=MessageType.TASK_ASSIGNMENT,
                sender_id="coordinator",
                recipient_id=best_agent,
                content={
                    'assignment_id': assignment_id,
                    'task_description': task_description,
                    'requirements': requirements,
                    'priority': priority,
                    'deadline': assignment.deadline.isoformat(),
                    'estimated_effort': assignment.estimated_effort
                },
                priority=priority,
                requires_response=True,
                response_timeout=300  # 5 minutes
            )
            
            success = await self.send_message(assignment_msg)
            
            if success:
                # Update metrics
                self.coordination_metrics['tasks_assigned'] += 1
                
                if best_agent in self.agent_metrics:
                    self.agent_metrics[best_agent]['tasks_assigned'] += 1
                
                logger.info(f"Task {assignment_id} assigned to agent {best_agent}")
                return assignment_id
            else:
                # Cleanup failed assignment
                self.active_assignments.pop(assignment_id, None)
                return None
                
        except Exception as e:
            logger.error(f"Task assignment failed: {e}")
            return None
    
    async def propose_consensus(self, proposer_id: str, proposal: Dict[str, Any], 
                              consensus_type: ConsensusType = ConsensusType.MAJORITY,
                              participants: Optional[List[str]] = None,
                              deadline_minutes: int = 5) -> str:
        """Propose consensus decision"""
        
        proposal_id = f"consensus-{uuid.uuid4().hex[:8]}"
        
        try:
            # Determine participants
            if participants is None:
                participants = list(self.agents.keys())
            
            # Validate participants
            valid_participants = [p for p in participants if p in self.agents]
            
            if not valid_participants:
                raise CommunicationError("No valid participants for consensus")
            
            # Create consensus proposal
            consensus_proposal = ConsensusProposal(
                id=proposal_id,
                proposer_id=proposer_id,
                type=consensus_type,
                proposal=proposal,
                deadline=datetime.utcnow() + timedelta(minutes=deadline_minutes),
                min_participants=max(1, len(valid_participants) // 2 + 1)
            )
            
            # Store proposal
            self.active_proposals[proposal_id] = consensus_proposal
            
            # Send proposal to participants
            proposal_msg = Message(
                id=self._generate_message_id(),
                type=MessageType.CONSENSUS_PROPOSAL,
                sender_id=proposer_id,
                recipient_id=None,  # Broadcast
                content={
                    'proposal_id': proposal_id,
                    'proposal': proposal,
                    'consensus_type': consensus_type.value,
                    'deadline': consensus_proposal.deadline.isoformat(),
                    'participants': valid_participants
                },
                priority=8,  # High priority for consensus
                requires_response=True,
                response_timeout=deadline_minutes * 60
            )
            
            # Send only to participants
            success_count = 0
            for participant_id in valid_participants:
                if await self._deliver_message(participant_id, proposal_msg):
                    success_count += 1
            
            if success_count > 0:
                self.coordination_metrics['consensus_proposals'] += 1
                logger.info(f"Consensus proposal {proposal_id} sent to {success_count} participants")
                return proposal_id
            else:
                # Cleanup failed proposal
                self.active_proposals.pop(proposal_id, None)
                raise CommunicationError("Failed to send consensus proposal to any participants")
                
        except Exception as e:
            logger.error(f"Consensus proposal failed: {e}")
            raise CommunicationError(f"Consensus proposal failed: {e}") from e
    
    async def vote_on_proposal(self, agent_id: str, proposal_id: str, 
                              vote: bool, weight: float = 1.0) -> bool:
        """Submit vote for consensus proposal"""
        
        try:
            if proposal_id not in self.active_proposals:
                logger.warning(f"Proposal {proposal_id} not found for vote from {agent_id}")
                return False
            
            proposal = self.active_proposals[proposal_id]
            
            # Check deadline
            if datetime.utcnow() > proposal.deadline:
                logger.warning(f"Vote from {agent_id} rejected - proposal {proposal_id} expired")
                return False
            
            # Record vote
            proposal.votes[agent_id] = vote
            proposal.weights[agent_id] = weight
            
            # Send vote confirmation
            vote_msg = Message(
                id=self._generate_message_id(),
                type=MessageType.CONSENSUS_VOTE,
                sender_id=agent_id,
                recipient_id=proposal.proposer_id,
                content={
                    'proposal_id': proposal_id,
                    'vote': vote,
                    'weight': weight,
                    'voter_id': agent_id
                },
                correlation_id=proposal_id
            )
            
            await self.send_message(vote_msg)
            
            # Check if consensus is reached
            await self._evaluate_consensus(proposal_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Vote submission failed: {e}")
            return False
    
    async def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Get comprehensive agent status"""
        
        if agent_id not in self.agents:
            raise AgentError(f"Agent {agent_id} not found")
        
        agent = self.agents[agent_id]
        status = self.agent_status[agent_id]
        capabilities = self.agent_capabilities.get(agent_id, [])
        metrics = self.agent_metrics.get(agent_id, {})
        
        # Get active assignments
        active_assignments = [
            {
                'assignment_id': assignment.id,
                'task_description': assignment.task_description[:100],
                'priority': assignment.priority,
                'assigned_at': assignment.assigned_at.isoformat(),
                'deadline': assignment.deadline.isoformat() if assignment.deadline else None,
                'status': assignment.status
            }
            for assignment in self.active_assignments.values()
            if assignment.agent_id == agent_id
        ]
        
        # Get recent messages
        recent_messages = [
            {
                'message_id': msg.id,
                'type': msg.type.value,
                'sender_id': msg.sender_id,
                'timestamp': msg.timestamp.isoformat(),
                'priority': msg.priority
            }
            for msg in list(self.message_queues[agent_id])[-5:]  # Last 5 messages
        ]
        
        return {
            'agent_id': agent_id,
            'name': agent.name,
            'type': agent.type.value,
            'status': status.value,
            'capabilities': [cap.name for cap in capabilities],
            'last_seen': self.agent_last_seen.get(agent_id, datetime.utcnow()).isoformat(),
            'metrics': metrics,
            'active_assignments': active_assignments,
            'pending_messages': len(self.message_queues[agent_id]),
            'recent_messages': recent_messages,
            'uptime_seconds': (datetime.utcnow() - agent.created_at).total_seconds() if agent.created_at else 0
        }
    
    async def _find_suitable_agents(self, requirements: List[str]) -> List[str]:
        """Find agents suitable for task requirements"""
        
        suitable_agents = []
        
        for agent_id, capabilities in self.agent_capabilities.items():
            # Skip unavailable agents
            if self.agent_status.get(agent_id) not in [AgentStatus.IDLE, AgentStatus.ACTIVE]:
                continue
            
            # Check capability matching
            capability_names = [cap.name.lower() for cap in capabilities]
            matches = 0
            
            for requirement in requirements:
                req_lower = requirement.lower()
                for cap_name in capability_names:
                    if req_lower in cap_name or cap_name in req_lower:
                        matches += 1
                        break
            
            # Agent is suitable if it matches at least half the requirements
            if matches >= len(requirements) / 2:
                suitable_agents.append(agent_id)
        
        return suitable_agents
    
    async def _select_best_agent(self, suitable_agents: List[str], 
                                requirements: List[str]) -> str:
        """Select best agent using load balancing algorithm"""
        
        if self.load_balancing_algorithm == "round_robin":
            return suitable_agents[0]  # Simplified round robin
        
        elif self.load_balancing_algorithm == "least_loaded":
            # Select agent with least active assignments
            agent_loads = {}
            
            for agent_id in suitable_agents:
                load = len([a for a in self.active_assignments.values() if a.agent_id == agent_id])
                agent_loads[agent_id] = load
            
            return min(agent_loads.items(), key=lambda x: x[1])[0]
        
        elif self.load_balancing_algorithm == "capability_weighted":
            # Select based on capability scores and availability
            best_agent = None
            best_score = -1
            
            for agent_id in suitable_agents:
                # Calculate capability score
                capabilities = self.agent_capabilities.get(agent_id, [])
                capability_score = 0
                
                for requirement in requirements:
                    req_lower = requirement.lower()
                    for cap in capabilities:
                        if req_lower in cap.name.lower() or cap.name.lower() in req_lower:
                            capability_score += cap.level * cap.success_rate
                            break
                
                # Factor in current load (lower is better)
                current_load = len([a for a in self.active_assignments.values() if a.agent_id == agent_id])
                load_factor = 1.0 / (1 + current_load)
                
                # Combined score
                total_score = capability_score * load_factor
                
                if total_score > best_score:
                    best_score = total_score
                    best_agent = agent_id
            
            return best_agent or suitable_agents[0]
        
        else:
            return suitable_agents[0]  # Default fallback
    
    def _estimate_task_effort(self, task_description: str, requirements: List[str]) -> float:
        """Estimate task effort in resource units"""
        
        base_effort = 1.0
        
        # Adjust based on description length (proxy for complexity)
        description_factor = min(3.0, len(task_description) / 100)
        
        # Adjust based on number of requirements
        requirements_factor = 1 + (len(requirements) * 0.2)
        
        return base_effort * description_factor * requirements_factor
    
    async def _evaluate_consensus(self, proposal_id: str):
        """Evaluate if consensus has been reached"""
        
        try:
            if proposal_id not in self.active_proposals:
                return
            
            proposal = self.active_proposals[proposal_id]
            
            # Check if enough votes collected
            total_votes = len(proposal.votes)
            if total_votes < proposal.min_participants:
                return  # Not enough votes yet
            
            # Calculate consensus based on type
            consensus_reached = False
            
            if proposal.type == ConsensusType.MAJORITY:
                yes_votes = sum(1 for vote in proposal.votes.values() if vote)
                consensus_reached = yes_votes > total_votes / 2
            
            elif proposal.type == ConsensusType.UNANIMOUS:
                consensus_reached = all(proposal.votes.values())
            
            elif proposal.type == ConsensusType.WEIGHTED:
                total_weight = sum(proposal.weights.values())
                weighted_yes = sum(weight for agent_id, weight in proposal.weights.items() 
                                 if proposal.votes.get(agent_id, False))
                consensus_reached = weighted_yes > total_weight / 2
            
            # Update proposal status
            if consensus_reached:
                proposal.status = "approved"
                self.coordination_metrics['consensus_successful'] += 1
            elif datetime.utcnow() > proposal.deadline:
                proposal.status = "expired"
            else:
                return  # Keep waiting
            
            # Move to history and cleanup
            self.consensus_history.append(proposal)
            self.active_proposals.pop(proposal_id, None)
            
            logger.info(f"Consensus proposal {proposal_id} resolved: {proposal.status}")
            
        except Exception as e:
            logger.error(f"Consensus evaluation failed for {proposal_id}: {e}")
    
    async def _cleanup_expired_messages(self):
        """Remove expired messages"""
        
        current_time = datetime.utcnow()
        cutoff_time = current_time - timedelta(hours=self.message_retention_hours)
        
        # Clean message history
        self.message_history = deque(
            [msg for msg in self.message_history if msg.timestamp >= cutoff_time],
            maxlen=self.message_history.maxlen
        )
        
        # Clean agent message queues
        for agent_id in self.message_queues:
            self.message_queues[agent_id] = deque(
                [msg for msg in self.message_queues[agent_id] 
                 if msg.timestamp >= cutoff_time and 
                    (msg.expires_at is None or msg.expires_at > current_time)],
                maxlen=self.message_queues[agent_id].maxlen
            )
    
    async def _cleanup_completed_assignments(self):
        """Move completed assignments to history"""
        
        completed_assignments = [
            assignment for assignment in self.active_assignments.values()
            if assignment.status in ["completed", "failed", "cancelled"]
        ]
        
        for assignment in completed_assignments:
            self.assignment_history.append(assignment)
            self.active_assignments.pop(assignment.id, None)
        
        # Keep history manageable
        self.assignment_history = self.assignment_history[-1000:]  # Keep last 1000
    
    async def _cleanup_expired_proposals(self):
        """Clean up expired consensus proposals"""
        
        current_time = datetime.utcnow()
        expired_proposals = [
            proposal for proposal in self.active_proposals.values()
            if current_time > proposal.deadline and proposal.status == "pending"
        ]
        
        for proposal in expired_proposals:
            proposal.status = "expired"
            self.consensus_history.append(proposal)
            self.active_proposals.pop(proposal.id, None)
        
        # Keep consensus history manageable
        self.consensus_history = self.consensus_history[-500:]  # Keep last 500
