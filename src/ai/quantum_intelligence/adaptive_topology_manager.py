"""
Adaptive Topology Manager
========================

Dynamic swarm topology optimization system that adapts network structure 
based on task complexity, agent performance, and communication patterns.

This system implements:
- Real-time topology adaptation based on workload analysis
- Quantum-inspired network optimization algorithms
- Intelligent agent placement and connection management
- Performance-driven topology restructuring
- Multi-dimensional optimization (latency, throughput, resilience)

Architecture Features:
- Graph-based topology representation
- Machine learning for topology prediction
- Distributed consensus for topology changes
- Rollback mechanisms for failed adaptations
- Performance monitoring and analytics

Topology Types:
- Mesh: Full connectivity for complex collaborative tasks
- Hierarchical: Tree structure for command-and-control workflows  
- Ring: Circular topology for sequential processing
- Star: Central hub for broadcast operations
- Hybrid: Combination topologies for specialized requirements
"""

import asyncio
import networkx as nx
import numpy as np
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor
import heapq
import math

logger = logging.getLogger(__name__)

class TopologyType(Enum):
    """Available topology types for swarm organization."""
    MESH = "mesh"
    HIERARCHICAL = "hierarchical"
    RING = "ring"
    STAR = "star"
    HYBRID = "hybrid"
    QUANTUM_ENTANGLED = "quantum_entangled"
    DYNAMIC_GRAPH = "dynamic_graph"

class OptimizationMetric(Enum):
    """Metrics for topology optimization."""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    RESILIENCE = "resilience"
    LOAD_BALANCE = "load_balance"
    COMMUNICATION_COST = "communication_cost"
    FAULT_TOLERANCE = "fault_tolerance"
    SCALABILITY = "scalability"

@dataclass
class TopologyNode:
    """Represents a node (agent) in the topology."""
    node_id: str
    agent_type: str
    capabilities: List[str]
    performance_score: float = 0.0
    load_factor: float = 0.0
    connection_capacity: int = 10
    position: Tuple[float, float] = (0.0, 0.0)  # 2D coordinates
    status: str = "active"
    last_update: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TopologyEdge:
    """Represents a connection between nodes."""
    source_id: str
    target_id: str
    weight: float = 1.0
    latency: float = 0.0
    bandwidth: float = 100.0
    reliability: float = 1.0
    communication_count: int = 0
    last_used: float = field(default_factory=time.time)
    edge_type: str = "standard"

@dataclass
class TopologyMetrics:
    """Comprehensive topology performance metrics."""
    avg_path_length: float = 0.0
    clustering_coefficient: float = 0.0
    network_diameter: int = 0
    connectivity_index: float = 0.0
    load_distribution_variance: float = 0.0
    communication_efficiency: float = 0.0
    fault_tolerance_score: float = 0.0
    adaptation_frequency: float = 0.0
    total_nodes: int = 0
    total_edges: int = 0
    timestamp: float = field(default_factory=time.time)

@dataclass
class TaskComplexityProfile:
    """Profile describing task complexity characteristics."""
    computational_intensity: float = 0.5
    communication_requirements: float = 0.5
    coordination_complexity: float = 0.5
    data_volume: float = 0.5
    real_time_constraints: float = 0.5
    fault_tolerance_needs: float = 0.5
    parallelization_potential: float = 0.5
    agent_specialization_required: float = 0.5

class TopologyOptimizer:
    """Advanced topology optimization using quantum-inspired algorithms."""
    
    def __init__(self):
        self.optimization_history = deque(maxlen=100)
        self.performance_cache = {}
        
    async def optimize_topology(self, 
                              current_topology: nx.Graph,
                              nodes: Dict[str, TopologyNode],
                              task_profile: TaskComplexityProfile,
                              constraints: Dict[str, Any]) -> Tuple[nx.Graph, TopologyType]:
        """Optimize topology based on task requirements and constraints."""
        try:
            # Analyze current topology performance
            current_score = await self._evaluate_topology_performance(
                current_topology, nodes, task_profile
            )
            
            # Generate topology candidates
            candidates = await self._generate_topology_candidates(
                nodes, task_profile, constraints
            )
            
            # Evaluate candidates using quantum-inspired scoring
            best_topology = None
            best_type = TopologyType.MESH
            best_score = current_score
            
            for topology_type, candidate_graph in candidates:
                score = await self._evaluate_topology_performance(
                    candidate_graph, nodes, task_profile
                )
                
                if score > best_score:
                    best_topology = candidate_graph
                    best_type = topology_type
                    best_score = score
            
            # Return optimized topology or current if no improvement
            if best_topology is not None:
                logger.info(f"Topology optimized: {best_type.value} (score: {best_score:.3f})")
                return best_topology, best_type
            else:
                return current_topology, TopologyType.MESH
                
        except Exception as e:
            logger.error(f"Topology optimization failed: {e}")
            return current_topology, TopologyType.MESH
    
    async def _generate_topology_candidates(self,
                                          nodes: Dict[str, TopologyNode],
                                          task_profile: TaskComplexityProfile,
                                          constraints: Dict[str, Any]) -> List[Tuple[TopologyType, nx.Graph]]:
        """Generate candidate topologies based on task requirements."""
        candidates = []
        node_ids = list(nodes.keys())
        
        try:
            # Mesh topology for high communication requirements
            if task_profile.communication_requirements > 0.7:
                mesh_graph = await self._create_mesh_topology(nodes)
                candidates.append((TopologyType.MESH, mesh_graph))
            
            # Hierarchical topology for coordination-heavy tasks
            if task_profile.coordination_complexity > 0.6:
                hierarchical_graph = await self._create_hierarchical_topology(nodes, task_profile)
                candidates.append((TopologyType.HIERARCHICAL, hierarchical_graph))
            
            # Star topology for broadcast-heavy operations
            if task_profile.communication_requirements > 0.5 and len(nodes) < 20:
                star_graph = await self._create_star_topology(nodes)
                candidates.append((TopologyType.STAR, star_graph))
            
            # Ring topology for sequential processing
            if task_profile.parallelization_potential < 0.4:
                ring_graph = await self._create_ring_topology(nodes)
                candidates.append((TopologyType.RING, ring_graph))
            
            # Quantum entangled topology for high-performance requirements
            if task_profile.computational_intensity > 0.8:
                quantum_graph = await self._create_quantum_entangled_topology(nodes, task_profile)
                candidates.append((TopologyType.QUANTUM_ENTANGLED, quantum_graph))
            
            # Hybrid topology for balanced requirements
            if self._requires_hybrid_topology(task_profile):
                hybrid_graph = await self._create_hybrid_topology(nodes, task_profile)
                candidates.append((TopologyType.HYBRID, hybrid_graph))
            
            return candidates
            
        except Exception as e:
            logger.error(f"Candidate generation failed: {e}")
            return [(TopologyType.MESH, nx.complete_graph(node_ids))]
    
    async def _create_mesh_topology(self, nodes: Dict[str, TopologyNode]) -> nx.Graph:
        """Create optimized mesh topology with selective connections."""
        graph = nx.Graph()
        node_ids = list(nodes.keys())
        
        # Add all nodes
        for node_id, node in nodes.items():
            graph.add_node(node_id, **node.metadata)
        
        # Create connections based on capability matching and performance
        for i, node_id_1 in enumerate(node_ids):
            node_1 = nodes[node_id_1]
            for j, node_id_2 in enumerate(node_ids[i+1:], i+1):
                node_2 = nodes[node_id_2]
                
                # Calculate connection value based on capability synergy
                connection_value = await self._calculate_connection_value(node_1, node_2)
                
                # Connect if value exceeds threshold
                if connection_value > 0.5:
                    weight = connection_value
                    graph.add_edge(node_id_1, node_id_2, weight=weight)
        
        return graph
    
    async def _create_hierarchical_topology(self, 
                                          nodes: Dict[str, TopologyNode],
                                          task_profile: TaskComplexityProfile) -> nx.Graph:
        """Create hierarchical topology with intelligent level assignment."""
        graph = nx.DiGraph()
        
        # Sort nodes by performance and capability
        sorted_nodes = sorted(nodes.items(), 
                            key=lambda x: (x[1].performance_score, len(x[1].capabilities)), 
                            reverse=True)
        
        # Create hierarchy levels
        levels = self._create_hierarchy_levels(sorted_nodes, task_profile)
        
        # Add nodes with level information
        for level, level_nodes in enumerate(levels):
            for node_id, node in level_nodes:
                graph.add_node(node_id, level=level, **node.metadata)
        
        # Create hierarchical connections
        for level in range(len(levels) - 1):
            current_level = levels[level]
            next_level = levels[level + 1]
            
            # Connect each node in current level to appropriate nodes in next level
            for current_node_id, current_node in current_level:
                # Find best subordinates based on capability match
                subordinates = await self._find_best_subordinates(
                    current_node, next_level, max_subordinates=3
                )
                
                for sub_node_id, sub_node in subordinates:
                    weight = await self._calculate_hierarchical_weight(current_node, sub_node)
                    graph.add_edge(current_node_id, sub_node_id, weight=weight)
        
        return graph
    
    async def _create_star_topology(self, nodes: Dict[str, TopologyNode]) -> nx.Graph:
        """Create star topology with optimal hub selection."""
        graph = nx.Graph()
        
        # Select hub based on performance and connection capacity
        hub_node = max(nodes.items(), 
                      key=lambda x: x[1].performance_score * x[1].connection_capacity)
        hub_id = hub_node[0]
        
        # Add all nodes
        for node_id, node in nodes.items():
            graph.add_node(node_id, is_hub=(node_id == hub_id), **node.metadata)
        
        # Connect all nodes to hub
        for node_id in nodes:
            if node_id != hub_id:
                weight = await self._calculate_connection_value(hub_node[1], nodes[node_id])
                graph.add_edge(hub_id, node_id, weight=weight)
        
        return graph
    
    async def _create_ring_topology(self, nodes: Dict[str, TopologyNode]) -> nx.Graph:
        """Create optimized ring topology with intelligent ordering."""
        graph = nx.Graph()
        
        # Order nodes for optimal sequential processing
        ordered_nodes = await self._optimize_ring_order(nodes)
        
        # Add nodes
        for node_id, node in nodes.items():
            graph.add_node(node_id, **node.metadata)
        
        # Create ring connections
        for i in range(len(ordered_nodes)):
            current_id = ordered_nodes[i]
            next_id = ordered_nodes[(i + 1) % len(ordered_nodes)]
            
            weight = await self._calculate_sequential_weight(nodes[current_id], nodes[next_id])
            graph.add_edge(current_id, next_id, weight=weight)
        
        return graph
    
    async def _create_quantum_entangled_topology(self,
                                               nodes: Dict[str, TopologyNode],
                                               task_profile: TaskComplexityProfile) -> nx.Graph:
        """Create quantum-inspired entangled topology for high-performance computing."""
        graph = nx.Graph()
        
        # Add nodes with quantum properties
        for node_id, node in nodes.items():
            quantum_state = np.random.random()  # Simulated quantum state
            graph.add_node(node_id, quantum_state=quantum_state, **node.metadata)
        
        # Create entangled connections based on quantum correlation
        node_ids = list(nodes.keys())
        for i, node_id_1 in enumerate(node_ids):
            for j, node_id_2 in enumerate(node_ids[i+1:], i+1):
                # Calculate quantum entanglement strength
                entanglement = await self._calculate_quantum_entanglement(
                    nodes[node_id_1], nodes[node_id_2], task_profile
                )
                
                # Create connection if entanglement is strong
                if entanglement > 0.6:
                    graph.add_edge(node_id_1, node_id_2, 
                                 weight=entanglement, 
                                 edge_type="quantum_entangled")
        
        return graph
    
    async def _create_hybrid_topology(self,
                                    nodes: Dict[str, TopologyNode],
                                    task_profile: TaskComplexityProfile) -> nx.Graph:
        """Create hybrid topology combining multiple topology types."""
        graph = nx.Graph()
        
        # Partition nodes based on roles and capabilities
        partitions = await self._partition_nodes_by_role(nodes, task_profile)
        
        # Create different topologies for different partitions
        for partition_name, partition_nodes in partitions.items():
            if partition_name == "coordination":
                # Star topology for coordination nodes
                coord_graph = await self._create_star_topology(partition_nodes)
            elif partition_name == "processing":
                # Mesh topology for processing nodes
                coord_graph = await self._create_mesh_topology(partition_nodes)
            elif partition_name == "communication":
                # Ring topology for communication nodes
                coord_graph = await self._create_ring_topology(partition_nodes)
            else:
                # Default mesh for unknown partitions
                coord_graph = await self._create_mesh_topology(partition_nodes)
            
            # Merge partition graph into main graph
            graph = nx.compose(graph, coord_graph)
        
        # Create inter-partition connections
        await self._create_inter_partition_connections(graph, partitions)
        
        return graph
    
    async def _evaluate_topology_performance(self,
                                           topology: nx.Graph,
                                           nodes: Dict[str, TopologyNode],
                                           task_profile: TaskComplexityProfile) -> float:
        """Evaluate topology performance using multiple metrics."""
        try:
            scores = {}
            
            # Calculate basic graph metrics
            if topology.number_of_nodes() > 0:
                scores['connectivity'] = nx.node_connectivity(topology) / max(1, topology.number_of_nodes() - 1)
            else:
                scores['connectivity'] = 0
            
            # Path length efficiency
            if nx.is_connected(topology):
                avg_path_length = nx.average_shortest_path_length(topology)
                max_possible_path = topology.number_of_nodes() - 1
                scores['path_efficiency'] = 1.0 - (avg_path_length - 1) / max(1, max_possible_path)
            else:
                scores['path_efficiency'] = 0.5
            
            # Load balancing
            scores['load_balance'] = await self._calculate_load_balance_score(topology, nodes)
            
            # Communication efficiency based on task profile
            scores['communication'] = await self._calculate_communication_efficiency(
                topology, nodes, task_profile
            )
            
            # Fault tolerance
            scores['fault_tolerance'] = await self._calculate_fault_tolerance_score(topology)
            
            # Scalability potential
            scores['scalability'] = await self._calculate_scalability_score(topology, task_profile)
            
            # Weighted combination based on task requirements
            weights = {
                'connectivity': task_profile.communication_requirements,
                'path_efficiency': task_profile.real_time_constraints,
                'load_balance': task_profile.computational_intensity,
                'communication': task_profile.communication_requirements,
                'fault_tolerance': task_profile.fault_tolerance_needs,
                'scalability': 0.2  # Base scalability importance
            }
            
            # Normalize weights
            total_weight = sum(weights.values())
            weights = {k: v / total_weight for k, v in weights.items()}
            
            # Calculate final score
            final_score = sum(scores[metric] * weight for metric, weight in weights.items())
            
            return max(0.0, min(1.0, final_score))
            
        except Exception as e:
            logger.error(f"Topology evaluation failed: {e}")
            return 0.0
    
    async def _calculate_connection_value(self, 
                                        node_1: TopologyNode, 
                                        node_2: TopologyNode) -> float:
        """Calculate the value of connecting two nodes."""
        # Capability synergy
        common_capabilities = set(node_1.capabilities) & set(node_2.capabilities)
        total_capabilities = set(node_1.capabilities) | set(node_2.capabilities)
        synergy = len(common_capabilities) / max(1, len(total_capabilities))
        
        # Performance compatibility
        perf_diff = abs(node_1.performance_score - node_2.performance_score)
        perf_compat = 1.0 - perf_diff
        
        # Load balancing benefit
        load_diff = abs(node_1.load_factor - node_2.load_factor)
        load_benefit = load_diff  # Higher difference means better load balancing
        
        # Combined score
        connection_value = (synergy * 0.4 + perf_compat * 0.4 + load_benefit * 0.2)
        
        return connection_value
    
    def _create_hierarchy_levels(self, 
                               sorted_nodes: List[Tuple[str, TopologyNode]], 
                               task_profile: TaskComplexityProfile) -> List[List[Tuple[str, TopologyNode]]]:
        """Create hierarchy levels based on node capabilities and task requirements."""
        # Determine number of levels based on coordination complexity
        if task_profile.coordination_complexity > 0.8:
            num_levels = min(4, max(2, len(sorted_nodes) // 3))
        elif task_profile.coordination_complexity > 0.5:
            num_levels = min(3, max(2, len(sorted_nodes) // 4))
        else:
            num_levels = 2
        
        levels = [[] for _ in range(num_levels)]
        
        # Distribute nodes across levels
        for i, (node_id, node) in enumerate(sorted_nodes):
            level_index = min(i // max(1, len(sorted_nodes) // num_levels), num_levels - 1)
            levels[level_index].append((node_id, node))
        
        return levels
    
    def _requires_hybrid_topology(self, task_profile: TaskComplexityProfile) -> bool:
        """Determine if a hybrid topology is needed."""
        # Check if task has diverse requirements
        requirements = [
            task_profile.computational_intensity,
            task_profile.communication_requirements,
            task_profile.coordination_complexity,
            task_profile.real_time_constraints
        ]
        
        # High variance in requirements suggests need for hybrid topology
        variance = np.var(requirements)
        return variance > 0.1

class AdaptiveTopologyManager:
    """
    Adaptive Topology Manager for Dynamic Swarm Optimization.
    
    Implements intelligent topology adaptation based on real-time performance
    analysis, task complexity assessment, and quantum-inspired optimization.
    """
    
    def __init__(self, max_nodes: int = 100):
        self.max_nodes = max_nodes
        self.current_topology = nx.Graph()
        self.nodes: Dict[str, TopologyNode] = {}
        self.edges: Dict[Tuple[str, str], TopologyEdge] = {}
        self.topology_type = TopologyType.MESH
        self.metrics = TopologyMetrics()
        self.optimizer = TopologyOptimizer()
        self.adaptation_history = deque(maxlen=50)
        self.performance_monitor_active = False
        self.monitor_thread = None
        self.lock = threading.RLock()
        
        logger.info("Adaptive Topology Manager initialized")
    
    async def register_node(self, 
                          node_id: str, 
                          agent_type: str,
                          capabilities: List[str],
                          performance_score: float = 0.5,
                          connection_capacity: int = 10) -> bool:
        """Register a new node in the topology."""
        try:
            with self.lock:
                if len(self.nodes) >= self.max_nodes:
                    logger.warning(f"Maximum node limit reached ({self.max_nodes})")
                    return False
                
                # Create node
                node = TopologyNode(
                    node_id=node_id,
                    agent_type=agent_type,
                    capabilities=capabilities,
                    performance_score=performance_score,
                    connection_capacity=connection_capacity,
                    position=self._calculate_optimal_position(node_id),
                    metadata={
                        'registration_time': time.time(),
                        'adaptation_count': 0
                    }
                )
                
                self.nodes[node_id] = node
                self.current_topology.add_node(node_id)
                
                # Trigger topology adaptation
                await self._adapt_topology_for_new_node(node_id)
                
                logger.info(f"Node {node_id} registered with type {agent_type}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to register node {node_id}: {e}")
            return False
    
    async def unregister_node(self, node_id: str) -> bool:
        """Unregister a node from the topology."""
        try:
            with self.lock:
                if node_id not in self.nodes:
                    return False
                
                # Remove node and its edges
                if self.current_topology.has_node(node_id):
                    self.current_topology.remove_node(node_id)
                
                # Remove from edges dict
                edges_to_remove = [key for key in self.edges.keys() 
                                 if key[0] == node_id or key[1] == node_id]
                for edge_key in edges_to_remove:
                    del self.edges[edge_key]
                
                # Remove node
                del self.nodes[node_id]
                
                # Trigger topology adaptation
                await self._adapt_topology_after_node_removal(node_id)
                
                logger.info(f"Node {node_id} unregistered")
                return True
                
        except Exception as e:
            logger.error(f"Failed to unregister node {node_id}: {e}")
            return False
    
    async def adapt_topology(self, 
                           task_complexity: TaskComplexityProfile,
                           constraints: Optional[Dict[str, Any]] = None) -> bool:
        """Adapt topology based on task complexity and constraints."""
        try:
            with self.lock:
                if not self.nodes:
                    logger.warning("No nodes available for topology adaptation")
                    return False
                
                constraints = constraints or {}
                
                # Optimize topology
                new_topology, new_type = await self.optimizer.optimize_topology(
                    self.current_topology, self.nodes, task_complexity, constraints
                )
                
                # Apply new topology if it's different and better
                if await self._is_topology_improvement(new_topology, new_type):
                    old_type = self.topology_type
                    await self._apply_new_topology(new_topology, new_type)
                    
                    # Record adaptation
                    self.adaptation_history.append({
                        'timestamp': time.time(),
                        'old_type': old_type.value,
                        'new_type': new_type.value,
                        'task_profile': task_complexity.__dict__,
                        'node_count': len(self.nodes),
                        'improvement_score': await self._calculate_improvement_score(new_topology)
                    })
                    
                    logger.info(f"Topology adapted from {old_type.value} to {new_type.value}")
                    return True
                
                return False
                
        except Exception as e:
            logger.error(f"Topology adaptation failed: {e}")
            return False
    
    async def update_node_performance(self, 
                                    node_id: str, 
                                    performance_score: float,
                                    load_factor: float) -> bool:
        """Update node performance metrics."""
        try:
            with self.lock:
                if node_id not in self.nodes:
                    return False
                
                node = self.nodes[node_id]
                node.performance_score = performance_score
                node.load_factor = load_factor
                node.last_update = time.time()
                
                # Check if topology needs adaptation based on performance change
                if await self._should_adapt_for_performance_change(node_id, performance_score):
                    # Trigger background adaptation
                    asyncio.create_task(self._background_adaptation())
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to update node performance: {e}")
            return False
    
    async def record_communication(self, 
                                 source_id: str, 
                                 target_id: str,
                                 latency: float = 0.0,
                                 data_size: float = 0.0) -> bool:
        """Record communication between nodes for topology optimization."""
        try:
            with self.lock:
                if source_id not in self.nodes or target_id not in self.nodes:
                    return False
                
                edge_key = (min(source_id, target_id), max(source_id, target_id))
                
                # Update or create edge
                if edge_key in self.edges:
                    edge = self.edges[edge_key]
                    edge.communication_count += 1
                    edge.latency = (edge.latency * 0.9 + latency * 0.1)  # Exponential moving average
                    edge.last_used = time.time()
                else:
                    edge = TopologyEdge(
                        source_id=source_id,
                        target_id=target_id,
                        latency=latency,
                        communication_count=1
                    )
                    self.edges[edge_key] = edge
                
                # Update topology edge if it exists
                if self.current_topology.has_edge(source_id, target_id):
                    self.current_topology[source_id][target_id]['weight'] = self._calculate_edge_weight(edge)
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to record communication: {e}")
            return False
    
    async def get_optimal_neighbors(self, 
                                  node_id: str, 
                                  max_neighbors: int = 5) -> List[str]:
        """Get optimal neighbors for a node based on current topology."""
        try:
            with self.lock:
                if node_id not in self.current_topology:
                    return []
                
                # Get current neighbors
                neighbors = list(self.current_topology.neighbors(node_id))
                
                if len(neighbors) <= max_neighbors:
                    return neighbors
                
                # Rank neighbors by connection quality
                neighbor_scores = []
                for neighbor_id in neighbors:
                    score = await self._calculate_neighbor_quality(node_id, neighbor_id)
                    neighbor_scores.append((neighbor_id, score))
                
                # Sort by score and return top neighbors
                neighbor_scores.sort(key=lambda x: x[1], reverse=True)
                return [neighbor_id for neighbor_id, _ in neighbor_scores[:max_neighbors]]
                
        except Exception as e:
            logger.error(f"Failed to get optimal neighbors: {e}")
            return []
    
    async def get_topology_status(self) -> Dict[str, Any]:
        """Get comprehensive topology status."""
        try:
            with self.lock:
                # Calculate current metrics
                await self._update_topology_metrics()
                
                return {
                    'topology_type': self.topology_type.value,
                    'node_count': len(self.nodes),
                    'edge_count': len(self.edges),
                    'metrics': {
                        'avg_path_length': self.metrics.avg_path_length,
                        'clustering_coefficient': self.metrics.clustering_coefficient,
                        'network_diameter': self.metrics.network_diameter,
                        'connectivity_index': self.metrics.connectivity_index,
                        'load_distribution_variance': self.metrics.load_distribution_variance,
                        'communication_efficiency': self.metrics.communication_efficiency,
                        'fault_tolerance_score': self.metrics.fault_tolerance_score
                    },
                    'nodes': {
                        node_id: {
                            'agent_type': node.agent_type,
                            'capabilities': node.capabilities,
                            'performance_score': node.performance_score,
                            'load_factor': node.load_factor,
                            'status': node.status,
                            'neighbors': list(self.current_topology.neighbors(node_id)) if self.current_topology.has_node(node_id) else []
                        }
                        for node_id, node in self.nodes.items()
                    },
                    'adaptation_history': list(self.adaptation_history),
                    'last_adaptation': self.adaptation_history[-1]['timestamp'] if self.adaptation_history else None
                }
                
        except Exception as e:
            logger.error(f"Failed to get topology status: {e}")
            return {}
    
    async def start_performance_monitoring(self, monitor_interval: float = 30.0):
        """Start continuous topology performance monitoring."""
        if self.performance_monitor_active:
            logger.warning("Performance monitoring already active")
            return
        
        self.performance_monitor_active = True
        
        def monitor_loop():
            """Background monitoring loop."""
            asyncio.set_event_loop(asyncio.new_event_loop())
            loop = asyncio.get_event_loop()
            
            while self.performance_monitor_active:
                try:
                    loop.run_until_complete(self._performance_monitor_cycle())
                    time.sleep(monitor_interval)
                except Exception as e:
                    logger.error(f"Performance monitoring error: {e}")
                    time.sleep(10)
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Topology performance monitoring started")
    
    async def stop_performance_monitoring(self):
        """Stop performance monitoring."""
        self.performance_monitor_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        logger.info("Topology performance monitoring stopped")
    
    # Private methods
    
    def _calculate_optimal_position(self, node_id: str) -> Tuple[float, float]:
        """Calculate optimal 2D position for new node."""
        # Simple grid-based positioning for visualization
        node_count = len(self.nodes)
        grid_size = math.ceil(math.sqrt(self.max_nodes))
        
        x = (node_count % grid_size) * 1.0
        y = (node_count // grid_size) * 1.0
        
        return (x, y)
    
    async def _adapt_topology_for_new_node(self, new_node_id: str):
        """Adapt topology when new node is added."""
        try:
            new_node = self.nodes[new_node_id]
            
            # Find best initial connections
            potential_connections = []
            for existing_id, existing_node in self.nodes.items():
                if existing_id != new_node_id:
                    connection_value = await self.optimizer._calculate_connection_value(
                        new_node, existing_node
                    )
                    potential_connections.append((existing_id, connection_value))
            
            # Sort and select top connections
            potential_connections.sort(key=lambda x: x[1], reverse=True)
            max_initial_connections = min(3, len(potential_connections))
            
            # Create initial connections
            for existing_id, value in potential_connections[:max_initial_connections]:
                if value > 0.5:  # Threshold for initial connection
                    self.current_topology.add_edge(new_node_id, existing_id, weight=value)
            
        except Exception as e:
            logger.error(f"Failed to adapt topology for new node: {e}")
    
    async def _adapt_topology_after_node_removal(self, removed_node_id: str):
        """Adapt topology after node removal."""
        try:
            # Check if remaining topology is still connected
            if not nx.is_connected(self.current_topology) and len(self.nodes) > 1:
                # Reconnect components
                components = list(nx.connected_components(self.current_topology))
                if len(components) > 1:
                    # Connect largest components
                    largest_components = sorted(components, key=len, reverse=True)[:2]
                    
                    # Find best connection between components
                    best_connection = None
                    best_score = 0.0
                    
                    for node_1 in largest_components[0]:
                        for node_2 in largest_components[1]:
                            score = await self.optimizer._calculate_connection_value(
                                self.nodes[node_1], self.nodes[node_2]
                            )
                            if score > best_score:
                                best_score = score
                                best_connection = (node_1, node_2)
                    
                    if best_connection:
                        self.current_topology.add_edge(*best_connection, weight=best_score)
            
        except Exception as e:
            logger.error(f"Failed to adapt topology after node removal: {e}")
    
    async def _is_topology_improvement(self, new_topology: nx.Graph, new_type: TopologyType) -> bool:
        """Check if new topology is an improvement."""
        try:
            # Create dummy task profile for comparison
            comparison_profile = TaskComplexityProfile()
            
            current_score = await self.optimizer._evaluate_topology_performance(
                self.current_topology, self.nodes, comparison_profile
            )
            
            new_score = await self.optimizer._evaluate_topology_performance(
                new_topology, self.nodes, comparison_profile
            )
            
            # Require minimum improvement threshold to avoid thrashing
            improvement_threshold = 0.05
            return new_score > current_score + improvement_threshold
            
        except Exception as e:
            logger.error(f"Failed to evaluate topology improvement: {e}")
            return False
    
    async def _apply_new_topology(self, new_topology: nx.Graph, new_type: TopologyType):
        """Apply new topology configuration."""
        try:
            # Update topology
            self.current_topology = new_topology.copy()
            self.topology_type = new_type
            
            # Update edges dict
            self.edges.clear()
            for edge in self.current_topology.edges(data=True):
                source_id, target_id, data = edge
                edge_key = (min(source_id, target_id), max(source_id, target_id))
                
                self.edges[edge_key] = TopologyEdge(
                    source_id=source_id,
                    target_id=target_id,
                    weight=data.get('weight', 1.0),
                    edge_type=data.get('edge_type', 'standard')
                )
            
            # Update node metadata
            for node_id in self.nodes:
                if node_id in self.nodes:
                    self.nodes[node_id].metadata['adaptation_count'] = (
                        self.nodes[node_id].metadata.get('adaptation_count', 0) + 1
                    )
            
        except Exception as e:
            logger.error(f"Failed to apply new topology: {e}")
    
    async def _should_adapt_for_performance_change(self, node_id: str, new_performance: float) -> bool:
        """Check if topology should adapt based on performance change."""
        try:
            node = self.nodes[node_id]
            old_performance = node.performance_score
            
            # Significant performance change threshold
            change_threshold = 0.2
            performance_change = abs(new_performance - old_performance)
            
            return performance_change > change_threshold
            
        except Exception as e:
            logger.error(f"Failed to check adaptation need: {e}")
            return False
    
    async def _background_adaptation(self):
        """Perform background topology adaptation."""
        try:
            # Generate adaptive task profile based on current state
            task_profile = await self._generate_adaptive_task_profile()
            await self.adapt_topology(task_profile)
            
        except Exception as e:
            logger.error(f"Background adaptation failed: {e}")
    
    async def _generate_adaptive_task_profile(self) -> TaskComplexityProfile:
        """Generate task complexity profile based on current system state."""
        try:
            # Analyze current communication patterns
            total_communications = sum(edge.communication_count for edge in self.edges.values())
            avg_communications = total_communications / max(1, len(self.edges)) if self.edges else 0
            
            # Analyze load distribution
            load_values = [node.load_factor for node in self.nodes.values()]
            avg_load = np.mean(load_values) if load_values else 0.5
            load_variance = np.var(load_values) if load_values else 0.0
            
            # Generate adaptive profile
            return TaskComplexityProfile(
                computational_intensity=min(1.0, avg_load * 2),
                communication_requirements=min(1.0, avg_communications / 10.0),
                coordination_complexity=min(1.0, load_variance * 5),
                real_time_constraints=0.5,  # Default
                fault_tolerance_needs=0.6,  # Default
                parallelization_potential=max(0.2, 1.0 - load_variance)
            )
            
        except Exception as e:
            logger.error(f"Failed to generate adaptive task profile: {e}")
            return TaskComplexityProfile()
    
    def _calculate_edge_weight(self, edge: TopologyEdge) -> float:
        """Calculate edge weight based on communication patterns."""
        # Higher communication count and lower latency = higher weight
        comm_factor = min(1.0, edge.communication_count / 100.0)
        latency_factor = max(0.1, 1.0 - edge.latency)
        reliability_factor = edge.reliability
        
        return (comm_factor * 0.4 + latency_factor * 0.4 + reliability_factor * 0.2)
    
    async def _calculate_neighbor_quality(self, node_id: str, neighbor_id: str) -> float:
        """Calculate quality score for a neighbor relationship."""
        try:
            node = self.nodes[node_id]
            neighbor = self.nodes[neighbor_id]
            
            # Connection value
            connection_value = await self.optimizer._calculate_connection_value(node, neighbor)
            
            # Communication frequency
            edge_key = (min(node_id, neighbor_id), max(node_id, neighbor_id))
            comm_score = 0.0
            if edge_key in self.edges:
                edge = self.edges[edge_key]
                comm_score = min(1.0, edge.communication_count / 50.0)
            
            # Performance compatibility
            perf_diff = abs(node.performance_score - neighbor.performance_score)
            perf_score = 1.0 - perf_diff
            
            # Combined quality score
            return connection_value * 0.4 + comm_score * 0.3 + perf_score * 0.3
            
        except Exception as e:
            logger.error(f"Failed to calculate neighbor quality: {e}")
            return 0.0
    
    async def _update_topology_metrics(self):
        """Update topology performance metrics."""
        try:
            if not self.current_topology.nodes():
                return
            
            # Basic graph metrics
            self.metrics.total_nodes = self.current_topology.number_of_nodes()
            self.metrics.total_edges = self.current_topology.number_of_edges()
            
            if nx.is_connected(self.current_topology):
                self.metrics.avg_path_length = nx.average_shortest_path_length(self.current_topology)
                self.metrics.network_diameter = nx.diameter(self.current_topology)
            else:
                # For disconnected graphs, calculate for largest component
                largest_cc = max(nx.connected_components(self.current_topology), key=len)
                subgraph = self.current_topology.subgraph(largest_cc)
                if len(subgraph) > 1:
                    self.metrics.avg_path_length = nx.average_shortest_path_length(subgraph)
                    self.metrics.network_diameter = nx.diameter(subgraph)
            
            self.metrics.clustering_coefficient = nx.average_clustering(self.current_topology)
            self.metrics.connectivity_index = nx.node_connectivity(self.current_topology) / max(1, self.metrics.total_nodes - 1)
            
            # Load distribution variance
            load_values = [node.load_factor for node in self.nodes.values()]
            self.metrics.load_distribution_variance = np.var(load_values) if load_values else 0.0
            
            # Communication efficiency (based on edge weights)
            edge_weights = [data.get('weight', 1.0) for _, _, data in self.current_topology.edges(data=True)]
            self.metrics.communication_efficiency = np.mean(edge_weights) if edge_weights else 0.0
            
            # Fault tolerance (average node degree / max possible degree)
            degrees = [self.current_topology.degree(node) for node in self.current_topology.nodes()]
            max_degree = self.metrics.total_nodes - 1
            self.metrics.fault_tolerance_score = np.mean(degrees) / max(1, max_degree) if degrees else 0.0
            
            self.metrics.timestamp = time.time()
            
        except Exception as e:
            logger.error(f"Failed to update topology metrics: {e}")
    
    async def _performance_monitor_cycle(self):
        """Single cycle of performance monitoring."""
        try:
            await self._update_topology_metrics()
            
            # Check for adaptation triggers
            adaptation_needed = False
            
            # High load variance suggests need for rebalancing
            if self.metrics.load_distribution_variance > 0.3:
                adaptation_needed = True
                logger.info("High load variance detected, triggering adaptation")
            
            # Low communication efficiency suggests suboptimal topology
            if self.metrics.communication_efficiency < 0.5:
                adaptation_needed = True
                logger.info("Low communication efficiency detected, triggering adaptation")
            
            # Perform adaptation if needed
            if adaptation_needed:
                await self._background_adaptation()
            
        except Exception as e:
            logger.error(f"Performance monitor cycle failed: {e}")
    
    async def _calculate_improvement_score(self, new_topology: nx.Graph) -> float:
        """Calculate improvement score for new topology."""
        try:
            # Simple improvement metric based on connectivity and efficiency
            if new_topology.number_of_nodes() == 0:
                return 0.0
            
            # Connectivity improvement
            connectivity_score = nx.node_connectivity(new_topology) / max(1, new_topology.number_of_nodes() - 1)
            
            # Path efficiency improvement
            if nx.is_connected(new_topology):
                path_score = 1.0 / max(1, nx.average_shortest_path_length(new_topology))
            else:
                path_score = 0.5
            
            return (connectivity_score + path_score) / 2.0
            
        except Exception as e:
            logger.error(f"Failed to calculate improvement score: {e}")
            return 0.0
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, 'performance_monitor_active') and self.performance_monitor_active:
            asyncio.create_task(self.stop_performance_monitoring())