"""
Dependency resolution utilities for Claude-TIU.

Provides task dependency resolution, DAG analysis, and execution ordering.
"""

import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TaskNode:
    """Represents a task node in the dependency graph."""
    task_id: str
    name: str
    description: str = ""
    dependencies: Set[str] = None
    dependents: Set[str] = None
    priority: int = 0
    estimated_duration: int = 0  # in minutes
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = set()
        if self.dependents is None:
            self.dependents = set()


class DependencyResolver:
    """
    Resolves task dependencies and provides execution ordering.
    
    Uses topological sorting to determine optimal task execution order
    while respecting dependencies and priorities.
    """
    
    def __init__(self):
        self.graph: Dict[str, TaskNode] = {}
        self.in_degree: Dict[str, int] = defaultdict(int)
        self.out_degree: Dict[str, int] = defaultdict(int)
        
    def add_task(
        self, 
        task_id: str, 
        name: str, 
        dependencies: Optional[List[str]] = None,
        priority: int = 0,
        estimated_duration: int = 0,
        description: str = ""
    ) -> None:
        """Add a task to the dependency graph."""
        dependencies = dependencies or []
        deps_set = set(dependencies)
        
        # Create task node
        task_node = TaskNode(
            task_id=task_id,
            name=name,
            description=description,
            dependencies=deps_set,
            priority=priority,
            estimated_duration=estimated_duration
        )
        
        self.graph[task_id] = task_node
        
        # Update degree counts
        self.in_degree[task_id] = len(deps_set)
        
        for dep_id in deps_set:
            self.out_degree[dep_id] += 1
            # Add this task as dependent of the dependency
            if dep_id in self.graph:
                self.graph[dep_id].dependents.add(task_id)
        
        logger.debug(f"Added task {task_id} with {len(deps_set)} dependencies")
    
    def remove_task(self, task_id: str) -> bool:
        """Remove a task from the dependency graph."""
        if task_id not in self.graph:
            return False
        
        task_node = self.graph[task_id]
        
        # Remove this task from its dependencies' dependents
        for dep_id in task_node.dependencies:
            if dep_id in self.graph:
                self.graph[dep_id].dependents.discard(task_id)
            self.out_degree[dep_id] -= 1
        
        # Remove this task's dependencies from dependents
        for dependent_id in task_node.dependents:
            if dependent_id in self.graph:
                self.graph[dependent_id].dependencies.discard(task_id)
            self.in_degree[dependent_id] -= 1
        
        # Remove from graph and degree tracking
        del self.graph[task_id]
        del self.in_degree[task_id]
        del self.out_degree[task_id]
        
        logger.debug(f"Removed task {task_id}")
        return True
    
    def add_dependency(self, task_id: str, dependency_id: str) -> bool:
        """Add a dependency relationship between tasks."""
        if task_id not in self.graph or dependency_id not in self.graph:
            logger.warning(f"Cannot add dependency: task not found")
            return False
        
        if dependency_id in self.graph[task_id].dependencies:
            return True  # Already exists
        
        # Check for circular dependency
        if self._would_create_cycle(task_id, dependency_id):
            logger.error(f"Cannot add dependency: would create circular dependency")
            return False
        
        # Add dependency
        self.graph[task_id].dependencies.add(dependency_id)
        self.graph[dependency_id].dependents.add(task_id)
        
        self.in_degree[task_id] += 1
        self.out_degree[dependency_id] += 1
        
        logger.debug(f"Added dependency: {task_id} depends on {dependency_id}")
        return True
    
    def remove_dependency(self, task_id: str, dependency_id: str) -> bool:
        """Remove a dependency relationship between tasks."""
        if task_id not in self.graph or dependency_id not in self.graph:
            return False
        
        if dependency_id not in self.graph[task_id].dependencies:
            return True  # Already doesn't exist
        
        # Remove dependency
        self.graph[task_id].dependencies.discard(dependency_id)
        self.graph[dependency_id].dependents.discard(task_id)
        
        self.in_degree[task_id] -= 1
        self.out_degree[dependency_id] -= 1
        
        logger.debug(f"Removed dependency: {task_id} no longer depends on {dependency_id}")
        return True
    
    def resolve_execution_order(self, priority_weight: float = 0.5) -> List[str]:
        """
        Resolve execution order using topological sorting with priority consideration.
        
        Args:
            priority_weight: Weight given to priority vs dependency order (0-1)
            
        Returns:
            List of task IDs in execution order
        """
        if not self.graph:
            return []
        
        # Check for circular dependencies first
        if self.has_circular_dependencies():
            cycles = self.find_circular_dependencies()
            raise ValueError(f"Cannot resolve execution order: circular dependencies found: {cycles}")
        
        # Clone degree counts for processing
        in_degree = self.in_degree.copy()
        
        # Priority queue: tasks with no dependencies, ordered by priority
        ready_queue = []
        
        # Initialize ready queue with tasks that have no dependencies
        for task_id, degree in in_degree.items():
            if degree == 0:
                priority = self.graph[task_id].priority
                ready_queue.append((priority, task_id))
        
        # Sort by priority (higher priority first)
        ready_queue.sort(reverse=True)
        
        execution_order = []
        
        while ready_queue:
            # Get highest priority ready task
            _, task_id = ready_queue.pop(0)
            execution_order.append(task_id)
            
            # Process dependents of completed task
            task_node = self.graph[task_id]
            new_ready_tasks = []
            
            for dependent_id in task_node.dependents:
                in_degree[dependent_id] -= 1
                
                # If dependent has no more dependencies, add to ready queue
                if in_degree[dependent_id] == 0:
                    priority = self.graph[dependent_id].priority
                    new_ready_tasks.append((priority, dependent_id))
            
            # Add new ready tasks and resort
            ready_queue.extend(new_ready_tasks)
            ready_queue.sort(reverse=True)
        
        # Verify all tasks were processed
        if len(execution_order) != len(self.graph):
            missing = set(self.graph.keys()) - set(execution_order)
            raise ValueError(f"Failed to resolve all tasks. Missing: {missing}")
        
        logger.info(f"Resolved execution order for {len(execution_order)} tasks")
        return execution_order
    
    def get_parallel_groups(self) -> List[List[str]]:
        """
        Get groups of tasks that can be executed in parallel.
        
        Returns:
            List of groups, where each group contains task IDs that can run concurrently
        """
        if self.has_circular_dependencies():
            raise ValueError("Cannot determine parallel groups: circular dependencies exist")
        
        # Clone graph for processing
        in_degree = self.in_degree.copy()
        parallel_groups = []
        remaining_tasks = set(self.graph.keys())
        
        while remaining_tasks:
            # Find all tasks with no remaining dependencies
            current_group = []
            for task_id in remaining_tasks:
                if in_degree[task_id] == 0:
                    current_group.append(task_id)
            
            if not current_group:
                # This shouldn't happen if no circular dependencies exist
                raise ValueError("Unable to find next parallel group - possible circular dependency")
            
            # Sort current group by priority
            current_group.sort(key=lambda x: self.graph[x].priority, reverse=True)
            parallel_groups.append(current_group)
            
            # Remove completed tasks and update dependencies
            for task_id in current_group:
                remaining_tasks.remove(task_id)
                
                # Update in_degree for dependents
                for dependent_id in self.graph[task_id].dependents:
                    if dependent_id in remaining_tasks:
                        in_degree[dependent_id] -= 1
        
        logger.info(f"Created {len(parallel_groups)} parallel execution groups")
        return parallel_groups
    
    def has_circular_dependencies(self) -> bool:
        """Check if the graph has circular dependencies."""
        try:
            self.resolve_execution_order()
            return False
        except ValueError:
            return True
    
    def find_circular_dependencies(self) -> List[List[str]]:
        """Find all circular dependencies in the graph."""
        cycles = []
        visited = set()
        rec_stack = set()
        
        def dfs_cycle_detection(node: str, path: List[str]) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for dependent in self.graph[node].dependents:
                if dependent not in visited:
                    dfs_cycle_detection(dependent, path.copy())
                elif dependent in rec_stack:
                    # Found cycle
                    cycle_start = path.index(dependent)
                    cycle = path[cycle_start:] + [dependent]
                    cycles.append(cycle)
            
            rec_stack.remove(node)
        
        for task_id in self.graph:
            if task_id not in visited:
                dfs_cycle_detection(task_id, [])
        
        return cycles
    
    def get_critical_path(self) -> Tuple[List[str], int]:
        """
        Calculate the critical path (longest path) through the task graph.
        
        Returns:
            Tuple of (critical path task IDs, total duration in minutes)
        """
        if self.has_circular_dependencies():
            raise ValueError("Cannot calculate critical path: circular dependencies exist")
        
        # Use dynamic programming to find longest path
        memo = {}
        
        def longest_path_from(task_id: str) -> Tuple[int, List[str]]:
            if task_id in memo:
                return memo[task_id]
            
            task_node = self.graph[task_id]
            
            if not task_node.dependents:
                # Leaf node
                result = (task_node.estimated_duration, [task_id])
            else:
                max_duration = 0
                max_path = []
                
                for dependent_id in task_node.dependents:
                    dep_duration, dep_path = longest_path_from(dependent_id)
                    total_duration = dep_duration + task_node.estimated_duration
                    
                    if total_duration > max_duration:
                        max_duration = total_duration
                        max_path = [task_id] + dep_path
                
                result = (max_duration, max_path)
            
            memo[task_id] = result
            return result
        
        # Find the longest path starting from any root node
        root_nodes = [task_id for task_id in self.graph if not self.graph[task_id].dependencies]
        
        critical_path = []
        critical_duration = 0
        
        for root_id in root_nodes:
            duration, path = longest_path_from(root_id)
            if duration > critical_duration:
                critical_duration = duration
                critical_path = path
        
        logger.info(f"Critical path: {len(critical_path)} tasks, {critical_duration} minutes")
        return critical_path, critical_duration
    
    def get_task_info(self, task_id: str) -> Optional[TaskNode]:
        """Get information about a specific task."""
        return self.graph.get(task_id)
    
    def get_all_tasks(self) -> Dict[str, TaskNode]:
        """Get all tasks in the graph."""
        return self.graph.copy()
    
    def clear(self) -> None:
        """Clear all tasks from the resolver."""
        self.graph.clear()
        self.in_degree.clear()
        self.out_degree.clear()
        logger.debug("Cleared dependency resolver")
    
    def _would_create_cycle(self, from_task: str, to_task: str) -> bool:
        """Check if adding a dependency would create a cycle."""
        # DFS from to_task to see if we can reach from_task
        visited = set()
        stack = [to_task]
        
        while stack:
            current = stack.pop()
            if current == from_task:
                return True
            
            if current in visited:
                continue
                
            visited.add(current)
            
            # Add all dependents to stack
            if current in self.graph:
                stack.extend(self.graph[current].dependents)
        
        return False
    
    def __str__(self) -> str:
        """String representation of the dependency graph."""
        lines = [f"DependencyResolver with {len(self.graph)} tasks:"]
        
        for task_id, task_node in self.graph.items():
            deps = ', '.join(task_node.dependencies) if task_node.dependencies else 'None'
            lines.append(f"  {task_id}: deps=[{deps}], priority={task_node.priority}")
        
        return '\n'.join(lines)