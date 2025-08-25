"""
Intelligent Conflict Resolution System
Advanced algorithms for detecting and resolving collaboration conflicts
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from uuid import UUID
from enum import Enum
from dataclasses import dataclass
import difflib
import re

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func

from .models import (
    ConflictResolution, ConflictType, ConflictStatus,
    Workspace, WorkspaceMember, ActivityFeed, ActivityType
)
from .sync_engine import FileEditOperation

logger = logging.getLogger(__name__)


class ResolutionStrategy(Enum):
    """Conflict resolution strategies"""
    AUTO_MERGE = "auto_merge"
    MANUAL_MERGE = "manual_merge"
    ACCEPT_INCOMING = "accept_incoming"
    ACCEPT_CURRENT = "accept_current"
    INTERACTIVE = "interactive"
    DEFER = "defer"


class ConflictSeverity(Enum):
    """Conflict severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ConflictContext:
    """Context information for conflict resolution"""
    file_path: str
    conflicting_operations: List[FileEditOperation]
    base_content: str
    current_content: str
    incoming_content: str
    affected_users: List[UUID]
    timestamp: datetime
    conflict_region: Dict[str, int]  # start_line, end_line, start_col, end_col


@dataclass
class ResolutionResult:
    """Result of conflict resolution attempt"""
    success: bool
    strategy_used: str
    resolved_content: Optional[str]
    confidence_score: float
    manual_intervention_required: bool
    resolution_notes: str
    affected_lines: List[int]


class ConflictResolver:
    """
    Intelligent conflict resolution system for collaborative editing.
    Uses advanced algorithms to detect, analyze, and resolve merge conflicts.
    """
    
    def __init__(self, db_session: Session):
        """
        Initialize conflict resolver.
        
        Args:
            db_session: Database session for operations
        """
        self.db = db_session
        
        # Resolution strategy preferences
        self._strategy_preferences: Dict[str, List[ResolutionStrategy]] = {
            'code': [
                ResolutionStrategy.AUTO_MERGE,
                ResolutionStrategy.MANUAL_MERGE,
                ResolutionStrategy.INTERACTIVE
            ],
            'text': [
                ResolutionStrategy.AUTO_MERGE,
                ResolutionStrategy.ACCEPT_INCOMING,
                ResolutionStrategy.MANUAL_MERGE
            ],
            'config': [
                ResolutionStrategy.MANUAL_MERGE,
                ResolutionStrategy.INTERACTIVE
            ]
        }
        
        # Conflict pattern recognition
        self._conflict_patterns = {
            'import_conflict': r'^(import|from)\s+.*',
            'function_conflict': r'^\s*def\s+\w+\s*\(',
            'class_conflict': r'^\s*class\s+\w+\s*[:\(]',
            'variable_assignment': r'^\s*\w+\s*=',
            'comment_conflict': r'^\s*(#|//|/\*|\*)',
            'whitespace_only': r'^\s*$'
        }
        
        # Auto-resolution rules
        self._auto_resolution_rules = [
            self._resolve_whitespace_conflicts,
            self._resolve_import_conflicts,
            self._resolve_comment_conflicts,
            self._resolve_non_overlapping_changes,
            self._resolve_append_only_changes
        ]
        
        logger.info("Conflict resolver initialized")
    
    async def detect_conflict(
        self,
        workspace_id: UUID,
        file_path: str,
        operations: List[FileEditOperation],
        base_content: str = ""
    ) -> Optional[ConflictResolution]:
        """
        Detect potential conflicts in file operations.
        
        Args:
            workspace_id: Workspace ID
            file_path: Path to the file
            operations: List of edit operations
            base_content: Original file content
            
        Returns:
            ConflictResolution object if conflict detected, None otherwise
        """
        logger.debug(f"Checking for conflicts in {file_path}")
        
        if len(operations) < 2:
            return None
        
        # Group operations by user and time
        user_operations = {}
        for op in operations:
            user_id = op.user_id
            if user_id not in user_operations:
                user_operations[user_id] = []
            user_operations[user_id].append(op)
        
        # Check for overlapping operations from different users
        conflicts = []
        users = list(user_operations.keys())
        
        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                user1_ops = user_operations[users[i]]
                user2_ops = user_operations[users[j]]
                
                conflict = await self._check_operation_overlap(
                    user1_ops, user2_ops, file_path
                )
                
                if conflict:
                    conflicts.append({
                        'users': [users[i], users[j]],
                        'operations': user1_ops + user2_ops,
                        'severity': conflict['severity']
                    })
        
        if not conflicts:
            return None
        
        # Create conflict resolution record
        conflict_type = self._determine_conflict_type(conflicts[0]['operations'])
        severity = self._calculate_conflict_severity(conflicts)
        
        conflict_resolution = ConflictResolution(
            workspace_id=workspace_id,
            conflict_type=conflict_type.value,
            affected_users=[c['users'][0] for c in conflicts] + [c['users'][1] for c in conflicts],
            affected_files=[file_path],
            conflict_data={
                'operations': [self._operation_to_dict(op) for op in operations],
                'base_content': base_content,
                'conflicts': conflicts,
                'severity': severity.value
            }
        )
        
        self.db.add(conflict_resolution)
        self.db.commit()
        
        logger.info(f"Conflict detected in {file_path}: {conflict_type.value} (severity: {severity.value})")
        return conflict_resolution
    
    async def resolve_conflict(
        self,
        conflict_id: UUID,
        strategy: Optional[ResolutionStrategy] = None,
        user_input: Optional[Dict[str, Any]] = None
    ) -> ResolutionResult:
        """
        Resolve a detected conflict using specified or automatic strategy.
        
        Args:
            conflict_id: Conflict resolution record ID
            strategy: Preferred resolution strategy
            user_input: Manual input for resolution
            
        Returns:
            ResolutionResult with resolution details
        """
        # Get conflict record
        conflict = self.db.query(ConflictResolution).filter(
            ConflictResolution.id == conflict_id
        ).first()
        
        if not conflict:
            return ResolutionResult(
                success=False,
                strategy_used="none",
                resolved_content=None,
                confidence_score=0.0,
                manual_intervention_required=True,
                resolution_notes="Conflict record not found",
                affected_lines=[]
            )
        
        logger.info(f"Resolving conflict {conflict_id} with strategy {strategy}")
        
        # Build conflict context
        context = await self._build_conflict_context(conflict)
        
        # Determine strategy if not provided
        if not strategy:
            strategy = await self._select_resolution_strategy(context)
        
        # Execute resolution
        result = await self._execute_resolution(context, strategy, user_input)
        
        # Update conflict record
        if result.success:
            conflict.status = ConflictStatus.RESOLVED.value
            conflict.resolved_at = datetime.now(timezone.utc)
            conflict.resolution_strategy = strategy.value
            conflict.resolution_notes = result.resolution_notes
        else:
            if result.manual_intervention_required:
                conflict.status = ConflictStatus.ESCALATED.value
                conflict.manual_intervention_required = True
            else:
                conflict.auto_resolution_attempts += 1
        
        self.db.commit()
        
        # Log resolution activity
        activity = ActivityFeed(
            workspace_id=conflict.workspace_id,
            user_id=conflict.affected_users[0] if conflict.affected_users else None,
            activity_type=ActivityType.CONFLICT_RESOLVED.value,
            description=f"Conflict in {context.file_path} resolved using {strategy.value}",
            metadata={
                "conflict_id": str(conflict_id),
                "strategy": strategy.value,
                "success": result.success,
                "confidence": result.confidence_score
            }
        )
        self.db.add(activity)
        self.db.commit()
        
        logger.info(f"Conflict {conflict_id} resolution: {result.success} (confidence: {result.confidence_score:.2f})")
        return result
    
    async def auto_resolve_conflicts(
        self,
        workspace_id: UUID,
        max_conflicts: int = 10
    ) -> List[ResolutionResult]:
        """
        Automatically resolve pending conflicts in workspace.
        
        Args:
            workspace_id: Workspace ID
            max_conflicts: Maximum number of conflicts to process
            
        Returns:
            List of resolution results
        """
        # Get pending conflicts
        pending_conflicts = (self.db.query(ConflictResolution)
                           .filter(
                               and_(
                                   ConflictResolution.workspace_id == workspace_id,
                                   ConflictResolution.status == ConflictStatus.PENDING.value,
                                   ConflictResolution.auto_resolution_attempts < 3
                               )
                           )
                           .order_by(ConflictResolution.detected_at)
                           .limit(max_conflicts)
                           .all())
        
        if not pending_conflicts:
            return []
        
        logger.info(f"Auto-resolving {len(pending_conflicts)} conflicts in workspace {workspace_id}")
        
        results = []
        for conflict in pending_conflicts:
            try:
                result = await self.resolve_conflict(conflict.id, ResolutionStrategy.AUTO_MERGE)
                results.append(result)
                
                # Small delay between resolutions
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error auto-resolving conflict {conflict.id}: {e}")
                results.append(ResolutionResult(
                    success=False,
                    strategy_used="auto_merge",
                    resolved_content=None,
                    confidence_score=0.0,
                    manual_intervention_required=True,
                    resolution_notes=f"Auto-resolution failed: {e}",
                    affected_lines=[]
                ))
        
        return results
    
    async def get_conflict_statistics(
        self,
        workspace_id: UUID,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get conflict statistics for workspace.
        
        Args:
            workspace_id: Workspace ID
            days: Number of days to analyze
            
        Returns:
            Dictionary with conflict statistics
        """
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        # Query conflicts in time period
        conflicts = (self.db.query(ConflictResolution)
                    .filter(
                        and_(
                            ConflictResolution.workspace_id == workspace_id,
                            ConflictResolution.detected_at >= cutoff_date
                        )
                    ).all())
        
        if not conflicts:
            return {
                "total_conflicts": 0,
                "resolved_conflicts": 0,
                "pending_conflicts": 0,
                "escalated_conflicts": 0,
                "auto_resolution_rate": 0.0,
                "avg_resolution_time": 0.0,
                "conflicts_by_type": {},
                "conflicts_by_file": {}
            }
        
        # Calculate statistics
        total_conflicts = len(conflicts)
        resolved_conflicts = len([c for c in conflicts if c.status == ConflictStatus.RESOLVED.value])
        pending_conflicts = len([c for c in conflicts if c.status == ConflictStatus.PENDING.value])
        escalated_conflicts = len([c for c in conflicts if c.status == ConflictStatus.ESCALATED.value])
        
        auto_resolved = len([c for c in conflicts 
                           if c.status == ConflictStatus.RESOLVED.value 
                           and c.resolution_strategy in [ResolutionStrategy.AUTO_MERGE.value]])
        
        auto_resolution_rate = (auto_resolved / resolved_conflicts) * 100 if resolved_conflicts > 0 else 0.0
        
        # Calculate average resolution time
        resolved_with_time = [c for c in conflicts if c.resolved_at and c.detected_at]
        if resolved_with_time:
            total_time = sum((c.resolved_at - c.detected_at).total_seconds() for c in resolved_with_time)
            avg_resolution_time = total_time / len(resolved_with_time) / 60  # minutes
        else:
            avg_resolution_time = 0.0
        
        # Group by type
        conflicts_by_type = {}
        for conflict in conflicts:
            conflict_type = conflict.conflict_type
            conflicts_by_type[conflict_type] = conflicts_by_type.get(conflict_type, 0) + 1
        
        # Group by file
        conflicts_by_file = {}
        for conflict in conflicts:
            for file_path in conflict.affected_files:
                conflicts_by_file[file_path] = conflicts_by_file.get(file_path, 0) + 1
        
        return {
            "total_conflicts": total_conflicts,
            "resolved_conflicts": resolved_conflicts,
            "pending_conflicts": pending_conflicts,
            "escalated_conflicts": escalated_conflicts,
            "auto_resolution_rate": round(auto_resolution_rate, 2),
            "avg_resolution_time": round(avg_resolution_time, 2),
            "conflicts_by_type": conflicts_by_type,
            "conflicts_by_file": dict(sorted(conflicts_by_file.items(), key=lambda x: x[1], reverse=True)[:10])
        }
    
    async def _check_operation_overlap(
        self,
        ops1: List[FileEditOperation],
        ops2: List[FileEditOperation],
        file_path: str
    ) -> Optional[Dict[str, Any]]:
        """Check if two sets of operations overlap"""
        overlaps = []
        
        for op1 in ops1:
            for op2 in ops2:
                # Check temporal overlap (operations within 30 seconds)
                time1 = datetime.fromisoformat(op1.timestamp)
                time2 = datetime.fromisoformat(op2.timestamp)
                time_diff = abs((time1 - time2).total_seconds())
                
                if time_diff > 30:  # Not concurrent
                    continue
                
                # Check spatial overlap
                line1 = op1.position.get('line', 0)
                col1 = op1.position.get('column', 0)
                line2 = op2.position.get('line', 0)
                col2 = op2.position.get('column', 0)
                
                # Simple overlap detection
                line_overlap = abs(line1 - line2) <= 2  # Within 2 lines
                
                if line_overlap:
                    severity = self._calculate_operation_severity(op1, op2)
                    overlaps.append({
                        'op1': op1,
                        'op2': op2,
                        'severity': severity,
                        'time_diff': time_diff
                    })
        
        if overlaps:
            max_severity = max(overlap['severity'] for overlap in overlaps)
            return {
                'overlaps': overlaps,
                'severity': max_severity
            }
        
        return None
    
    def _calculate_operation_severity(self, op1: FileEditOperation, op2: FileEditOperation) -> ConflictSeverity:
        """Calculate severity of conflict between two operations"""
        # Check if operations are on the same line
        if op1.position.get('line') == op2.position.get('line'):
            # Same line conflicts are more severe
            if op1.operation_type == 'delete' or op2.operation_type == 'delete':
                return ConflictSeverity.HIGH
            return ConflictSeverity.MEDIUM
        
        # Adjacent line conflicts
        line_diff = abs(op1.position.get('line', 0) - op2.position.get('line', 0))
        if line_diff <= 1:
            return ConflictSeverity.MEDIUM
        
        return ConflictSeverity.LOW
    
    def _determine_conflict_type(self, operations: List[FileEditOperation]) -> ConflictType:
        """Determine the type of conflict based on operations"""
        # Check if all operations are concurrent edits
        if all(op.operation_type in ['insert', 'delete', 'replace'] for op in operations):
            return ConflictType.CONCURRENT_EDIT
        
        # Check for merge conflicts (different versions)
        if len(set(op.user_id for op in operations)) > 1:
            return ConflictType.MERGE_CONFLICT
        
        return ConflictType.STATE_CONFLICT
    
    def _calculate_conflict_severity(self, conflicts: List[Dict[str, Any]]) -> ConflictSeverity:
        """Calculate overall conflict severity"""
        if not conflicts:
            return ConflictSeverity.LOW
        
        severities = [c['severity'] for c in conflicts]
        
        if ConflictSeverity.CRITICAL in severities:
            return ConflictSeverity.CRITICAL
        elif ConflictSeverity.HIGH in severities:
            return ConflictSeverity.HIGH
        elif ConflictSeverity.MEDIUM in severities:
            return ConflictSeverity.MEDIUM
        else:
            return ConflictSeverity.LOW
    
    async def _build_conflict_context(self, conflict: ConflictResolution) -> ConflictContext:
        """Build context for conflict resolution"""
        operations = [
            FileEditOperation(**op) for op in conflict.conflict_data.get('operations', [])
        ]
        
        file_path = conflict.affected_files[0] if conflict.affected_files else ""
        base_content = conflict.conflict_data.get('base_content', "")
        
        # Simulate applying operations to get current and incoming content
        current_content = base_content
        incoming_content = base_content
        
        # Apply operations to simulate different states
        for op in operations:
            if op.user_id == conflict.affected_users[0]:
                current_content = self._apply_operation_to_content(current_content, op)
            else:
                incoming_content = self._apply_operation_to_content(incoming_content, op)
        
        # Determine conflict region
        conflict_region = self._find_conflict_region(operations)
        
        return ConflictContext(
            file_path=file_path,
            conflicting_operations=operations,
            base_content=base_content,
            current_content=current_content,
            incoming_content=incoming_content,
            affected_users=[UUID(uid) for uid in conflict.affected_users],
            timestamp=conflict.detected_at,
            conflict_region=conflict_region
        )
    
    async def _select_resolution_strategy(self, context: ConflictContext) -> ResolutionStrategy:
        """Select best resolution strategy for conflict"""
        file_type = self._get_file_type(context.file_path)
        
        # Get preferred strategies for file type
        strategies = self._strategy_preferences.get(file_type, [ResolutionStrategy.MANUAL_MERGE])
        
        # Try auto-resolution rules first
        for rule in self._auto_resolution_rules:
            if await rule(context):
                return ResolutionStrategy.AUTO_MERGE
        
        # Return first preferred strategy
        return strategies[0]
    
    async def _execute_resolution(
        self,
        context: ConflictContext,
        strategy: ResolutionStrategy,
        user_input: Optional[Dict[str, Any]] = None
    ) -> ResolutionResult:
        """Execute conflict resolution with given strategy"""
        try:
            if strategy == ResolutionStrategy.AUTO_MERGE:
                return await self._auto_merge_resolution(context)
            elif strategy == ResolutionStrategy.ACCEPT_INCOMING:
                return self._accept_incoming_resolution(context)
            elif strategy == ResolutionStrategy.ACCEPT_CURRENT:
                return self._accept_current_resolution(context)
            elif strategy == ResolutionStrategy.MANUAL_MERGE:
                return await self._manual_merge_resolution(context, user_input)
            elif strategy == ResolutionStrategy.INTERACTIVE:
                return await self._interactive_resolution(context, user_input)
            else:
                return ResolutionResult(
                    success=False,
                    strategy_used=strategy.value,
                    resolved_content=None,
                    confidence_score=0.0,
                    manual_intervention_required=True,
                    resolution_notes="Strategy not implemented",
                    affected_lines=[]
                )
        
        except Exception as e:
            logger.error(f"Error executing resolution strategy {strategy}: {e}")
            return ResolutionResult(
                success=False,
                strategy_used=strategy.value,
                resolved_content=None,
                confidence_score=0.0,
                manual_intervention_required=True,
                resolution_notes=f"Resolution failed: {e}",
                affected_lines=[]
            )
    
    async def _auto_merge_resolution(self, context: ConflictContext) -> ResolutionResult:
        """Attempt automatic merge resolution"""
        # Try each auto-resolution rule
        for rule in self._auto_resolution_rules:
            result = await rule(context)
            if result:
                return ResolutionResult(
                    success=True,
                    strategy_used="auto_merge",
                    resolved_content=result['content'],
                    confidence_score=result['confidence'],
                    manual_intervention_required=False,
                    resolution_notes=result['notes'],
                    affected_lines=result.get('affected_lines', [])
                )
        
        # Fallback to three-way merge
        return await self._three_way_merge(context)
    
    async def _three_way_merge(self, context: ConflictContext) -> ResolutionResult:
        """Perform three-way merge of conflicting content"""
        base_lines = context.base_content.split('\n')
        current_lines = context.current_content.split('\n')
        incoming_lines = context.incoming_content.split('\n')
        
        # Use difflib to find differences
        current_diff = list(difflib.unified_diff(base_lines, current_lines, lineterm=''))
        incoming_diff = list(difflib.unified_diff(base_lines, incoming_lines, lineterm=''))
        
        # Simple merge logic
        merged_lines = base_lines.copy()
        confidence = 0.7
        
        # Apply non-conflicting changes
        try:
            # This is a simplified merge - in production, use more sophisticated algorithms
            if len(current_diff) == 0:
                merged_lines = incoming_lines
                confidence = 0.9
            elif len(incoming_diff) == 0:
                merged_lines = current_lines
                confidence = 0.9
            else:
                # Both have changes - attempt simple merge
                merged_lines = current_lines  # Bias toward current
                confidence = 0.5
            
            return ResolutionResult(
                success=True,
                strategy_used="three_way_merge",
                resolved_content='\n'.join(merged_lines),
                confidence_score=confidence,
                manual_intervention_required=confidence < 0.6,
                resolution_notes="Three-way merge completed",
                affected_lines=list(range(len(merged_lines)))
            )
        
        except Exception as e:
            return ResolutionResult(
                success=False,
                strategy_used="three_way_merge",
                resolved_content=None,
                confidence_score=0.0,
                manual_intervention_required=True,
                resolution_notes=f"Three-way merge failed: {e}",
                affected_lines=[]
            )
    
    def _accept_incoming_resolution(self, context: ConflictContext) -> ResolutionResult:
        """Accept incoming changes"""
        return ResolutionResult(
            success=True,
            strategy_used="accept_incoming",
            resolved_content=context.incoming_content,
            confidence_score=1.0,
            manual_intervention_required=False,
            resolution_notes="Accepted incoming changes",
            affected_lines=[]
        )
    
    def _accept_current_resolution(self, context: ConflictContext) -> ResolutionResult:
        """Accept current changes"""
        return ResolutionResult(
            success=True,
            strategy_used="accept_current",
            resolved_content=context.current_content,
            confidence_score=1.0,
            manual_intervention_required=False,
            resolution_notes="Accepted current changes",
            affected_lines=[]
        )
    
    async def _manual_merge_resolution(
        self,
        context: ConflictContext,
        user_input: Optional[Dict[str, Any]]
    ) -> ResolutionResult:
        """Handle manual merge resolution"""
        if not user_input or 'resolved_content' not in user_input:
            return ResolutionResult(
                success=False,
                strategy_used="manual_merge",
                resolved_content=None,
                confidence_score=0.0,
                manual_intervention_required=True,
                resolution_notes="Manual resolution required - awaiting user input",
                affected_lines=[]
            )
        
        resolved_content = user_input['resolved_content']
        confidence = user_input.get('confidence', 0.8)
        
        return ResolutionResult(
            success=True,
            strategy_used="manual_merge",
            resolved_content=resolved_content,
            confidence_score=confidence,
            manual_intervention_required=False,
            resolution_notes="Manually resolved by user",
            affected_lines=user_input.get('affected_lines', [])
        )
    
    async def _interactive_resolution(
        self,
        context: ConflictContext,
        user_input: Optional[Dict[str, Any]]
    ) -> ResolutionResult:
        """Handle interactive resolution with user guidance"""
        # This would integrate with UI for interactive conflict resolution
        return ResolutionResult(
            success=False,
            strategy_used="interactive",
            resolved_content=None,
            confidence_score=0.0,
            manual_intervention_required=True,
            resolution_notes="Interactive resolution not yet implemented",
            affected_lines=[]
        )
    
    # Auto-resolution rule implementations
    
    async def _resolve_whitespace_conflicts(self, context: ConflictContext) -> Optional[Dict[str, Any]]:
        """Resolve conflicts that only involve whitespace changes"""
        current_stripped = '\n'.join(line.strip() for line in context.current_content.split('\n'))
        incoming_stripped = '\n'.join(line.strip() for line in context.incoming_content.split('\n'))
        
        if current_stripped == incoming_stripped:
            # Only whitespace differences - prefer incoming
            return {
                'content': context.incoming_content,
                'confidence': 0.95,
                'notes': 'Resolved whitespace-only conflict'
            }
        return None
    
    async def _resolve_import_conflicts(self, context: ConflictContext) -> Optional[Dict[str, Any]]:
        """Resolve conflicts in import statements"""
        current_lines = context.current_content.split('\n')
        incoming_lines = context.incoming_content.split('\n')
        
        # Check if conflict is only in import section
        import_pattern = re.compile(r'^(import|from)\s+')
        
        current_imports = [line for line in current_lines if import_pattern.match(line.strip())]
        incoming_imports = [line for line in incoming_lines if import_pattern.match(line.strip())]
        
        # Merge unique imports
        all_imports = list(set(current_imports + incoming_imports))
        all_imports.sort()  # Sort imports
        
        # Replace import section
        merged_lines = []
        in_import_section = True
        
        for line in context.base_content.split('\n'):
            if import_pattern.match(line.strip()) and in_import_section:
                continue  # Skip original imports
            elif in_import_section and line.strip() == '':
                # End of import section
                merged_lines.extend(all_imports)
                merged_lines.append('')
                in_import_section = False
            else:
                merged_lines.append(line)
                in_import_section = False
        
        if all_imports and len(set(current_imports) | set(incoming_imports)) == len(all_imports):
            return {
                'content': '\n'.join(merged_lines),
                'confidence': 0.85,
                'notes': 'Merged import statements'
            }
        
        return None
    
    async def _resolve_comment_conflicts(self, context: ConflictContext) -> Optional[Dict[str, Any]]:
        """Resolve conflicts in comments"""
        current_lines = context.current_content.split('\n')
        incoming_lines = context.incoming_content.split('\n')
        
        comment_pattern = re.compile(r'^\s*(#|//|/\*|\*)')
        
        # Check if differences are only in comments
        current_non_comments = [line for line in current_lines if not comment_pattern.match(line)]
        incoming_non_comments = [line for line in incoming_lines if not comment_pattern.match(line)]
        
        if current_non_comments == incoming_non_comments:
            # Only comment differences - prefer incoming
            return {
                'content': context.incoming_content,
                'confidence': 0.8,
                'notes': 'Resolved comment-only conflict'
            }
        
        return None
    
    async def _resolve_non_overlapping_changes(self, context: ConflictContext) -> Optional[Dict[str, Any]]:
        """Resolve conflicts where changes don't actually overlap"""
        current_lines = context.current_content.split('\n')
        incoming_lines = context.incoming_content.split('\n')
        base_lines = context.base_content.split('\n')
        
        # Simple check for non-overlapping line changes
        current_diff_lines = set()
        incoming_diff_lines = set()
        
        for i, (base_line, current_line) in enumerate(zip(base_lines, current_lines)):
            if base_line != current_line:
                current_diff_lines.add(i)
        
        for i, (base_line, incoming_line) in enumerate(zip(base_lines, incoming_lines)):
            if base_line != incoming_line:
                incoming_diff_lines.add(i)
        
        # If no overlap in changed lines, merge both changes
        if not current_diff_lines & incoming_diff_lines:
            merged_lines = base_lines.copy()
            
            # Apply current changes
            for i in current_diff_lines:
                if i < len(current_lines):
                    merged_lines[i] = current_lines[i]
            
            # Apply incoming changes
            for i in incoming_diff_lines:
                if i < len(incoming_lines):
                    merged_lines[i] = incoming_lines[i]
            
            return {
                'content': '\n'.join(merged_lines),
                'confidence': 0.9,
                'notes': 'Merged non-overlapping changes',
                'affected_lines': list(current_diff_lines | incoming_diff_lines)
            }
        
        return None
    
    async def _resolve_append_only_changes(self, context: ConflictContext) -> Optional[Dict[str, Any]]:
        """Resolve conflicts where changes are append-only"""
        current_lines = context.current_content.split('\n')
        incoming_lines = context.incoming_content.split('\n')
        base_lines = context.base_content.split('\n')
        
        # Check if both are just appending to the base
        current_is_append = (len(current_lines) >= len(base_lines) and 
                           current_lines[:len(base_lines)] == base_lines)
        incoming_is_append = (len(incoming_lines) >= len(base_lines) and
                            incoming_lines[:len(base_lines)] == base_lines)
        
        if current_is_append and incoming_is_append:
            # Merge appended content
            current_appended = current_lines[len(base_lines):]
            incoming_appended = incoming_lines[len(base_lines):]
            
            merged_lines = base_lines + current_appended + incoming_appended
            
            return {
                'content': '\n'.join(merged_lines),
                'confidence': 0.85,
                'notes': 'Merged append-only changes'
            }
        
        return None
    
    # Helper methods
    
    def _get_file_type(self, file_path: str) -> str:
        """Determine file type from path"""
        if file_path.endswith(('.py', '.js', '.ts', '.java', '.cpp', '.c', '.go')):
            return 'code'
        elif file_path.endswith(('.md', '.txt', '.rst')):
            return 'text'
        elif file_path.endswith(('.json', '.yaml', '.yml', '.xml', '.ini', '.cfg')):
            return 'config'
        else:
            return 'text'
    
    def _apply_operation_to_content(self, content: str, operation: FileEditOperation) -> str:
        """Apply a file edit operation to content"""
        lines = content.split('\n')
        line_num = operation.position.get('line', 0)
        col_num = operation.position.get('column', 0)
        
        if line_num >= len(lines):
            # Extend lines if necessary
            lines.extend([''] * (line_num - len(lines) + 1))
        
        line = lines[line_num]
        
        if operation.operation_type == 'insert':
            new_line = line[:col_num] + operation.content + line[col_num:]
            lines[line_num] = new_line
        elif operation.operation_type == 'delete':
            end_col = min(col_num + operation.length, len(line))
            new_line = line[:col_num] + line[end_col:]
            lines[line_num] = new_line
        elif operation.operation_type == 'replace':
            end_col = min(col_num + operation.length, len(line))
            new_line = line[:col_num] + operation.content + line[end_col:]
            lines[line_num] = new_line
        
        return '\n'.join(lines)
    
    def _find_conflict_region(self, operations: List[FileEditOperation]) -> Dict[str, int]:
        """Find the region affected by conflicting operations"""
        lines = [op.position.get('line', 0) for op in operations]
        columns = [op.position.get('column', 0) for op in operations]
        
        return {
            'start_line': min(lines),
            'end_line': max(lines),
            'start_col': min(columns),
            'end_col': max(columns)
        }
    
    def _operation_to_dict(self, operation: FileEditOperation) -> Dict[str, Any]:
        """Convert FileEditOperation to dictionary"""
        return {
            'file_path': operation.file_path,
            'operation_type': operation.operation_type,
            'position': operation.position,
            'content': operation.content,
            'length': operation.length,
            'user_id': operation.user_id,
            'timestamp': operation.timestamp
        }