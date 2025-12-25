"""
Visualization utilities for circle topology transformations.

This module provides tools to visualize and analyze flip transformations
of circle topologies, as described in Section 2.2 of the paper.
"""

from typing import List, Set, Tuple, Dict
from circle_topology import CircleTopology


class CircleExpression:
    """
    Represents a circle topology using nested parentheses notation.
    
    The notation uses '(' and ')' to represent circles, where matching
    pairs represent a single circle boundary.
    """
    
    def __init__(self, expr: str):
        """Initialize with a parentheses expression."""
        self.expr = expr.strip()
        self._validate()
    
    def _validate(self):
        """Validate that the expression is well-formed."""
        depth = 0
        for c in self.expr:
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
                if depth < 0:
                    raise ValueError(f"Invalid expression: {self.expr}")
        if depth != 0:
            raise ValueError(f"Unmatched parentheses: {self.expr}")
    
    def count_circles(self) -> int:
        """Count the number of circle pairs in the expression."""
        return self.expr.count('(')
    
    def factor_count(self) -> int:
        """
        Count the number of factors (top-level groups).
        
        A factor is a maximal well-formed subexpression at depth 0.
        """
        factors = 0
        depth = 0
        for c in self.expr:
            if c == '(':
                if depth == 0:
                    factors += 1
                depth += 1
            elif c == ')':
                depth -= 1
        return factors
    
    def flip_transform(self) -> Set[str]:
        """
        Generate all expressions reachable by flip transformations.
        
        A flip transformation moves a factor to the outside by wrapping
        all other factors and reversing inside/outside.
        
        Returns:
            Set of expression strings reachable by flip operations
        """
        if not self.expr:
            return {self.expr}
        
        # Parse into factors
        factors = self._parse_factors()
        
        if len(factors) == 0:
            return {self.expr}
        
        if len(factors) == 1:
            # Single factor - can only flip itself
            # This is the identity unless we can unwrap and rewrap
            return {self.expr}
        
        # Generate flips by moving each factor outside
        results = {self.expr}  # Include original
        
        for i in range(len(factors)):
            # Take factor i and wrap all others
            other_factors = factors[:i] + factors[i+1:]
            if other_factors:
                flipped = f"({' '.join(other_factors)}) {factors[i]}"
                # Normalize spacing
                flipped = flipped.replace(' ', '')
                results.add(flipped)
        
        return results
    
    def _parse_factors(self) -> List[str]:
        """
        Parse the expression into top-level factors.
        
        Returns:
            List of factor expressions
        """
        factors = []
        depth = 0
        current = []
        
        for c in self.expr:
            if c == '(':
                if depth == 0 and current:
                    # New factor starting
                    factors.append(''.join(current))
                    current = []
                current.append(c)
                depth += 1
            elif c == ')':
                depth -= 1
                current.append(c)
                if depth == 0:
                    # Factor complete
                    factors.append(''.join(current))
                    current = []
        
        return factors
    
    def __str__(self):
        return self.expr
    
    def __repr__(self):
        return f"CircleExpression('{self.expr}')"
    
    def __eq__(self, other):
        if isinstance(other, CircleExpression):
            return self.expr == other.expr
        return False
    
    def __hash__(self):
        return hash(self.expr)


def find_flip_clusters(expressions: List[str]) -> List[Set[str]]:
    """
    Find clusters of expressions connected by flip transformations.
    
    Args:
        expressions: List of circle expressions as strings
        
    Returns:
        List of sets, where each set contains expressions in the same cluster
    """
    # Build adjacency graph
    expr_objs = [CircleExpression(e) for e in expressions]
    
    # Find all reachable expressions from each starting point
    visited = set()
    clusters = []
    
    for expr in expr_objs:
        if expr.expr in visited:
            continue
        
        # BFS to find all connected expressions
        cluster = set()
        queue = [expr.expr]
        cluster.add(expr.expr)
        
        while queue:
            current = queue.pop(0)
            ce = CircleExpression(current)
            
            # Get all flip transformations
            for flipped in ce.flip_transform():
                if flipped not in cluster and flipped in expressions:
                    cluster.add(flipped)
                    queue.append(flipped)
        
        visited.update(cluster)
        clusters.append(cluster)
    
    return clusters


def generate_c4_expressions() -> List[str]:
    """
    Generate all 9 expressions for C_4 (4 circles).
    
    Returns the expressions shown in the paper's Figure 1.
    """
    return [
        '()()()()',  # 4 factors
        '(())()()',  # 3 factors
        '((()))()',  # 2 factors
        '(()())()',  # 2 factors
        '((()()))',  # 1 factor
        '(())(())',  # 2 factors
        '((())())',  # 1 factor
        '(()()())',  # 1 factor
        '(((())))',  # 1 factor
    ]


def analyze_flip_structure(n: int) -> Dict[str, any]:
    """
    Analyze the flip transformation structure for n circles.
    
    Args:
        n: Number of circles
        
    Returns:
        Dictionary with analysis results
    """
    # For now, we provide the known cases from the paper
    if n == 4:
        exprs = generate_c4_expressions()
        clusters = find_flip_clusters(exprs)
        return {
            'n': n,
            'total_topologies': len(exprs),
            'clusters': clusters,
            'num_clusters': len(clusters),
            'cluster_sizes': [len(c) for c in clusters]
        }
    else:
        return {
            'n': n,
            'total_topologies': CircleTopology.non_intersecting_circles(n),
            'note': 'Cluster analysis not yet implemented for this n'
        }


def print_flip_analysis(n: int = 4):
    """
    Print analysis of flip transformation clusters.
    
    Args:
        n: Number of circles (default 4)
    """
    analysis = analyze_flip_structure(n)
    
    print(f"Flip Transformation Analysis for {n} circles")
    print("=" * 60)
    print(f"Total topologies: {analysis['total_topologies']}")
    
    if 'num_clusters' in analysis:
        print(f"Number of flip-equivalence clusters: {analysis['num_clusters']}")
        print(f"Cluster sizes: {analysis['cluster_sizes']}")
        print()
        print("Clusters:")
        for i, cluster in enumerate(analysis['clusters'], 1):
            print(f"  Cluster {i} (size {len(cluster)}):")
            for expr in sorted(cluster):
                ce = CircleExpression(expr)
                print(f"    {expr} [{ce.factor_count()} factors]")
        print()
        print("Note: Each cluster represents circle topologies that are")
        print("equivalent when embedded on a sphere surface.")
    else:
        print(f"Note: {analysis.get('note', 'No additional analysis available')}")


if __name__ == "__main__":
    # Demo the flip transformation analysis
    print_flip_analysis(4)
