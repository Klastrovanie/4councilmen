"""
4CM Theory - Torus Mathematical Framework
==========================================
PhD Dissertation (2011) - Core Mathematical Model

f(x,y) = [(x+a)^a1 + (y+b)^b1] * e^(-(x^c + y^d))

This creates a torus (donut) structure where:
- 4 peaks = orthogonal agent positions
- Central singularity ring = consensus zone
- f(x,y) value = consensus probability

The Judge Function: PURE MATH. No model. No weights. No AI.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List


class TorusField:
    """
    The mathematical field that constrains all 4CM operations.

    This is a mathematical space.
    Nothing is learned. Nothing is stored. Nothing persists.
    There is no weights, no shared model structure to begin with.

    This function defines the optimal conditions for a virtual space
    where four radically opposed agents — each with fixed perspectives,
    no shared weights, and no memory of each other — can arrive at
    consensus without negotiation.

    The field does not guide them. It only recognizes when they arrive.

    Even after they create a singularity, the model or weights won't be stored.
    """

    def __init__(self, a=0, b=0, a1=4, b1=4, c=8, d=8):
        """
        Parameters from PhD dissertation Figure 55

        a, b: Position shifts (default 0 - symmetric torus)
        a1, b1: Power exponents (4 = strong 4-peak activation)
        c, d: Decay rates (8 = sharp boundary collapse)
        """
        self.a = a
        self.b = b
        self.a1 = a1
        self.b1 = b1
        self.c = c
        self.d = d

        # Pre-compute singularity region
        self._singularity = self._find_singularity_region()

    def f(self, x: float, y: float) -> float:
        """
        The core function. Pure math.

        f(x,y) = [(x+a)^a1 + (y+b)^b1] * e^(-(x^c + y^d))
        """
        polynomial = (x + self.a)**self.a1 + (y + self.b)**self.b1
        decay = np.exp(-(x**self.c + y**self.d))
        return polynomial * decay

    def f_grid(self, resolution=200) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute f over entire grid"""
        x = np.linspace(-2, 2, resolution)
        y = np.linspace(-2, 2, resolution)
        X, Y = np.meshgrid(x, y)
        Z = np.vectorize(self.f)(X, Y)
        return X, Y, Z

    def _find_singularity_region(self, resolution=200, percentile=90):
        """
        Find the singularity ring - where consensus CAN exist.
        This is pre-computed because it's a property of the FIELD,
        not of any agent response.
        """
        X, Y, Z = self.f_grid(resolution)
        threshold = np.percentile(Z, percentile)
        mask = Z > threshold

        y_idx, x_idx = np.where(mask)

        if len(x_idx) == 0:
            return None

        center_x = np.mean(X[y_idx, x_idx])
        center_y = np.mean(Y[y_idx, x_idx])

        # Compute the ring radius (distance from center to peak density)
        distances = np.sqrt((X[y_idx, x_idx] - center_x)**2 +
                          (Y[y_idx, x_idx] - center_y)**2)
        ring_radius = np.median(distances)

        return {
            'center': (center_x, center_y),
            'ring_radius': ring_radius,
            'threshold': threshold,
            'max_value': np.max(Z),
            'peak_positions': self._find_peaks(X, Y, Z)
        }

    def _find_peaks(self, X, Y, Z, n_peaks=4):
        """Find the 4 peak positions on the torus"""
        peaks = []
        Z_copy = Z.copy()

        for _ in range(n_peaks):
            idx = np.unravel_index(np.argmax(Z_copy), Z_copy.shape)
            peak_x = X[idx]
            peak_y = Y[idx]
            peak_z = Z_copy[idx]
            peaks.append((peak_x, peak_y, peak_z))

            # Mask area around found peak
            dist = np.sqrt((X - peak_x)**2 + (Y - peak_y)**2)
            Z_copy[dist < 0.5] = 0

        return peaks

    @property
    def singularity(self):
        return self._singularity

    def is_in_singularity(self, x: float, y: float) -> bool:
        """Check if a point falls within the singularity region"""
        if self._singularity is None:
            return False
        value = self.f(x, y)
        return value > self._singularity['threshold']


class JudgeFunction:
    """
    THE JUDGE: Pure mathematical function.

    No model. No weights. No parameters to learn.
    No gradient. No backpropagation. No training.

    Takes 4 vectors, returns a single number.
    If that number exceeds threshold -> singularity exists.
    If not -> nothing happens. The 5th response was never born.

    Like the phone booth in Fight Club:
    The phone either rings, or it doesn't.
    There is no in-between.
    """

    def __init__(self, torus: TorusField, convergence_threshold: float = 0.75):
        self.torus = torus
        self.convergence_threshold = convergence_threshold

    def compute_convergence(self,
                           embeddings: List[np.ndarray],
                           positions: List[Tuple[float, float]]) -> Dict:
        """
        Given 4 response embeddings from 4 orthogonal agents,
        compute whether they converge at the singularity.

        This is the ONLY place where the 5th response can be born.
        And it's not AI. It's math.

        Args:
            embeddings: 4 embedding vectors (one per agent)
            positions: 4 (x,y) positions on torus (one per agent)

        Returns:
            Judgment dict with convergence metrics
        """
        assert len(embeddings) == 4, "Exactly 4 agents required"
        assert len(positions) == 4, "Exactly 4 positions required"

        # 1. Pairwise cosine similarities between all 4 agents
        similarities = []
        for i in range(4):
            for j in range(i+1, 4):
                sim = self._cosine_similarity(embeddings[i], embeddings[j])
                similarities.append(sim)

        # 6 pairwise similarities for 4 agents
        mean_similarity = np.mean(similarities)
        min_similarity = np.min(similarities)
        std_similarity = np.std(similarities)

        # 2. Compute centroid distance (how close all 4 vectors cluster)
        centroid = np.mean(embeddings, axis=0)
        centroid_distances = [np.linalg.norm(e - centroid) for e in embeddings]
        mean_centroid_dist = np.mean(centroid_distances)
        # Normalize: smaller distance = tighter cluster = higher agreement
        max_possible_dist = max(np.linalg.norm(e) for e in embeddings) * 2
        cluster_tightness = 1.0 - (mean_centroid_dist / max_possible_dist) if max_possible_dist > 0 else 0

        # 3. Combined convergence score:
        # - mean_similarity: average agreement between pairs
        # - min_similarity: weakest link (ALL must agree, not just some)
        # - cluster_tightness: how close all 4 are in vector space
        # Weight min_similarity heavily — 4CM requires UNANIMOUS convergence
        convergence_score = (
            0.25 * mean_similarity +
            0.50 * min_similarity +    # min is king — all 4 must agree
            0.25 * cluster_tightness
        )

        # 4. Map convergence score to torus coordinates
        convergence_x = self._similarity_to_coordinate(convergence_score)
        convergence_y = self._similarity_to_coordinate(convergence_score)

        # 3. Compute f(x,y) at the convergence point
        consensus_value = self.torus.f(convergence_x, convergence_y)

        # 4. Check singularity
        is_singularity = self.torus.is_in_singularity(convergence_x, convergence_y)

        # 5. Compute agent distances from their orthogonal positions to convergence
        agent_movements = []
        for pos in positions:
            dist = np.sqrt((pos[0] - convergence_x)**2 +
                         (pos[1] - convergence_y)**2)
            agent_movements.append(dist)

        return {
            'is_singularity': is_singularity,
            'consensus_value': float(consensus_value),
            'convergence_point': (float(convergence_x), float(convergence_y)),
            'mean_similarity': float(mean_similarity),
            'min_similarity': float(min_similarity),
            'std_similarity': float(std_similarity),
            'all_similarities': [float(s) for s in similarities],
            'agent_movements': [float(m) for m in agent_movements],
            'threshold': float(self.torus.singularity['threshold']) if self.torus.singularity else 0,
            # The ratio: how close to singularity (0 = far, 1 = at singularity, >1 = deep in)
            'singularity_ratio': float(consensus_value / self.torus.singularity['threshold']) if self.torus.singularity else 0
        }

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors"""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _similarity_to_coordinate(self, similarity: float) -> float:
        """
        Map similarity [0, 1] to torus coordinate.

        The torus peaks are at ~0.83 on each axis.
        f(x,y) is HIGH on the ring (radius ~0.85), LOW at center (0,0).

        So the mapping must be:
        - Low similarity  -> coordinate far from ring  -> f ~ 0
        - High similarity -> coordinate ON the ring    -> f is high
        - The transition is sharp (like a phone that either rings or doesn't)

        This maps the "consensus strength" onto the torus geometry.

        convergence_threshold: adjustable singularity gate (default 0.75)
        - 0.75: strict — requires strong semantic alignment
        - 0.50: relaxed — allows emergent convergence via dialogue
        """
        peak = 0.83

        if similarity < 0.40:
            # No agreement: far from ring -> f ~ 0
            return 1.5

        elif similarity >= self.convergence_threshold:
            # Strong semantic agreement: on the peak -> singularity zone
            return peak

        else:
            # Transition zone: [0.40, convergence_threshold] -> [1.4, peak]
            # Sharp sigmoid — phone either rings or it doesn't
            t = (similarity - 0.40) / (self.convergence_threshold - 0.40)
            t_sharp = 1.0 / (1.0 + np.exp(-12 * (t - 0.5)))
            return 1.4 - (1.4 - peak) * t_sharp

    def compute_convergence_from_semantic(self,
                                           semantic_score: float,
                                           positions: list,
                                           conclusion_score: float = None,
                                           reasoning_score: float = None) -> dict:
        """
        Compute torus convergence using two independent semantic axes.

        X-AXIS: conclusion_convergence  — do all 4 agents reach the same answer?
        Y-AXIS: reasoning_convergence   — do all 4 agents use the same logic?

        Each axis maps independently to torus coordinate via _similarity_to_coordinate().
        This means the 4 agents occupy all 4 quadrants of the torus simultaneously:

            agent_0: (+x_coord, +y_coord)  — quadrant 1
            agent_1: (-x_coord, +y_coord)  — quadrant 2
            agent_2: (-x_coord, -y_coord)  — quadrant 3
            agent_3: (+x_coord, -y_coord)  — quadrant 4

        Singularity only when BOTH axes converge — the ring forms in all 4 quadrants.

        If conclusion_score and reasoning_score not provided,
        falls back to semantic_score for both (backward compatible).
        """
        # Use two-axis scores if available, else fall back to single score
        x_score = conclusion_score if conclusion_score is not None else semantic_score
        y_score = reasoning_score if reasoning_score is not None else semantic_score

        # Map each axis independently to torus coordinate
        x_coord = self._similarity_to_coordinate(x_score)
        y_coord = self._similarity_to_coordinate(y_score)

        # Map each axis independently to torus coordinate.
        # The 4 agents are already assigned to 4 quadrants by design (ConstraintLayer).
        # agent_0:(+,+) agent_1:(-,+) agent_2:(-,-) agent_3:(+,-)
        # So +/- assignment is implicit — we only need the magnitude of convergence.

        # f(x,y) at the convergence point
        # Using the actual 2D torus — x and y are now independent
        consensus_value = self.torus.f(x_coord, y_coord)
        is_singularity = self.torus.is_in_singularity(x_coord, y_coord)

        # Agent positions in all 4 quadrants
        # Each agent moves toward its own quadrant's peak
        quadrant_signs = [(+1, +1), (-1, +1), (-1, -1), (+1, -1)]
        agent_movements = []
        for i, pos in enumerate(positions):
            sx, sy = quadrant_signs[i % 4]
            target_x = sx * x_coord
            target_y = sy * y_coord
            dist = float(np.sqrt((pos[0] - target_x)**2 + (pos[1] - target_y)**2))
            agent_movements.append(dist)

        threshold = float(self.torus.singularity['threshold']) if self.torus.singularity else 0
        ratio = float(consensus_value / threshold) if threshold > 0 else 0

        return {
            'is_singularity': is_singularity,
            'consensus_value': float(consensus_value),
            'convergence_point': (float(x_coord), float(y_coord)),
            'conclusion_score': float(x_score),
            'reasoning_score': float(y_score),
            'semantic_score_input': float(semantic_score),
            'agent_movements': agent_movements,
            'threshold': threshold,
            'singularity_ratio': ratio,
            # Backward compatibility
            'mean_similarity': float(semantic_score),
            'min_similarity': float(min(x_score, y_score)),
            'std_similarity': float(abs(x_score - y_score)),
            'all_similarities': [float(semantic_score)] * 6,
        }


class ConstraintLayer:
    """
    The invisible cage that keeps each agent in their orthogonal position.

    Each agent's output is mapped to a position on the torus.
    If an agent tries to "move toward center" (compromise),
    the constraint pushes it back to its corner.

    Agents are NOT allowed to seek consensus.
    Consensus must find THEM.
    """

    # The 4 orthogonal corners of the torus
    POSITIONS = {
        'agent_0': (0.85, 0.85),    # Top-right peak
        'agent_1': (-0.85, 0.85),   # Top-left peak
        'agent_2': (-0.85, -0.85),  # Bottom-left peak
        'agent_3': (0.85, -0.85),   # Bottom-right peak
    }

    def __init__(self, torus: TorusField, drift_tolerance: float = 0.3):
        """
        drift_tolerance: how far an agent can drift from its assigned peak.
        Lower = more orthogonal = harder to achieve consensus = more trustworthy when found.
        """
        self.torus = torus
        self.drift_tolerance = drift_tolerance

    def validate_agent_position(self, agent_id: str,
                                  embedding: np.ndarray,
                                  neutral_embedding: np.ndarray) -> Dict:
        """
        Check if an agent's response stays in its orthogonal zone.

        Compares agent's response embedding against a "neutral" baseline.
        If the agent is drifting toward neutral (compromising), flag it.

        Args:
            agent_id: 'agent_0' through 'agent_3'
            embedding: the agent's response embedding
            neutral_embedding: embedding of a neutral/balanced response

        Returns:
            Validation result
        """
        assigned_pos = self.POSITIONS[agent_id]

        # Measure how "neutral" the response is
        neutrality = self._cosine_similarity(embedding, neutral_embedding)

        # High neutrality = agent is compromising = BAD
        # Low neutrality = agent is orthogonal = GOOD (for 4CM)
        is_orthogonal = neutrality < (1.0 - self.drift_tolerance)

        # Compute effective position on torus
        # Orthogonal response stays near assigned peak
        # Neutral response drifts toward center
        drift = neutrality * 0.85  # Max drift = 0.85 (to center)
        effective_x = assigned_pos[0] * (1.0 - drift / 1.5)
        effective_y = assigned_pos[1] * (1.0 - drift / 1.5)

        return {
            'agent_id': agent_id,
            'assigned_position': assigned_pos,
            'effective_position': (effective_x, effective_y),
            'neutrality': float(neutrality),
            'is_orthogonal_enough': is_orthogonal,
            'drift_from_peak': float(np.sqrt(
                (effective_x - assigned_pos[0])**2 +
                (effective_y - assigned_pos[1])**2
            )),
            'f_value_at_position': float(self.torus.f(effective_x, effective_y))
        }

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
