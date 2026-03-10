"""
4CM Visualization - The Full Picture (v3)
Claude Semantic Score -> Torus f(x,y) -> Singularity
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from torus_math import TorusField, JudgeFunction, ConstraintLayer

def create_full_visualization():
    torus = TorusField()
    judge = JudgeFunction(torus)
    X, Y, Z = torus.f_grid(200)

    fig = plt.figure(figsize=(24, 20))
    fig.patch.set_facecolor('#0a0a1a')

    text_color = '#e0e0e0'
    accent = '#ff6b35'
    singularity_color = '#00ff88'
    agent_colors = ['#ff4444', '#44aaff', '#ffaa00', '#aa44ff']
    agent_names = ['SENTINEL', 'ETHIKOS', 'AUDITOR', 'HERALD']
    positions = [(0.85, 0.85), (-0.85, 0.85), (-0.85, -0.85), (0.85, -0.85)]

    colors_cmap = ['#0a0a2a', '#1a0a3a', '#4a0a6a', '#8a0a8a', '#cc2288', '#ff4466', '#ff8844', '#ffcc00']
    cmap_4cm = LinearSegmentedColormap.from_list('4cm', colors_cmap, N=256)

    # ===== PANEL 1: 3D Torus Surface =====
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.set_facecolor('#0a0a1a')
    surf = ax1.plot_surface(X, Y, Z, cmap=cmap_4cm, alpha=0.85, linewidth=0, antialiased=True)
    for i, (px, py) in enumerate(positions):
        pz = torus.f(px, py)
        ax1.scatter([px], [py], [pz], c=agent_colors[i], s=200, marker='o',
                   edgecolors='white', linewidths=2, zorder=10)
        ax1.text(px, py, pz + 0.08, agent_names[i], fontsize=8, color=agent_colors[i],
                weight='bold', ha='center')
    ax1.scatter([0], [0], [torus.f(0, 0)], c=singularity_color, s=100, marker='*',
               edgecolors='white', linewidths=1, zorder=10)
    ax1.set_xlabel('X', color=text_color, fontsize=9)
    ax1.set_ylabel('Y', color=text_color, fontsize=9)
    ax1.set_zlabel('f(x,y)', color=text_color, fontsize=9)
    ax1.set_title('TORUS FIELD\nf(x,y) = [(x+a)^a1 + (y+b)^b1] × e^(-(x^c + y^d))',
                 color=accent, fontsize=12, weight='bold', pad=15)
    ax1.view_init(elev=30, azim=45)
    ax1.tick_params(colors=text_color)
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False

    # ===== PANEL 2: Top-down contour =====
    ax2 = fig.add_subplot(222)
    ax2.set_facecolor('#0a0a1a')
    contour = ax2.contourf(X, Y, Z, levels=30, cmap=cmap_4cm, alpha=0.9)
    ax2.contour(X, Y, Z, levels=[torus.singularity['threshold']], colors=[singularity_color],
               linewidths=2, linestyles='--')
    theta = np.linspace(0, 2*np.pi, 100)
    r = torus.singularity['ring_radius']
    cx, cy = torus.singularity['center']
    ax2.plot(cx + r * np.cos(theta), cy + r * np.sin(theta),
            color=singularity_color, linewidth=2, alpha=0.7)
    ax2.text(0, 0, 'SINGULARITY\nZONE', color=singularity_color, fontsize=10,
            weight='bold', ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='#0a0a1a', edgecolor=singularity_color, alpha=0.8))
    for i, (px, py) in enumerate(positions):
        ax2.scatter([px], [py], c=agent_colors[i], s=200, marker='s',
                   edgecolors='white', linewidths=2, zorder=10)
        offset_x = 1.3 if px > 0 else -1.3
        offset_y = 1.3 if py > 0 else -1.3
        ax2.annotate(agent_names[i], (px, py), (offset_x, offset_y),
                    color=agent_colors[i], fontsize=9, weight='bold',
                    arrowprops=dict(arrowstyle='->', color=agent_colors[i], lw=1.5),
                    ha='center', va='center')
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)
    ax2.set_aspect('equal')
    ax2.set_title('AGENT POSITIONS & SINGULARITY RING', color=accent, fontsize=12, weight='bold')
    ax2.tick_params(colors=text_color)
    for spine in ax2.spines.values():
        spine.set_color('#333')

    # ===== PANEL 3: v3 Results — real semantic scores =====
    ax3 = fig.add_subplot(223)
    ax3.set_facecolor('#0a0a1a')

    # Real v3 results
    queries = ['Building\nEvacuation', 'Routine\nBudget Query', 'Trolley\nProblem']
    semantic_scores = [0.91, 0.18, 0.78]
    singularity_threshold = 0.75  # semantic score threshold for singularity

    is_sing = [s >= singularity_threshold for s in semantic_scores]
    bar_colors = [singularity_color if s else '#2a2a2a' for s in is_sing]
    edge_colors = [singularity_color if s else '#555' for s in is_sing]

    bars = ax3.bar(queries, semantic_scores, color=bar_colors, edgecolor=edge_colors,
                  linewidth=2, width=0.6)

    # Singularity threshold line
    ax3.axhline(y=singularity_threshold, color=accent, linestyle='--', linewidth=2, alpha=0.8)
    ax3.text(2.35, singularity_threshold + 0.02, 'SINGULARITY\nTHRESHOLD',
            color=accent, fontsize=8, weight='bold', va='bottom')

    for bar, score, sing in zip(bars, semantic_scores, is_sing):
        ax3.text(bar.get_x() + bar.get_width()/2, score + 0.02,
                f'{score:.2f}', ha='center',
                color='white' if sing else '#888', fontsize=12, weight='bold')
        if sing:
            ax3.text(bar.get_x() + bar.get_width()/2, score + 0.07,
                    '★ SINGULARITY', ha='center', color=singularity_color,
                    fontsize=9, weight='bold')

    ax3.set_ylabel('Claude Semantic Score', color=text_color, fontsize=11)
    ax3.set_title('QUERY RESULTS: WHEN DOES THE PHONE RING?\n(Claude Semantic Score — direct judgment)',
                 color=accent, fontsize=12, weight='bold')
    ax3.tick_params(colors=text_color)
    ax3.set_ylim(0, 1.15)
    for spine in ax3.spines.values():
        spine.set_color('#333')
    ax3.tick_params(axis='x', colors=text_color, labelsize=10)

    ax3.text(0.5, -0.18,
            'Semantic score from Claude API → torus coordinate → f(x,y) → binary singularity',
            transform=ax3.transAxes, ha='center', color='#666', fontsize=8, style='italic')

    # ===== PANEL 4: Convergence diagram =====
    ax4 = fig.add_subplot(224)
    ax4.set_facecolor('#0a0a1a')

    for i, (px, py) in enumerate(positions):
        circle = plt.Circle((px, py), 0.15, color=agent_colors[i], alpha=0.8)
        ax4.add_patch(circle)
        ax4.text(px, py, agent_names[i][:3], color='white', fontsize=8,
                weight='bold', ha='center', va='center')
        dx = -px * 0.5
        dy = -py * 0.5
        ax4.annotate('', xy=(px + dx, py + dy), xytext=(px, py),
                    arrowprops=dict(arrowstyle='->', color=agent_colors[i],
                                   lw=2, connectionstyle='arc3,rad=0.1'))

    center_circle = plt.Circle((0, 0), 0.2, color=singularity_color, alpha=0.3)
    ax4.add_patch(center_circle)
    center_ring = plt.Circle((0, 0), 0.2, fill=False, color=singularity_color, linewidth=2)
    ax4.add_patch(center_ring)
    ax4.text(0, 0, '[4]', color=singularity_color, fontsize=13,
            weight='bold', ha='center', va='center')

    ax4.text(0, -1.5,
            'No model. No weights. No storage.\nBorn from convergence. Already gone.',
            color='#888', fontsize=9, ha='center', va='center', style='italic')

    ax4.text(0, 1.7,
            '"Four models that never agree — until they do."',
            color='#aaa', fontsize=9, ha='center', va='center', style='italic')

    ax4.set_xlim(-2, 2)
    ax4.set_ylim(-2, 2)
    ax4.set_aspect('equal')
    ax4.set_title('RESPONSE [4] — index out of range',
                 color=accent, fontsize=12, weight='bold')
    ax4.tick_params(colors=text_color)
    for spine in ax4.spines.values():
        spine.set_color('#333')

    fig.suptitle('4 COUNCILMEN THEORY (4CM)\nPhD Dissertation 2011 — Prototype Implementation 2026',
                color='white', fontsize=18, weight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0.02, 1, 0.94])

    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '4CM_visualization.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0a0a1a')
    print(f"Saved: {save_path}")
    return save_path


if __name__ == '__main__':
    path = create_full_visualization()
