import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as patches

# Create figure
plt.figure(figsize=(20, 15))
ax = plt.gca()
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Light pastel colors
colors = {
    'dag': '#FADBD8',
    'ivm': '#D6EAF8', 
    'xai': '#FCF3CF',
    'src': '#D5DBDB',
    'lse': '#FDEDEC',
    'dui': '#E8DAEF',
    'are': '#D1F2EB',
    'fmo': '#D6EAF8',
    'arrow': '#2C3E50'
}

# Title
plt.text(50, 97, 'PHANTOMNET++ SYSTEM ARCHITECTURE', 
         ha='center', va='center', fontsize=20, fontweight='bold', color='#2F2F2F')

# ==================== DAG MODULE ====================
dag_box = FancyBboxPatch((5, 70), 25, 25, boxstyle="round,pad=0.5", 
                                facecolor=colors['dag'], edgecolor='black', alpha=0.9)
ax.add_patch(dag_box)
plt.text(17.5, 82, 'DAG MODULE\nDynamic Attack Generator', 
         ha='center', va='center', fontsize=12, fontweight='bold')

dag_components = [
    (10, 77, 'Foolbox\n(PGD Attacks)'),
    (10, 73, 'AdvGAN\n(Wearables)'),
    (25, 77, 'Blender+NeRF\n(3D Objects)'),
    (25, 73, 'Real-world\nIntegration')
]
for x, y, text in dag_components:
    plt.text(x, y, text, ha='center', va='center', fontsize=9, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

# ==================== IVM MODULE ====================
ivm_box = FancyBboxPatch((40, 60), 30, 30, boxstyle="round,pad=0.5", 
                                facecolor=colors['ivm'], edgecolor='black', alpha=0.9)
ax.add_patch(ivm_box)
plt.text(55, 88, 'IVM MODULE\nIntelligent Vision System', 
         ha='center', va='center', fontsize=12, fontweight='bold')

ivm_components = [
    (45, 80, 'EfficientNet-Lite3\n(QKeras Quantized)'),
    (55, 80, 'YOLOv5s\n(TensorRT Optimized)'),
    (65, 80, 'Autoencoder\n(Jacobian Regularization)'),
    (55, 70, 'Multi-Agent Consensus\n(Byzantine Fault Tolerant)')
]
for x, y, text in ivm_components:
    plt.text(x, y, text, ha='center', va='center', fontsize=9, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

# ==================== XAI MODULE ====================
xai_box = FancyBboxPatch((80, 70), 15, 20, boxstyle="round,pad=0.5", 
                                facecolor=colors['xai'], edgecolor='black', alpha=0.9)
ax.add_patch(xai_box)
plt.text(87.5, 80, 'XAI MODULE\nExplainable AI', 
         ha='center', va='center', fontsize=12, fontweight='bold')
xai_components = [
    (87.5, 75, 'Grad-CAM\nVisual Explanations'),
    (87.5, 71, 'SHAP/LIME\nFeature Importance')
]
for x, y, text in xai_components:
    plt.text(x, y, text, ha='center', va='center', fontsize=9, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

# ==================== SRC MODULE ====================
src_box = FancyBboxPatch((5, 35), 20, 20, boxstyle="round,pad=0.5", 
                                facecolor=colors['src'], edgecolor='black', alpha=0.9)
ax.add_patch(src_box)
plt.text(15, 45, 'SRC MODULE\nSecure Coordination', 
         ha='center', va='center', fontsize=11, fontweight='bold')
src_components = [
    (15, 40, 'gRPC + TLS\nSecure Communication')
]
for x, y, text in src_components:
    plt.text(x, y, text, ha='center', va='center', fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

# ==================== LSE MODULE ====================
lse_box = FancyBboxPatch((35, 35), 20, 20, boxstyle="round,pad=0.5", 
                                facecolor=colors['lse'], edgecolor='black', alpha=0.9)
ax.add_patch(lse_box)
plt.text(45, 45, 'LSE MODULE\nLogging & SIEM Integration', 
         ha='center', va='center', fontsize=11, fontweight='bold')
lse_components = [
    (45, 40, 'ELK Stack\nReal-time Analytics'),
    (45, 36, 'Immutable Logging\nSHA-3 Hashing')
]
for x, y, text in lse_components:
    plt.text(x, y, text, ha='center', va='center', fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

# ==================== DUI MODULE ====================
dui_box = FancyBboxPatch((65, 35), 20, 20, boxstyle="round,pad=0.5", 
                                facecolor=colors['dui'], edgecolor='black', alpha=0.9)
ax.add_patch(dui_box)
plt.text(75, 45, 'DUI MODULE\nDashboard User Interface', 
         ha='center', va='center', fontsize=11, fontweight='bold')
dui_components = [
    (75, 40, 'React.js\nReal-time Visualization'),
    (75, 36, 'Role-Based Access Control')
]
for x, y, text in dui_components:
    plt.text(x, y, text, ha='center', va='center', fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

# ==================== ARE MODULE ====================
are_box = FancyBboxPatch((40, 5), 20, 20, boxstyle="round,pad=0.5", 
                                facecolor=colors['are'], edgecolor='black', alpha=0.9)
ax.add_patch(are_box)
plt.text(50, 15, 'ARE MODULE\nAutonomous Response Engine', 
         ha='center', va='center', fontsize=11, fontweight='bold')
are_components = [
    (50, 11, 'Policy-Driven Actions'),
    (50, 7, 'Threat Containment')
]
for x, y, text in are_components:
    plt.text(x, y, text, ha='center', va='center', fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

# ==================== FMO MODULE ====================
fmo_box = FancyBboxPatch((75, 5), 20, 20, boxstyle="round,pad=0.5", 
                                facecolor=colors['fmo'], edgecolor='black', alpha=0.9)
ax.add_patch(fmo_box)
plt.text(85, 15, 'FMO MODULE\nFramework Management', 
         ha='center', va='center', fontsize=11, fontweight='bold')
fmo_components = [
    (85, 11, 'Docker/Kubernetes'),
    (85, 7, 'Edge Optimization')
]
for x, y, text in fmo_components:
    plt.text(x, y, text, ha='center', va='center', fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

# ==================== ARROWS ====================
# DAG to IVM
plt.arrow(30, 82.5, 9, -2.5, head_width=1.2, head_length=2, fc=colors['arrow'], ec=colors['arrow'])
plt.text(35, 81, 'Adversarial\nExamples', ha='center', va='center', fontsize=9)

# SRC to IVM models
for x in [45, 55, 65]:
    plt.arrow(25, 45, x-25, 35, head_width=1.2, head_length=2, fc=colors['arrow'], ec=colors['arrow'])

# IVM to LSE
plt.arrow(55, 60, 0, -5, head_width=1.2, head_length=2, fc=colors['arrow'], ec=colors['arrow'])
plt.text(58, 53, 'Event Logging', ha='center', va='center', fontsize=9)

# IVM to XAI
plt.arrow(70, 80, 10, 0, head_width=1.2, head_length=2, fc=colors['arrow'], ec=colors['arrow'])
plt.text(75, 82, 'Predictions', ha='center', va='center', fontsize=9)

# XAI to DUI
plt.arrow(87.5, 70, 0, -25, head_width=1.2, head_length=2, fc=colors['arrow'], ec=colors['arrow'])

# LSE to DUI
plt.arrow(55, 35, 10, 0, head_width=1.2, head_length=2, fc=colors['arrow'], ec=colors['arrow'])

# ARE to FMO
plt.arrow(60, 15, 14, 0, head_width=1.2, head_length=2, fc=colors['arrow'], ec=colors['arrow'])

# Global Dockerization arrows → all modules to FMO
for (x, y) in [(17.5,82),(55,88),(87.5,80),(15,45),(45,45),(75,45),(50,15)]:
    plt.arrow(x, y, 30, -65, head_width=1.2, head_length=2, linestyle="dashed", 
              fc=colors['arrow'], ec=colors['arrow'], alpha=0.6)

# ==================== LEGEND ====================
legend_elements = [
    plt.Rectangle((0,0),1,1, facecolor=colors['dag'], edgecolor='black', label='DAG: Attack Generation'),
    plt.Rectangle((0,0),1,1, facecolor=colors['ivm'], edgecolor='black', label='IVM: Core Defense'),
    plt.Rectangle((0,0),1,1, facecolor=colors['xai'], edgecolor='black', label='XAI: Explainability'),
    plt.Rectangle((0,0),1,1, facecolor=colors['src'], edgecolor='black', label='SRC: Security'),
    plt.Rectangle((0,0),1,1, facecolor=colors['lse'], edgecolor='black', label='LSE: Logging'),
    plt.Rectangle((0,0),1,1, facecolor=colors['dui'], edgecolor='black', label='DUI: Dashboard'),
    plt.Rectangle((0,0),1,1, facecolor=colors['are'], edgecolor='black', label='ARE: Response'),
    plt.Rectangle((0,0),1,1, facecolor=colors['fmo'], edgecolor='black', label='FMO: Deployment')
]
legend = ax.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(0.02, 0.02), 
                   fontsize=10, title='MODULE LEGEND', title_fontsize=11)
legend.get_frame().set_facecolor('#FDFEFE')
legend.get_frame().set_alpha(0.9)

plt.tight_layout()
plt.savefig('phantomnet_architecture_updated.png', dpi=300, bbox_inches='tight')
plt.show()

