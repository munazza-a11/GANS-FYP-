from graphviz import Digraph

# Initialize the flowchart
flow = Digraph('AdversarialDefenseSystem', filename='full_flowchart.gv')
flow.attr(rankdir='TB', size='15,20', bgcolor='white', fontsize='12')

# -------------------------
# 1. DATA SOURCES (Gray)
# -------------------------
with flow.subgraph(name='cluster_data') as c:
    c.attr(label='<<B>Data Sources</B>>', color='gray40', fontsize='14',
           style='filled', fillcolor='#F5F5F5')
    sources = [
        ('IMAGENET', 'ImageNet\n(1.2M images)'),
        ('CELEBA', 'CelebA-HQ\n(30K faces)'),
        ('SHAPENET', 'ShapeNet\n(3D models)'),
        ('COCO', 'COCO Dataset'),
        ('CIFAR10', 'CIFAR-10'),
        ('LAVAN', 'LaVAN\n(Physical attacks)')
    ]
    for node_id, label in sources:
        c.node(node_id, label, shape='cylinder', fillcolor='#E6E6FA')

# -------------------------
# 2. DAG MODULE (Red)
# -------------------------
with flow.subgraph(name='cluster_dag') as c:
    c.attr(label='<<B>DAG Module</B>\nDynamic Attack Generator',
           color='red3', fontsize='14', style='filled', fillcolor='#FFE6E6')

    tools = [
        ('FOOLBOX', 'Foolbox\n(PGD/FGSM)'),
        ('ADVGAN', 'AdvGAN\n(Wearables)'),
        ('BLENDER', 'Blender+NeRF\n(3D Attacks)')
    ]
    for node_id, label in tools:
        c.node(node_id, label, shape='box', fillcolor='#FFCCCC')

    outputs = [
        ('PATCHES', '2D Patches\n(.png)'),
        ('WEARABLES', 'Wearables\n(.pt)'),
        ('3DOBJ', '3D Objects\n(.obj)')
    ]
    for node_id, label in outputs:
        c.node(node_id, label, shape='folder', fillcolor='#FFDDDD')

    c.edges([
        ('IMAGENET', 'FOOLBOX'),
        ('CELEBA', 'ADVGAN'),
        ('SHAPENET', 'BLENDER'),
        ('FOOLBOX', 'PATCHES'),
        ('ADVGAN', 'WEARABLES'),
        ('BLENDER', '3DOBJ')
    ])

# -------------------------
# 3. IVM MODULE (Green)
# -------------------------
with flow.subgraph(name='cluster_ivm') as c:
    c.attr(label='<<B>IVM Module</B>\nIntelligent Vision',
           color='green4', fontsize='14', style='filled', fillcolor='#E6FFE6')

    models = [
        ('MOBILENET', 'Mobilenet'),
        ('YOLO', 'YOLOv5s'),
        ('AUTOENC', 'Autoencoder')
    ]
    for node_id, label in models:
        c.node(node_id, label, shape='component', fillcolor='#CCFFCC')

    agents = [
        ('AGENT1', 'Agent 1\n(Classifier)'),
        ('AGENT2', 'Agent 2\n(Detector)'),
        ('AGENT3', 'Agent 3\n(Anomaly)')
    ]
    for node_id, label in agents:
        c.node(node_id, label, shape='box', fillcolor='#DDFFDD')

    c.node('CONSENSUS', 'Consensus Engine\n(Byzantine Check)',
           shape='diamond', fillcolor='#FFFFCC')

    c.edges([
        ('MOBILENET', 'AGENT1'),
        ('YOLO', 'AGENT2'),
        ('AUTOENC', 'AGENT3'),
        ('AGENT1', 'CONSENSUS'),
        ('AGENT2', 'CONSENSUS'),
        ('AGENT3', 'CONSENSUS'),
        ('PATCHES', 'AGENT1'),
        ('WEARABLES', 'AGENT2'),
        ('3DOBJ', 'AGENT3')
    ])

# -------------------------
# 4. XAI MODULE (Blue)
# -------------------------
with flow.subgraph(name='cluster_xai') as c:
    c.attr(label='<<B>XAI Module</B>\nExplainable AI',
           color='blue', fontsize='14', style='filled', fillcolor='#E6F3FF')

    components = [
        ('LIME', 'LIME\n(Local Explanations)'),
        ('SHAP', 'SHAP\n(Feature Importance)'),
        ('GRADCAM', 'Grad-CAM\n(Visual Heatmaps)')
    ]
    for node_id, label in components:
        c.node(node_id, label, shape='note', fillcolor='#CCE5FF')

    c.edges([
        ('CONSENSUS', 'LIME'),
        ('CONSENSUS', 'SHAP'),
        ('CONSENSUS', 'GRADCAM')
    ])

# -------------------------
# 5. SRC & LSE MODULES (Purple)
# -------------------------
with flow.subgraph(name='cluster_security') as c:
    c.attr(label='<<B>Security Modules</B>>',
           color='purple', fontsize='14', style='filled', fillcolor='#F0E6FF')

    c.node('GRPC', 'gRPC+TLS\n(RSA/AES)', shape='box', fillcolor='#E0CCFF')
    c.node('POLICY', 'Policy Engine\n(Isolate/Switch)', shape='box3d', fillcolor='#DDBBFF')
    c.node('ELK', 'ELK Stack\n(Log Analysis)', shape='folder', fillcolor='#EECCFF')
    c.node('QLDB', 'Immutable Logs\n(SHA-3)', shape='folder', fillcolor='#EECCFF')

    c.edges([
        ('CONSENSUS', 'GRPC'),
        ('GRPC', 'POLICY'),
        ('POLICY', 'ELK'),
        ('POLICY', 'QLDB')
    ])

# -------------------------
# 6. ARE MODULE (Cyan)
# -------------------------
with flow.subgraph(name='cluster_are') as c:
    c.attr(label='<<B>ARE Module</B>\nAutonomous Response Engine',
           color='cyan4', fontsize='14', style='filled', fillcolor='#E0FFFF')

    c.node('ARE', 'Policy-driven Actions\nThreat Containment',
           shape='box', fillcolor='#CCFFFF')

# Connections: Security -> ARE -> FMO
flow.edge('POLICY', 'ARE', label='Response Commands')
flow.edge('ARE', 'DOCKER', label='Deployment Commands')

# -------------------------
# 7. DUI MODULE (Orange)
# -------------------------
with flow.subgraph(name='cluster_ui') as c:
    c.attr(label='<<B>DUI Module</B>\nDashboard',
           color='darkorange', fontsize='14', style='filled', fillcolor='#FFF0E6')

    components = [
        ('REACT', 'React.js\n(Real-time UI)'),
        ('THREATVIZ', 'Threat Visualizer\n(Grad-CAM)'),
        ('RBAC', 'Role-Based Access\n(Admin/Dev/Analyst)')
    ]
    for node_id, label in components:
        c.node(node_id, label, shape='tab', fillcolor='#FFDDCC')

    c.edges([
        ('ELK', 'REACT'),
        ('GRADCAM', 'THREATVIZ'),
        ('REACT', 'RBAC')
    ])

# -------------------------
# 8. FMO MODULE (Brown)
# -------------------------
with flow.subgraph(name='cluster_ops') as c:
    c.attr(label='<<B>FMO Module</B>\nDeployment',
           color='brown', fontsize='14', style='filled', fillcolor='#F5E6D9')

    components = [
        ('DOCKER', 'Docker Containers'),
        ('K8S', 'Kubernetes\n(Orchestration)'),
        ('EDGE', 'Edge Deployment\n(RPi/Jetson)')
    ]
    for node_id, label in components:
        c.node(node_id, label, shape='box', fillcolor='#EEDDCC')

    c.edges([
        ('EFFNET', 'DOCKER'),
        ('YOLO', 'K8S'),
        ('AUTOENC', 'EDGE')
    ])

# -------------------------
# RENDER & SAVE
# -------------------------
flow.format = 'png'
flow.render('full_adversarial_defense_flowchart', view=True, cleanup=True)

print("Flowchart saved as 'full_adversarial_defense_flowchart.png'")

