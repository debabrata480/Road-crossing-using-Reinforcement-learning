"""
Workflow Diagram Generator for DQN Crossy Road Training System

This script generates a visual workflow diagram showing how:
- Model (DQN neural network)
- Agent (DQNAgent with replay buffer)
- Environment (CrossyEnv game)
- Train function (training loop)
- Test function (evaluation)
- dqn_crossy.pth (saved model file)

all work together in the reinforcement learning pipeline.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch
import numpy as np

def create_workflow_diagram():
    """Create a comprehensive workflow diagram showing the DQN training and testing pipeline."""
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Define colors
    colors = {
        'model': '#4A90E2',      # Blue
        'agent': '#50C878',      # Green
        'env': '#FF6B6B',        # Red
        'train': '#FFA500',      # Orange
        'test': '#9B59B6',       # Purple
        'file': '#E67E22',       # Dark Orange
        'data': '#34495E',       # Dark Gray
        'process': '#95A5A6'     # Light Gray
    }
    
    # Title
    ax.text(5, 11.5, 'DQN Crossy Road: Training & Testing Workflow', 
            ha='center', va='center', fontsize=20, fontweight='bold')
    
    # ===== INITIALIZATION PHASE =====
    ax.text(5, 10.5, 'INITIALIZATION', ha='center', va='center', 
            fontsize=14, fontweight='bold', style='italic')
    
    # Model box
    model_box = FancyBboxPatch((1, 9), 1.5, 0.8, boxstyle="round,pad=0.1", 
                                facecolor=colors['model'], edgecolor='black', linewidth=2)
    ax.add_patch(model_box)
    ax.text(1.75, 9.6, 'MODEL\n(DQN)', ha='center', va='center', 
            fontsize=10, fontweight='bold', color='white')
    ax.text(1.75, 9.2, 'model.py\nNeural Network', ha='center', va='center', 
            fontsize=8, color='white')
    
    # Agent box
    agent_box = FancyBboxPatch((4, 9), 1.5, 0.8, boxstyle="round,pad=0.1", 
                               facecolor=colors['agent'], edgecolor='black', linewidth=2)
    ax.add_patch(agent_box)
    ax.text(4.75, 9.6, 'AGENT', ha='center', va='center', 
            fontsize=10, fontweight='bold', color='white')
    ax.text(4.75, 9.2, 'agent.py\nDQNAgent + ReplayBuffer', ha='center', va='center', 
            fontsize=8, color='white')
    
    # Environment box
    env_box = FancyBboxPatch((7, 9), 1.5, 0.8, boxstyle="round,pad=0.1", 
                             facecolor=colors['env'], edgecolor='black', linewidth=2)
    ax.add_patch(env_box)
    ax.text(7.75, 9.6, 'ENVIRONMENT', ha='center', va='center', 
            fontsize=10, fontweight='bold', color='white')
    ax.text(7.75, 9.2, 'game_crossy.py\nCrossyEnv', ha='center', va='center', 
            fontsize=8, color='white')
    
    # Connections: Model -> Agent
    arrow1 = FancyArrowPatch((2.5, 9.4), (4, 9.4), 
                             arrowstyle='->', mutation_scale=20, 
                             linewidth=2, color='black')
    ax.add_patch(arrow1)
    ax.text(3.25, 9.6, 'uses', ha='center', va='center', fontsize=8)
    
    # ===== TRAINING PHASE =====
    ax.text(5, 7.5, 'TRAINING PHASE (train_crossy.py)', ha='center', va='center', 
            fontsize=14, fontweight='bold', style='italic')
    
    # Training loop box
    train_box = FancyBboxPatch((2.5, 6.5), 5, 0.8, boxstyle="round,pad=0.1", 
                               facecolor=colors['train'], edgecolor='black', linewidth=2)
    ax.add_patch(train_box)
    ax.text(5, 6.9, 'TRAINING LOOP', ha='center', va='center', 
            fontsize=11, fontweight='bold', color='white')
    ax.text(5, 6.5, 'For each episode: reset → select_action → step → optimize → save', 
            ha='center', va='center', fontsize=9, color='white')
    
    # Training flow (detailed steps)
    steps_y = 5.5
    step_height = 0.35
    
    # Step 1: Reset
    step1 = FancyBboxPatch((1, steps_y), 1.8, step_height, boxstyle="round,pad=0.05", 
                           facecolor=colors['process'], edgecolor='black', linewidth=1)
    ax.add_patch(step1)
    ax.text(1.9, steps_y + step_height/2, '1. env.reset()', ha='center', va='center', 
            fontsize=8, fontweight='bold')
    
    # Step 2: Select Action
    step2 = FancyBboxPatch((3, steps_y), 1.8, step_height, boxstyle="round,pad=0.05", 
                           facecolor=colors['process'], edgecolor='black', linewidth=1)
    ax.add_patch(step2)
    ax.text(3.9, steps_y + step_height/2, '2. agent.select_action(state)', ha='center', 
            va='center', fontsize=8, fontweight='bold')
    
    # Step 3: Environment Step
    step3 = FancyBboxPatch((5, steps_y), 1.8, step_height, boxstyle="round,pad=0.05", 
                           facecolor=colors['process'], edgecolor='black', linewidth=1)
    ax.add_patch(step3)
    ax.text(5.9, steps_y + step_height/2, '3. env.step(action)', ha='center', 
            va='center', fontsize=8, fontweight='bold')
    
    # Step 4: Store & Optimize
    step4 = FancyBboxPatch((7, steps_y), 1.8, step_height, boxstyle="round,pad=0.05", 
                           facecolor=colors['process'], edgecolor='black', linewidth=1)
    ax.add_patch(step4)
    ax.text(7.9, steps_y + step_height/2, '4. optimize()', ha='center', va='center', 
            fontsize=8, fontweight='bold')
    
    # Arrows between steps
    for x in [2.8, 4.8, 6.8]:
        arrow = FancyArrowPatch((x, steps_y + step_height/2), (x + 0.2, steps_y + step_height/2), 
                               arrowstyle='->', mutation_scale=15, linewidth=1.5, color='black')
        ax.add_patch(arrow)
    
    # Detailed sub-steps
    detail_y = 4.8
    detail_height = 0.25
    
    # Step 2 details
    detail2 = FancyBboxPatch((2.5, detail_y), 3, detail_height, boxstyle="round,pad=0.05", 
                            facecolor='#ECF0F1', edgecolor='gray', linewidth=1)
    ax.add_patch(detail2)
    ax.text(4, detail_y + detail_height/2, 'Model predicts Q-values → Agent selects best action (ε-greedy)', 
            ha='center', va='center', fontsize=7, style='italic')
    
    # Step 3 details
    detail3 = FancyBboxPatch((2.5, detail_y - 0.3), 3, detail_height, boxstyle="round,pad=0.05", 
                            facecolor='#ECF0F1', edgecolor='gray', linewidth=1)
    ax.add_patch(detail3)
    ax.text(4, detail_y - 0.3 + detail_height/2, 'Environment returns: (next_state, reward, done, info)', 
            ha='center', va='center', fontsize=7, style='italic')
    
    # Step 4 details
    detail4 = FancyBboxPatch((6, detail_y), 3, detail_height, boxstyle="round,pad=0.05", 
                            facecolor='#ECF0F1', edgecolor='gray', linewidth=1)
    ax.add_patch(detail4)
    ax.text(7.5, detail_y + detail_height/2, 'Store transition in ReplayBuffer → Sample batch → Update Q-network', 
            ha='center', va='center', fontsize=7, style='italic')
    
    # Save model file
    file_y = 3.8
    file_box = FancyBboxPatch((3.5, file_y), 3, 0.5, boxstyle="round,pad=0.1", 
                             facecolor=colors['file'], edgecolor='black', linewidth=2)
    ax.add_patch(file_box)
    ax.text(5, file_y + 0.25, 'dqn_crossy.pth (Saved Model Weights)', ha='center', 
            va='center', fontsize=10, fontweight='bold', color='white')
    
    # Arrow from training to file
    arrow_save = FancyArrowPatch((5, 6.5), (5, file_y + 0.5), 
                                arrowstyle='->', mutation_scale=20, 
                                linewidth=2, color=colors['file'], linestyle='--')
    ax.add_patch(arrow_save)
    ax.text(5.5, (6.5 + file_y + 0.5)/2, 'agent.save()', ha='left', va='center', 
            fontsize=8, color=colors['file'], fontweight='bold')
    
    # ===== TESTING PHASE =====
    ax.text(5, 3, 'TESTING PHASE (test_crossy.py)', ha='center', va='center', 
            fontsize=14, fontweight='bold', style='italic')
    
    # Testing loop box
    test_box = FancyBboxPatch((2.5, 2), 5, 0.8, boxstyle="round,pad=0.1", 
                              facecolor=colors['test'], edgecolor='black', linewidth=2)
    ax.add_patch(test_box)
    ax.text(5, 2.4, 'TESTING LOOP', ha='center', va='center', 
            fontsize=11, fontweight='bold', color='white')
    ax.text(5, 2, 'Load model → Run episodes with ε=0 (greedy policy)', 
            ha='center', va='center', fontsize=9, color='white')
    
    # Arrow from file to test
    arrow_load = FancyArrowPatch((5, file_y), (5, 2.8), 
                                arrowstyle='->', mutation_scale=20, 
                                linewidth=2, color=colors['test'], linestyle='--')
    ax.add_patch(arrow_load)
    ax.text(5.5, (file_y + 2.8)/2, 'agent.load()', ha='left', va='center', 
            fontsize=8, color=colors['test'], fontweight='bold')
    
    # Test flow
    test_y = 1.2
    test_step1 = FancyBboxPatch((1, test_y), 1.8, 0.3, boxstyle="round,pad=0.05", 
                                facecolor=colors['process'], edgecolor='black', linewidth=1)
    ax.add_patch(test_step1)
    ax.text(1.9, test_y + 0.15, 'Load model', ha='center', va='center', fontsize=8)
    
    test_step2 = FancyBboxPatch((3, test_y), 1.8, 0.3, boxstyle="round,pad=0.05", 
                                facecolor=colors['process'], edgecolor='black', linewidth=1)
    ax.add_patch(test_step2)
    ax.text(3.9, test_y + 0.15, 'Select action', ha='center', va='center', fontsize=8)
    
    test_step3 = FancyBboxPatch((5, test_y), 1.8, 0.3, boxstyle="round,pad=0.05", 
                                facecolor=colors['process'], edgecolor='black', linewidth=1)
    ax.add_patch(test_step3)
    ax.text(5.9, test_y + 0.15, 'env.step()', ha='center', va='center', fontsize=8)
    
    test_step4 = FancyBboxPatch((7, test_y), 1.8, 0.3, boxstyle="round,pad=0.05", 
                                facecolor=colors['process'], edgecolor='black', linewidth=1)
    ax.add_patch(test_step4)
    ax.text(7.9, test_y + 0.15, 'Render & Evaluate', ha='center', va='center', fontsize=8)
    
    # Arrows between test steps
    for x in [2.8, 4.8, 6.8]:
        arrow = FancyArrowPatch((x, test_y + 0.15), (x + 0.2, test_y + 0.15), 
                               arrowstyle='->', mutation_scale=15, linewidth=1.5, color='black')
        ax.add_patch(arrow)
    
    # ===== KEY DATA FLOWS =====
    # State flow
    ax.text(0.5, 8.5, 'State\n(observation)', ha='center', va='center', 
            fontsize=8, bbox=dict(boxstyle='round', facecolor=colors['data'], alpha=0.7))
    
    # Action flow
    ax.text(0.5, 7.5, 'Action\n(0-4)', ha='center', va='center', 
            fontsize=8, bbox=dict(boxstyle='round', facecolor=colors['data'], alpha=0.7))
    
    # Reward flow
    ax.text(0.5, 6.5, 'Reward\n(+/-)', ha='center', va='center', 
            fontsize=8, bbox=dict(boxstyle='round', facecolor=colors['data'], alpha=0.7))
    
    # ===== LEGEND =====
    legend_y = 0.3
    legend_items = [
        ('Model', colors['model']),
        ('Agent', colors['agent']),
        ('Environment', colors['env']),
        ('Training', colors['train']),
        ('Testing', colors['test']),
        ('Model File', colors['file']),
        ('Process', colors['process'])
    ]
    
    legend_x = 1
    for i, (label, color) in enumerate(legend_items):
        rect = FancyBboxPatch((legend_x + i*1.2, legend_y), 1, 0.2, 
                             boxstyle="round,pad=0.05", facecolor=color, 
                             edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        ax.text(legend_x + i*1.2 + 0.5, legend_y + 0.1, label, 
                ha='center', va='center', fontsize=7, fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_mermaid_diagram():
    """Generate Mermaid diagram code for documentation."""
    mermaid_code = """
```mermaid
graph TB
    subgraph Init["INITIALIZATION"]
        Model[Model<br/>DQN Neural Network<br/>model.py]
        Agent[Agent<br/>DQNAgent + ReplayBuffer<br/>agent.py]
        Env[Environment<br/>CrossyEnv<br/>game_crossy.py]
        Model -->|uses| Agent
    end
    
    subgraph Train["TRAINING PHASE (train_crossy.py)"]
        TrainLoop[Training Loop<br/>For each episode]
        Step1[1. env.reset<br/>Get initial state]
        Step2[2. agent.select_action<br/>Model predicts Q-values<br/>ε-greedy policy]
        Step3[3. env.step<br/>Returns: next_state,<br/>reward, done, info]
        Step4[4. agent.optimize<br/>Store in ReplayBuffer<br/>Sample batch<br/>Update Q-network]
        Save[agent.save<br/>Save to file]
        
        TrainLoop --> Step1
        Step1 --> Step2
        Step2 --> Step3
        Step3 --> Step4
        Step4 -->|repeat| TrainLoop
        Step4 -->|every 50 episodes| Save
    end
    
    subgraph File["MODEL FILE"]
        ModelFile[dqn_crossy.pth<br/>Saved Model Weights]
    end
    
    subgraph Test["TESTING PHASE (test_crossy.py)"]
        Load[agent.load<br/>Load from file]
        TestLoop[Testing Loop<br/>For each episode]
        TestStep1[Load model]
        TestStep2[Select action<br/>ε=0 greedy]
        TestStep3[env.step<br/>Execute action]
        TestStep4[Render & Evaluate<br/>Calculate metrics]
        
        Load --> TestLoop
        TestLoop --> TestStep1
        TestStep1 --> TestStep2
        TestStep2 --> TestStep3
        TestStep3 -->|repeat| TestLoop
        TestStep3 --> TestStep4
    end
    
    Agent -->|creates| TrainLoop
    Env -->|interacts with| TrainLoop
    Save -->|writes| ModelFile
    ModelFile -->|reads| Load
    Agent -->|uses| TestLoop
    Env -->|interacts with| TestLoop
    
    style Model fill:#4A90E2,stroke:#333,stroke-width:2px,color:#fff
    style Agent fill:#50C878,stroke:#333,stroke-width:2px,color:#fff
    style Env fill:#FF6B6B,stroke:#333,stroke-width:2px,color:#fff
    style TrainLoop fill:#FFA500,stroke:#333,stroke-width:2px,color:#fff
    style TestLoop fill:#9B59B6,stroke:#333,stroke-width:2px,color:#fff
    style ModelFile fill:#E67E22,stroke:#333,stroke-width:2px,color:#fff
```
"""
    return mermaid_code

def create_architecture_diagram():
    """Generate Mermaid architecture diagram showing component structure."""
    mermaid_code = """
```mermaid
graph TB
    subgraph ModelArch["DQN Model Architecture (model.py)"]
        Input[Input Layer<br/>Observation Shape<br/>12-dimensional vector]
        FC1[Linear Layer 1<br/>12 → 256<br/>+ ReLU]
        FC2[Linear Layer 2<br/>256 → 256<br/>+ ReLU]
        Output[Output Layer<br/>256 → 5<br/>Q-values per action]
        
        Input --> FC1
        FC1 --> FC2
        FC2 --> Output
        
        Note1["Note: For image input,<br/>uses Conv2D layers<br/>before FC layers"]
    end
    
    subgraph AgentArch["DQN Agent Architecture (agent.py)"]
        PolicyNet[Policy Network<br/>DQN Model<br/>Updated continuously]
        TargetNet[Target Network<br/>DQN Model<br/>Updated every N steps]
        ReplayBuffer[Replay Buffer<br/>Stores transitions<br/>Capacity: 100,000]
        Optimizer[Adam Optimizer<br/>Learning Rate: 1e-3<br/>Updates Policy Net]
        EpsilonGreedy[Epsilon-Greedy<br/>Action Selection<br/>ε: 1.0 → 0.05]
        
        PolicyNet -->|Q-values| EpsilonGreedy
        ReplayBuffer -->|Sample batch| Optimizer
        PolicyNet -->|Forward pass| Optimizer
        TargetNet -->|Target Q-values| Optimizer
        Optimizer -->|Update weights| PolicyNet
        PolicyNet -.->|Copy weights every N steps| TargetNet
        
        Methods[Agent Methods<br/>- select_action<br/>- push_transition<br/>- optimize<br/>- save/load]
    end
    
    subgraph EnvArch["Environment Architecture (game_crossy.py)"]
        GameState[Game State<br/>- Player position x,z<br/>- Car positions x,z<br/>- Car speeds]
        ActionSpace[Action Space<br/>5 actions:<br/>0: Left<br/>1: Right<br/>2: Up<br/>3: Down<br/>4: Stay]
        RewardSystem[Reward System<br/>- Step penalty: -0.01<br/>- Collision: -10.0<br/>- Out of bounds: -50.0<br/>- Success: +20.0]
        Termination[Termination Conditions<br/>- Collision with car<br/>- Out of bounds<br/>- Reached finish line]
        
        GameState -->|Observation| ObservationOutput[Observation<br/>12D vector:<br/>[player_x, player_z,<br/>car1_x, car1_z, ...]]
        ActionSpace -->|Action input| GameState
        GameState --> RewardSystem
        RewardSystem --> RewardOutput[Reward<br/>Float value]
        GameState --> Termination
        Termination --> DoneOutput[Done Flag<br/>Boolean]
        
        EnvMethods[Environment Methods<br/>- reset<br/>- step<br/>- render<br/>- _get_obs]
    end
    
    subgraph DataFlow["Data Flow Between Components"]
        State[State<br/>Observation Vector]
        QValues[Q-Values<br/>5 action values]
        Action[Action<br/>Selected action<br/>0-4]
        Reward[Reward<br/>Float]
        NextState[Next State<br/>New observation]
        Transition[Transition<br/>state, action,<br/>reward, next_state, done]
    end
    
    %% Connections
    ObservationOutput -->|state| State
    State -->|input| PolicyNet
    PolicyNet -->|output| QValues
    QValues -->|select| EpsilonGreedy
    EpsilonGreedy -->|action| Action
    Action -->|input| ActionSpace
    RewardOutput --> Reward
    DoneOutput --> Transition
    ObservationOutput -->|next_state| NextState
    State -->|state| Transition
    Action -->|action| Transition
    Reward -->|reward| Transition
    NextState -->|next_state| Transition
    Transition -->|store| ReplayBuffer
    
    %% Styling
    style ModelArch fill:#E3F2FD,stroke:#1976D2,stroke-width:2px
    style AgentArch fill:#E8F5E9,stroke:#388E3C,stroke-width:2px
    style EnvArch fill:#FFEBEE,stroke:#C62828,stroke-width:2px
    style DataFlow fill:#FFF3E0,stroke:#F57C00,stroke-width:2px
    
    style PolicyNet fill:#4CAF50,stroke:#2E7D32,stroke-width:2px,color:#fff
    style TargetNet fill:#81C784,stroke:#2E7D32,stroke-width:2px,color:#fff
    style ReplayBuffer fill:#66BB6A,stroke:#2E7D32,stroke-width:2px,color:#fff
    style Optimizer fill:#A5D6A7,stroke:#2E7D32,stroke-width:2px
    style Input fill:#2196F3,stroke:#1565C0,stroke-width:2px,color:#fff
    style Output fill:#2196F3,stroke:#1565C0,stroke-width:2px,color:#fff
    style GameState fill:#EF5350,stroke:#C62828,stroke-width:2px,color:#fff
    style RewardSystem fill:#E57373,stroke:#C62828,stroke-width:2px,color:#fff
```
"""
    return mermaid_code

if __name__ == "__main__":
    # Generate visual workflow diagram
    fig = create_workflow_diagram()
    plt.savefig('workflow_diagram.png', dpi=300, bbox_inches='tight')
    print("[OK] Workflow diagram saved as 'workflow_diagram.png'")
    
    # Generate Mermaid workflow diagram code
    mermaid = create_mermaid_diagram()
    with open('workflow_diagram.mermaid.md', 'w', encoding='utf-8') as f:
        f.write(mermaid)
    print("[OK] Mermaid workflow diagram code saved as 'workflow_diagram.mermaid.md'")
    
    # Generate Mermaid architecture diagram code
    architecture = create_architecture_diagram()
    with open('architecture_diagram.mermaid.md', 'w', encoding='utf-8') as f:
        f.write(architecture)
    print("[OK] Mermaid architecture diagram code saved as 'architecture_diagram.mermaid.md'")
    
    print("\n" + "="*60)
    print("All Diagrams Generated Successfully!")
    print("="*60)
    print("\nGenerated Files:")
    print("  1. workflow_diagram.png - Visual workflow diagram (matplotlib)")
    print("  2. workflow_diagram.mermaid.md - Mermaid workflow diagram code")
    print("  3. architecture_diagram.mermaid.md - Mermaid architecture diagram code")
    print("\nKey Components:")
    print("  - Model (DQN): Neural network that predicts Q-values")
    print("  - Agent (DQNAgent): Manages model, replay buffer, and learning")
    print("  - Environment (CrossyEnv): Game simulation providing states/rewards")
    print("  - Train: Training loop that learns optimal policy")
    print("  - Test: Evaluation loop using trained model")
    print("  - dqn_crossy.pth: Saved model weights file")
    print("\nData Flow:")
    print("  State -> Agent -> Action -> Environment -> Reward -> Agent -> Update")
    print("="*60)

