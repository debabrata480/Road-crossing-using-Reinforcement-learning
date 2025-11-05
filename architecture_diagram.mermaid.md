
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
