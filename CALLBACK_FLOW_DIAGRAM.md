# Visual Callback Flow Diagrams

## 🔄 **Complete Training Flow with Callback**

```mermaid
sequenceDiagram
    participant Agent as PPO Agent
    participant Env as Environment
    participant CB as Callback
    participant TB as TensorBoard
    participant Files as Plot Files

    Note over Agent,Files: Training Session Start
    Agent->>CB: _on_training_start()
    CB->>TB: Initialize SummaryWriter
    CB->>CB: Setup data storage

    loop Every Step
        Agent->>Env: Take action
        Env->>Agent: Return (obs, reward, done, info)
        Agent->>CB: _on_step(reward, info)
        
        CB->>CB: Extract metrics safely
        CB->>TB: Log real-time metrics
        CB->>CB: Store step data
        
        alt Episode Ends (done=True)
            CB->>CB: _log_episode_metrics()
            CB->>TB: Log episode summary
            CB->>CB: Reset episode tracking
        end
    end

    Note over Agent,Files: Training Complete
    Agent->>CB: _on_training_end()
    CB->>TB: Close writer
    CB->>Files: create_final_plots()
```

## 📊 **Data Flow Inside Callback**

```mermaid
graph TD
    A[Environment Info Dict] --> B{Safe Extraction}
    B --> C[phoneme: [2,5,1]]
    B --> D[reward: 0.85]
    B --> E[gini: 0.23]
    B --> F[action: 3]
    B --> G[energy: [0.8,0.3,0.9]]
    
    C --> H[Current Episode Data]
    D --> H
    E --> H
    F --> H
    G --> H
    
    H --> I[TensorBoard Logging]
    H --> J[Episode Summary Storage]
    
    I --> K[Real-time Graphs]
    J --> L[Final Plots]
```

## 🎯 **Step-by-Step Callback Execution**

```mermaid
flowchart TD
    Start([Training Starts]) --> Init[_on_training_start]
    Init --> Setup[Create TensorBoard Writer]
    Setup --> StepLoop{New Step?}
    
    StepLoop -->|Yes| OnStep[_on_step called]
    OnStep --> Extract[Extract metrics from info]
    Extract --> Validate[Validate data types]
    Validate --> Store[Store in episode_data]
    Store --> LogTB[Log to TensorBoard]
    LogTB --> CheckDone{Episode Done?}
    
    CheckDone -->|No| StepLoop
    CheckDone -->|Yes| EpisodeEnd[_log_episode_metrics]
    
    EpisodeEnd --> CalcStats[Calculate episode stats]
    CalcStats --> SaveEpisode[Save episode summary]
    SaveEpisode --> Reset[Reset episode tracking]
    Reset --> MemCheck[Check memory limit]
    MemCheck --> StepLoop
    
    StepLoop -->|Training Complete| TrainEnd[_on_training_end]
    TrainEnd --> CloseWriter[Close TensorBoard]
    CloseWriter --> CreatePlots[create_final_plots]
    CreatePlots --> End([Complete])
```

## 🧠 **Memory Management Strategy**

```mermaid
graph LR
    A[New Episode Data] --> B{episodes_step_data length}
    B -->|< max_stored_episodes| C[Add to list]
    B -->|≥ max_stored_episodes| D[Remove oldest episode]
    D --> E[Add new episode]
    C --> F[Continue training]
    E --> F
    
    subgraph Memory Limit
        G[Episode 1 - REMOVED]
        H[Episode 2]
        I[Episode 3]
        J[...]
        K[Episode N - NEW]
    end
```

## 📈 **Data Types and Structure**

```mermaid
classDiagram
    class CallbackPerEpisode {
        +episode_rewards: List[float]
        +episode_phonemes: List[List[int]]
        +episode_gini: List[float]
        +current_episode_data: Dict
        +writer: SummaryWriter
        
        +_on_step() bool
        +_log_episode_metrics()
        +_safe_get_metric()
        +create_final_plots()
    }
    
    class CurrentEpisodeData {
        +rewards: List[float]
        +env_rewards: List[float]
        +phonemes: List[List[int]]
        +gini: List[float]
        +actions: List[int]
        +energy: List[List[float]]
    }
    
    class EnvironmentInfo {
        +phoneme: List[int]
        +current_gini: float
        +env_reward: float
        +total_reward: float
        +action_number: int
        +energy: List[float]
    }
    
    CallbackPerEpisode --> CurrentEpisodeData
    CallbackPerEpisode --> EnvironmentInfo
```

## 🎨 **Plot Generation Process**

```mermaid
graph TD
    A[create_final_plots called] --> B[Create results directory]
    B --> C[Determine episodes to plot]
    C --> D{For each episode}
    
    D --> E[_create_episode_plot]
    E --> F[Extract episode data]
    F --> G[Create 4x3 subplot grid]
    G --> H[Plot rewards over steps]
    H --> I[Plot phonemes per agent]
    I --> J[Plot action counts]
    J --> K[Plot energy levels]
    K --> L[Save episode plot]
    
    L --> M{More episodes?}
    M -->|Yes| D
    M -->|No| N[_create_summary_plots]
    
    N --> O[Plot episode rewards trend]
    O --> P[Plot phoneme evolution]
    P --> Q[Calculate balance metrics]
    Q --> R[Plot Gini coefficients]
    R --> S[Save summary plots]
    S --> T[Complete]
```

## 🔍 **Error Handling Flow**

```mermaid
graph TD
    A[Data Access Attempt] --> B{Try to get metric}
    B -->|Success| C[Return actual value]
    B -->|KeyError| D[Use default value]
    B -->|TypeError| E[Log warning + use default]
    B -->|Other Exception| F[Log error + use default]
    
    D --> G[Continue execution]
    E --> G
    F --> G
    C --> G
    
    G --> H{Critical for training?}
    H -->|No| I[Graceful degradation]
    H -->|Yes| J[Ensure training continues]
```

## 🚀 **Performance Optimization Points**

```mermaid
mindmap
  root((Callback Performance))
    Memory Management
      Max stored episodes
      Data cleanup
      Efficient storage
    Logging Frequency
      TensorBoard batching
      Conditional logging
      Reduced I/O
    Data Processing
      Safe extraction
      Type validation
      Efficient arrays
    Plot Generation
      Lazy creation
      Configurable episodes
      Memory efficient
```

## 🎯 **Key Callback Features**

```mermaid
graph LR
    A[CallbackPerEpisode] --> B[Dynamic Agent Support]
    A --> C[Dual Reward Tracking]
    A --> D[Memory Management]
    A --> E[Error Handling]
    A --> F[Real-time Logging]
    A --> G[Detailed Plotting]
    
    B --> B1[Works with any number of agents]
    C --> C1[PPO rewards + Environment rewards]
    D --> D1[Prevents memory leaks]
    E --> E1[Graceful failure handling]
    F --> F1[TensorBoard integration]
    G --> G1[Episode + Summary plots]
```

These diagrams show exactly how your callback works at every level - from the high-level training flow down to the specific data structures and error handling mechanisms!