
flowchart TB
    subgraph Row1[" "]
        direction LR
        A["Conveyor Objects (Bottles & Cans)"] -->|"Material flow: objects on belt"| B["Detect Object Presence & Estimate Location"]
        B -->|"Information flow: object candidate + pose estimate"| C["Classify Object & Compute Pick Target"]
    end

    subgraph Row2[" "]
        direction RL
        D["Schedule Pick Window & Generate Motion Command"] -->|"Information flow: joint references + timing"| E["Execute Pick & Transport"]
        E -->|"Material flow: picked object"| F["Place Object in Class Bin"]
        F -->|"Information flow: placement status"| G["Verify Pick and Placement"]
    end

    C -->|"Information flow: target pose + class + confidence"| D
    G -->|"Information flow: success/fault feedback"| D

    H["Power Source"] -->|"Energy flow: electrical power"| B
    H -->|"Energy flow: electrical power"| C
    H -->|"Energy flow: electrical power"| D
    H -->|"Energy flow: electrical/pneumatic power"| E
    H -->|"Energy flow: electrical power"| G

