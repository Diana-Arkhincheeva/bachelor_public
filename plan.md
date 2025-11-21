```mermaid
flowchart TD

    subgraph Part1["1.Optimierung von Δ"]
        A1[Drei mögliche Kalender-Parameter] --> B1[Generiere einen Zeitplan nach WSPT]
        B1 --> C1[10 Angriffe -> Mittelwert ε]
        C1 -->|ε > 0.5| B1[Suche einen Zeitplan]
        C1 -->|ε ≤ 0.5| E1[Wende Algorithmus an]

        E1 --> SA1[Simulated Annealing: 10 Iterationen]
        E1 --> BS1[Beam Search]
        E1 --> GA1[Genetischer Algorithmus]
    end

    subgraph Part2["2.Optimierung von ε"]
        F1[Drei mögliche Kalender-Parameter] --> G1[Generiere einen Zeitplan nach WSPT]
        G1 --> H1[10 Angriffe -> Mittelwert ε]
        H1 --> I1[Wende Algorithmus an]

        I1 --> SA3[Simulated Annealing: bei jedem Schritt 10 Angriffe → besten Zeitplan speichern]
        I1 --> BS3[Beam Search]
        I1 --> GA3[Genetischer Algorithmus]
    end

    Part1 --> J[Endanalyse: Welcher Algorithmus passt zu welchen Kalender-Parametern]
    Part2 --> J
