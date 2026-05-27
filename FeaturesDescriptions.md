# The 12 features in plain English

| Index | Feature                            | Meaning                                                                  |
| ----: | ---------------------------------- | ------------------------------------------------------------------------ |
|     0 | `forward_speed_norm`               | How fast the car is moving forward/reversing relative to its own heading |
|     1 | `lateral_speed_norm`               | How much the car is sliding sideways                                     |
|     2 | `vertical_speed_norm`              | How much the car is moving up/down                                       |
|     3 | `heading_error_norm`               | How misaligned the car is with the current track direction               |
|     4 | `lateral_error_norm`               | How far left/right the car is from the centreline                        |
|     5 | `lookahead_heading_error_20m_norm` | Angle from car heading to the track point 20m ahead                      |
|     6 | `lookahead_heading_error_40m_norm` | Angle from car heading to the track point 40m ahead                      |
|     7 | `lookahead_heading_error_80m_norm` | Angle from car heading to the track point 80m ahead                      |
|     8 | `curvature_20m_norm`               | Track bend/sharpness 20m ahead                                           |
|     9 | `curvature_40m_norm`               | Track bend/sharpness 40m ahead                                           |
|    10 | `curvature_80m_norm`               | Track bend/sharpness 80m ahead                                           |
|    11 | `progress_ratio`                   | Where the car is around the lap, from 0 to 1                             |

---
